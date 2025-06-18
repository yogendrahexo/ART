from art_e.rollout import ProjectTrajectory
from typing import List, cast
import json
from litellm import acompletion
from textwrap import dedent
import tenacity
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field


class JudgeGroupScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")


class JudgeGroupResponse(BaseModel):
    scores: List[JudgeGroupScore]


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
async def judge_group(
    rollouts: list[ProjectTrajectory],
    *,
    debug: bool = False,
) -> list[ProjectTrajectory]:
    """Judge a list of trajectories with an LLM-as-a-judge.

    This keeps the original trajectories but overwrites ``reward`` with the
    score returned by the judge (0–1).  The original reward is preserved in
    ``traj.metrics['independent_reward']`` and the new score is written to
    ``traj.metrics['judge_group_reward']``.
    """

    if not rollouts:
        return rollouts

    # Serialize each rollout's messages (keeping tool_calls as-is)
    serialized_rollouts: List[str] = []
    # Keep structured messages for nicer debug printing
    debug_rollouts: List[tuple[int, list]] = [] if debug else []
    for idx, traj in enumerate(rollouts, start=1):
        # Save the original reward
        traj.metrics["independent_reward"] = traj.reward
        # Flatten messages to regular OpenAI format (role/content/…)
        messages = traj.messages()
        if debug:
            debug_rollouts.append((idx, messages))
        serialized_rollouts.append(
            f'<rollout id="{idx}">\n' + json.dumps(messages) + "\n</rollout>"
        )

    if debug:
        print("\n[judge_group] Serialized rollouts (pretty JSON):")
        for idx, msg_list in debug_rollouts:
            print(f"\nRollout {idx}:")
            print(json.dumps(msg_list, indent=2, ensure_ascii=False))

        print("\n[judge_group] Rollout metrics:")
        for idx, traj in enumerate(rollouts, start=1):
            print(f"\nRollout {idx} metrics:")
            print(json.dumps(traj.metrics, indent=2, ensure_ascii=False))

    rubric_text = dedent(
        """
        All of the rollouts below have been given the same task. Your job is to consider each of them and give them a score between 0 and 1.

        Rubric:
        - A rollout that gets the answer wrong should always get a lower score than a rollout that gets the answer right.
        - A rollout that takes more turns to get the right answer should always get a lower score than a rollout that takes fewer turns to get the right answer.
        - A rollout that gives an incorrect answer should always get a lower score than a rollout that says 'I don't know'.
        - A rollout that gets the answer right but doesn't cite a relevant source should always get a lower score than a rollout that cites a relevant source.

        Always return your scores in the same order as the rollouts.
        """
    )

    user_text = "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

    messages = [
        {"role": "system", "content": rubric_text},
        {"role": "user", "content": user_text},
    ]

    response = await acompletion(
        model="openai/o3",
        messages=messages,
        response_format=JudgeGroupResponse,
        caching=True,
    )

    first_choice = response.choices[0]  # type: ignore[attr-defined]

    if debug:
        raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        print("\n[judge_group] Raw LLM choice content:")
        print(raw_content)

        try:
            print("\n[judge_group] Pretty-printed LLM choice JSON:")
            print(json.dumps(json.loads(raw_content), indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"[judge_group] Could not parse choice content as JSON: {e}")

    content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    parsed = JudgeGroupResponse.model_validate_json(content)
    assert len(parsed.scores) == len(rollouts)

    for idx, (traj, score) in enumerate(zip(rollouts, parsed.scores)):
        traj.reward = score.score
        traj.metrics["judge_group_reward"] = traj.reward
        traj.log(f"Judge group explanation: {score.explanation}")

    return rollouts


if __name__ == "__main__":
    import asyncio

    async def main():
        """Run a quick smoke-test: generate one rollout per model and judge them."""
        from dotenv import load_dotenv
        import art
        from art_e.project_types import ProjectPolicyConfig
        from art_e.data.query_iterators import load_synthetic_queries
        from art_e.rollout import rollout

        load_dotenv()

        MODEL_CONFIGS = [
            "openai/gpt-4o",
            "openai/gpt-4.1",
            "openai/o4-mini",
            "openai/o3",
        ]

        # Create the four models we want to benchmark.
        models: list[art.Model] = []
        for litellm_name in MODEL_CONFIGS:
            models.append(
                art.Model(
                    name=litellm_name,
                    project="email_agent",
                    config=ProjectPolicyConfig(
                        litellm_model_name=litellm_name,
                    ),
                )
            )

        # Use the same scenario for every model so the judge can compare rollouts fairly.
        scenario = load_synthetic_queries(split="test", limit=1)[0]

        rollouts = await tqdm.gather(*[rollout(model, scenario) for model in models])

        print("Independent rewards (before judging):")
        for m, t in zip(models, rollouts):
            print(f"  {m.name:10s}: {t.reward:.3f}")

        # Judge the group of rollouts.
        judged_rollouts = await judge_group(rollouts, debug=True)

        print("\nJudge-group rewards:")
        for m, t in zip(models, judged_rollouts):
            print(f"  {m.name:10s}: {t.reward:.3f}")

    asyncio.run(main())
