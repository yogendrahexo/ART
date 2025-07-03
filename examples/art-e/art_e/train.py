import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List, cast
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig
from art_e.evaluate.benchmark import benchmark_model
from art_e.judge_group import judge_group
from art_e.rollout import ProjectTrajectory
import os
import statistics
import tenacity

load_dotenv()


# Retry up to 5 times if the training fails, usually happens because vllm dies
@tenacity.retry(stop=tenacity.stop_after_attempt(5))
async def train(model: art.TrainableModel[ProjectPolicyConfig]):
    generate_database()

    if model.config.training_config is None:
        raise ValueError("Training config is not set")
    with LocalBackend() as backend:
        print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
        )
        await model.register(backend)

        print("Loading training data...")
        # Load the training data with deterministic shuffling if a seed is provided.
        tc = model.config.training_config
        seed = tc.training_dataset_seed if tc is not None else None
        train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="train",
            limit=tc.training_dataset_size if tc is not None else None,
            seed=seed,
        )
        print("Loading validation data...")
        val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="test", limit=model.config.training_config.val_set_size
        )

        print(f"Training data size: {len(train_scenarios)}")
        print(f"Validation data size: {len(val_scenarios)}")

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=model.config.training_config.groups_per_step,
            num_epochs=model.config.training_config.num_epochs,
            initial_step=await model.get_step(),
        )

        for batch, epoch, global_step, epoch_step in train_iterator:
            if global_step % model.config.training_config.eval_steps == 0:
                print(f"\n--- Evaluating at Iteration {global_step} ---")
                await benchmark_model(model)
                await model.delete_checkpoints()
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(
                                model.config.training_config.trajectories_per_group
                            )
                        )
                    )
                    for scenario in batch
                )
            )

            # Optionally rescore each trajectory group with the LLM-judge before training.
            training_cfg = model.config.training_config
            if training_cfg.use_judge_group_variant is not None:
                # Run the rescoring concurrently for better throughput.
                # If any call to `judge_group` fails, we don't want the entire
                # training run to crash. We therefore collect exceptions and
                # log a clear warning that can be grepped later.

                judge_tasks = [
                    judge_group(
                        model.name,
                        cast(list[ProjectTrajectory], g.trajectories),
                        training_cfg,
                    )
                    for g in groups
                ]

                results = await asyncio.gather(*judge_tasks, return_exceptions=True)

                # Determine which groups succeeded.
                successful_groups = []
                for grp_idx, (g, res) in enumerate(zip(groups, results)):
                    if isinstance(res, Exception):
                        print(
                            f"WARNING:JUDGE_GROUP_FAILED group={grp_idx} step={global_step}: {res!r}",
                            flush=True,
                        )
                    else:
                        successful_groups.append(g)

                # Replace `groups` with the subset that passed judgement so
                # that training only uses trajectories with judge rewards.
                groups = successful_groups

                # If every group failed, skip this training step entirely.
                if not groups:
                    print(
                        f"WARNING:ALL_JUDGE_GROUPS_FAILED step={global_step}; skipping training step",
                        flush=True,
                    )
                    continue  # Proceed to next batch/epoch without training.

            # Drop groups with reward standard deviation below threshold
            if (
                training_cfg.minimum_reward_std_dev is not None
                and training_cfg.minimum_reward_std_dev > 0
            ):
                filtered_groups = []
                for grp_idx, g in enumerate(groups):
                    rewards = [t.reward for t in g.trajectories]
                    if len(rewards) < 2:
                        std_dev = 0.0
                    else:
                        std_dev = statistics.pstdev(rewards)
                    if std_dev < training_cfg.minimum_reward_std_dev:
                        print(
                            f"WARNING:REWARD_STD_DEV_TOO_LOW group={grp_idx} step={global_step} stddev={std_dev:.4f}; dropping group",
                            flush=True,
                        )
                        continue
                    filtered_groups.append(g)

                # Replace groups with only those meeting the std dev threshold
                groups = filtered_groups

                # If every group failed the std dev filter, skip this training step
                if not groups:
                    print(
                        f"WARNING:ALL_GROUPS_DROPPED_LOW_STD_DEV step={global_step}; skipping training step",
                        flush=True,
                    )
                    continue  # Proceed to next batch/epoch without training.

            await model.train(
                groups,
                config=art.TrainConfig(
                    learning_rate=model.config.training_config.learning_rate
                ),
                _config={"allow_training_without_logprobs": True}
            )

        await benchmark_model(model)
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )
        print("Training finished.")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model_dict["config"] = ProjectPolicyConfig(**model_dict["config"])
    model: art.TrainableModel[ProjectPolicyConfig] = art.TrainableModel(
        **model_dict,
    )
    asyncio.run(train(model))
