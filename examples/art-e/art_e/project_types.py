from pydantic import BaseModel
from typing import Literal


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    use_judge_group_variant: Literal["v1"] | Literal["v2"] | None = (
        None  # e.g., "v1", "v2"; None disables judge-group rescoring
    )
    # Model name to use for judge-group rescoring (LLM-as-a-judge). Defaults to
    # OpenAI's o3 model.  You can override this per-training run.
    judge_group_model_name: str = "openai/o3"
    minimum_reward_std_dev: float = 0.0
    # Random seed to control which subset of the training data is sampled. When None, the sampler can
    # choose its own default (e.g., derive from the current time).
    training_dataset_seed: int | None = None
    allow_training_without_logprobs: bool = False


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    log_to_openpipe: bool = False
    litellm_model_name: str | None = None
    stupid_simple_reward_fn: bool = False

    training_config: TrainingConfig | None = None
