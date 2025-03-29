import dataclasses
from pydantic import BaseModel
from typing import Literal, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from trl import GRPOConfig

    from .. import types


def get_model_config(
    base_model: "types.BaseModel", output_dir: str, config: "ModelConfig | None"
) -> "ModelConfig":
    if config is None:
        config = ModelConfig()
    init_args = InitArgs(
        model_name=base_model,
        max_seq_length=8192,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_requests=True,
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.62,  # Reduce if out of memory
        max_lora_rank=32,
        num_scheduler_steps=16,
        use_async=True,
    )
    init_args.update(config.init_args or {})
    peft_args = PeftArgs(
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=32,
        # Enable long context finetuning
        use_gradient_checkpointing="unsloth",  # type: ignore
        random_state=3407,
    )
    peft_args.update(config.peft_args or {})
    train_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        save_strategy="no",
        output_dir=output_dir,
    )
    train_args = dataclasses.replace(
        train_args, **dataclasses.asdict(config.train_args or GRPOConfig())
    )
    # TODO: Add base model conditional configuration
    return ModelConfig(init_args=init_args, peft_args=peft_args, train_args=train_args)


class ModelConfig(BaseModel):
    """
    Model configuration.

    Args:
        init: Arguments for initializing an Unsloth FastLanguageModel.
        peft: Arguments for creating an Unsloth PEFT model wrapper.
        train: Arguments for training the model.
    """

    init_args: "InitArgs | None" = None
    peft_args: "PeftArgs | None" = None
    train_args: "GRPOConfig | None" = None


class InitArgs(TypedDict, total=False):
    model_name: str
    max_seq_length: int
    dtype: str | None
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    token: str | None
    device_map: str
    rope_scaling: dict | None
    fix_tokenizer: bool
    trust_remote_code: bool
    use_gradient_checkpointing: str
    resize_model_vocab: int | None
    revision: str | None
    use_exact_model_name: bool
    fast_inference: bool
    gpu_memory_utilization: float
    float8_kv_cache: bool
    random_state: int
    max_lora_rank: int
    disable_log_requests: bool
    disable_log_stats: bool
    enable_prefix_caching: bool
    num_scheduler_steps: int
    use_async: bool


class PeftArgs(TypedDict, total=False):
    r: int
    target_modules: list[str]
    lora_alpha: int
    lora_dropout: float
    bias: str
    layers_to_transform: list[int] | None
    layers_pattern: str | None
    use_gradient_checkpointing: bool | str
    random_state: int
    max_seq_length: int  # not used anymore
    use_rslora: bool
    modules_to_save: list[str] | None
    init_lora_weights: bool
    loftq_config: dict
    temporary_location: str
