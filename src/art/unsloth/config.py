from pydantic import BaseModel
from typing import Any, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .vllm import ServerArgs


class ModelConfig(BaseModel):
    """
    Model configuration.

    Args:
        init: Arguments for initializing an Unsloth FastLanguageModel.
        peft: Arguments for creating an Unsloth PEFT model wrapper.
        openai_server: vLLM OpenAI-compatible server configuration.
        engine_args: Additional vLLM engine arguments for the OpenAI-compatible server.
                     Note that since the vLLM engine will be initialized with Unsloth,
                     these additional arguments will only have an effect if the
                     OpenAI-compatible server uses them elsewhere.
    """

    init: "InitArgs"
    peft: "PeftArgs"
    openai_server: "ServerArgs"
    engine_args: dict[str, Any]


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
    disable_log_stats: bool


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
