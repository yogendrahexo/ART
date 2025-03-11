from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
import pydantic
from typing import Literal

BaseModel = Literal[
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "NousResearch/Hermes-2-Theta-Llama-3-8B",
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "unsloth/Llama-3.3-70B-Instruct",
]

Message = ChatCompletionMessageParam  # | Choice
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]


class Trajectory(pydantic.BaseModel):
    messages: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}


class TuneConfig(pydantic.BaseModel):
    # GRPO params
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    kl_coef: float = 0.0
    tanh: bool = False

    # Optimizer params
    lr: float = 6e-6
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1

    # Tensor packing params
    sequence_length: int = 16_384

    # Logging params
    plot_tensors: bool = False
    verbosity: "Verbosity" = 1


Verbosity = Literal[0, 1, 2]
