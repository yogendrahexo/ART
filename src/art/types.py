from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import pydantic
from typing import Literal

BaseModel = Literal[
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}


class TuneConfig(pydantic.BaseModel):
    learning_rate_multiplier: float = 1.0  # REMOVE?
    beta: float = 0.0

    # GRPO params
    clip_epsilon: float = 0.2  # RENAME?
    kl_coef: float = pydantic.Field(
        default=0.0,
        deprecated="`kl_coef` is deprecated, use `beta` instead.",
    )

    # Optimizer params
    lr: float = pydantic.Field(
        default=5e-6,
        deprecated="`lr` is deprecated, use `learning_rate_multiplier` instead.",
    )  # KEEP?
    betas: tuple[float, float] = (0.9, 0.99)  # REMOVE?
    weight_decay: float = 0.1  # REMOVE?

    # Tensor packing params
    sequence_length: int | None = pydantic.Field(
        default=None,
        deprecated="Sequence length is now automatically determined from trajectory data.",
    )

    # Logging params
    plot_tensors: bool = False  # REMOVE?
    verbosity: "Verbosity" = 1  # REMOVE?


Verbosity = Literal[0, 1, 2]
