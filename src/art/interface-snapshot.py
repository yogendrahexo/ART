# Interface as of 2025-03-14 @ 10:40am Mountain Time
from contextlib import asynccontextmanager
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai import AsyncOpenAI
from typing import AsyncGenerator, Iterable, Literal


BaseModel = Literal[
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "NousResearch/Hermes-2-Theta-Llama-3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "unsloth/Llama-3.3-70B-Instruct",
]

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = Iterable[ChatCompletionToolParam]


class Trajectory:
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float]
    tools: Tools | None


class TuneConfig:
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


class API:
    async def get_or_create_model(self, name: str, base_model: BaseModel) -> "Model":
        """
        Retrieves an existing model or creates a new one.

        Args:
            name: The model's name.
            base_model: The model's base model.

        Returns:
            Model: A model instance.
        """
        ...


class Model:
    @asynccontextmanager
    async def openai_client(
        self,
        estimated_token_usage: int = 1024,
        tool_use: bool = False,
        verbosity: Verbosity = 1,
    ) -> AsyncGenerator[AsyncOpenAI, None]:
        """
        Context manager for an OpenAI client to a managed inference service.

        Args:
            estimated_token_usage: Estimated token usage per request.
            tool_use: Whether to enable tool use.
            verbosity: Verbosity level.

        Yields:
            AsyncOpenAI: An asynchronous OpenAI client.

        Example:
            ```python
            async with model.openai_client() as client:
                chat_completion = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": "Hello, world!"}],
                )
            ```
        """
        ...

    async def get_iteration(self) -> int:
        """
        Get the model's current training iteration.
        """
        ...

    async def clear_iterations(
        self, benchmark: str = "val/reward", benchmark_smoothing: float = 1.0
    ) -> None:
        """
        Delete all but the latest and best iteration checkpoints.

        Args:
            benchmark: The benchmark to use to determine the best iteration.
            benchmark_smoothing: Smoothing factor (0-1) that controls how much to reduce
                variance when determining the best iteration. Defaults to 1.0 (no smoothing).
        """
        ...

    async def save(
        self, trajectory_groups: list[list[Trajectory]], name: str = "val"
    ) -> None:
        """
        Save the model's performance on an evaluation batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            name: The evaluation's name. Defaults to "val".
        """
        ...

    async def tune(
        self,
        trajectory_groups: list[list[Trajectory]],
        config: TuneConfig = TuneConfig(),
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration with options for the optimizer,
                loss, other hyperparameters, logging, etc.
        """
        ...
