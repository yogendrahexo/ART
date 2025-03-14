import httpx

from .model import Model
from .trajectory import Trajectory
from .types import BaseModel, TuneConfig, Verbosity


class API:
    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
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
            async with model.openai_client() as client:
                chat_completion = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": "Hello, world!"}],
                )
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
