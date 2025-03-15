from contextlib import asynccontextmanager
from dataclasses import dataclass
from openai import AsyncOpenAI
from typing import AsyncGenerator, TYPE_CHECKING

from .openai import patch_openai
from .types import BaseModel, Trajectory, TuneConfig, Verbosity

if TYPE_CHECKING:
    from .api import API


@dataclass
class Model:
    api: "API"
    name: str
    base_model: BaseModel

    @asynccontextmanager
    async def openai_client(
        self,
        estimated_completion_tokens: int = 1024,
        tool_use: bool = False,
        verbosity: Verbosity = 1,
    ) -> AsyncGenerator[AsyncOpenAI, None]:
        """
        Context manager for an OpenAI client to a managed inference service.

        Args:
            estimated_completion_tokens: Estimated completion tokens per request.
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
        client, semaphore = await self.api._get_openai_client(
            self, estimated_completion_tokens, tool_use, verbosity
        )
        try:
            yield patch_openai(client, semaphore)
        finally:
            await self.api._close_openai_client(client)

    async def get_iteration(self) -> int:
        """
        Get the model's current training iteration.
        """
        return await self.api._get_iteration(self)

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
        await self.api._clear_iterations(self, benchmark, benchmark_smoothing)

    async def save(
        self,
        trajectory_groups: list[list[Trajectory | BaseException]],
        name: str = "val",
    ) -> None:
        """
        Save the model's performance on an evaluation batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            name: The evaluation's name. Defaults to "val".
        """
        await self.api._save(self, trajectory_groups, name)

    async def tune(
        self,
        trajectory_groups: list[list[Trajectory | BaseException]],
        config: TuneConfig = TuneConfig(),
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration with options for the optimizer,
                loss, other hyperparameters, logging, etc.
        """
        await self.api._tune_model(self, trajectory_groups, config)
