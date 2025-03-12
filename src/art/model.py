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
        self, estimated_token_usage: int = 1024, verbosity: Verbosity = 1
    ) -> AsyncGenerator[AsyncOpenAI, None]:
        client, semaphore = await self.api._get_openai_client(
            self, estimated_token_usage, verbosity
        )
        try:
            yield patch_openai(client, semaphore)
        finally:
            await self.api._close_openai_client(client)

    async def get_iteration(self) -> int:
        return await self.api._get_iteration(self)

    async def save_eval(self, trajectory_groups: list[list[Trajectory]]) -> None:
        await self.api._save_eval(self, trajectory_groups)

    async def tune(
        self,
        trajectory_groups: list[list[Trajectory]],
        config: TuneConfig = TuneConfig(),
    ) -> None:
        await self.api._tune_model(self, trajectory_groups, config)
