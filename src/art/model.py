from contextlib import asynccontextmanager
from dataclasses import dataclass
from openai import AsyncOpenAI
from typing import AsyncGenerator, TYPE_CHECKING

# from .openai import AsyncOpenAI
from .types import Trajectory, Verbosity

if TYPE_CHECKING:
    from .api import API


@dataclass
class Model:
    api: "API"
    name: str
    base_model: str

    @asynccontextmanager
    async def openai_client(
        self, verbosity: Verbosity = 2
    ) -> AsyncGenerator[AsyncOpenAI, None]:
        client = await self.api._get_openai_client(self, verbosity)
        try:
            yield client
        finally:
            await self.api._close_openai_client(client)

    @property
    def iteration(self) -> int:
        return 0

    async def save_eval(self, trajectory_groups: list[list[Trajectory]]) -> None:
        await self.api._save_eval(self, trajectory_groups)

    async def tune(
        self, trajectory_groups: list[list[Trajectory]], verbosity: Verbosity = 2
    ) -> None:
        await self.api._tune_model(self, trajectory_groups, verbosity)
