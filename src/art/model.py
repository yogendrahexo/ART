from dataclasses import dataclass

from .openai import AsyncOpenAI
from .types import Trajectory


@dataclass
class Model:
    name: str
    base_model: str

    @property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    @property
    def iteration(self) -> int:
        return 0

    async def put_eval(self, trajectory_groups: list[list[Trajectory]]) -> None: ...

    async def tune(self, trajectory_groups: list[list[Trajectory]]) -> None: ...
