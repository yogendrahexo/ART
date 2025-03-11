from .gather_groups import gather_groups
from .model import Model
from .openai import AsyncOpenAI
from .types import Messages, Trajectory

__all__ = ["gather_groups", "Messages", "Model", "Trajectory"]


class Client:
    openai = AsyncOpenAI(api_key="")

    async def get_or_create_model(self, name: str, base_model: str) -> Model:
        return Model(name=name, base_model=base_model)
