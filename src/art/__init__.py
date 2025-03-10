from .model import Model
from .openai import AsyncOpenAI
from .types import Messages, Trajectory

__all__ = ["Messages", "Model", "Trajectory"]


client = AsyncOpenAI()


async def get_or_create_model(name: str, base_model: str) -> Model:
    return Model(name=name, base_model=base_model)
