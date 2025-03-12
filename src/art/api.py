import asyncio
import httpx
from openai import AsyncOpenAI
import os

from .model import Model
from .types import BaseModel, Trajectory, TuneConfig, Verbosity


class API:
    def __init__(self, *, base_url: str | httpx.URL | None = None) -> None:
        if base_url is None:
            base_url = os.environ.get("ART_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openpipe.ai/art/v1"
        self._client = httpx.AsyncClient(base_url=base_url)

    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
        response = await self._client.post(
            "/models",
            json={"name": name, "base_model": base_model},
        )
        response.raise_for_status()
        return Model(api=self, name=name, base_model=base_model)

    async def _get_iteration(self, model: Model) -> int:
        response = await self._client.get(
            f"/models/{model.name}/iteration",
        )
        response.raise_for_status()
        return response.json()["iteration"]

    async def _get_openai_client(
        self, model: Model, estimated_token_usage: int, verbosity: Verbosity
    ) -> tuple[AsyncOpenAI, asyncio.Semaphore]:
        response = await self._client.post(
            f"/openai_clients", json={"model": model.name}
        )
        response.raise_for_status()
        data = response.json()
        return AsyncOpenAI(**data["client_kwargs"]), asyncio.Semaphore(
            data["max_concurrent_requests"]
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        response = await self._client.post(
            f"/openai_clients/close",
            json={
                "api_key": client.api_key,
                "organization": client.organization,
                "project": client.project,
                "base_url": client.base_url,
            },
        )
        response.raise_for_status()

    async def _save_eval(
        self, model: Model, trajectory_groups: list[list[Trajectory]]
    ) -> None:
        response = await self._client.post(
            "/evals",
            json={
                "model": model.name,
                "trajectory_groups": trajectory_groups,
            },
        )
        response.raise_for_status()

    async def _tune_model(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory]],
        config: TuneConfig,
    ) -> None:
        response = await self._client.post(
            "/models/tune",
            json={
                "model": model.name,
                "trajectory_groups": trajectory_groups,
            },
        )
        response.raise_for_status()
