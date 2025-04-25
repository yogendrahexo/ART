from openai import AsyncOpenAI
from typing import cast, Iterable, TYPE_CHECKING, Optional

from . import dev
from .openai import patch_openai
from .trajectories import Trajectory, TrajectoryGroup
from .types import TrainConfig
from pydantic import BaseModel
from openai import (
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
import httpx

from openai import OpenAI

client = OpenAI()

client.base_url

if TYPE_CHECKING:
    from .local.api import LocalAPI

# ---------------------------------------------------------------------------
# Inference configuration (new API)
# ---------------------------------------------------------------------------
#
# Historically ART stored inference connection information inside an
# `InferenceConfig` object that lived on each `art.Model`.  After some user
# feedback we found that this extra layer of indirection was confusing and made
# it harder to grok a model at a glance.  From now on the three relevant pieces
# of information live directly on the model instance:
#
#     inference_api_key   – The API key used to authenticate with the
#                           OpenAI-compatible inference endpoint.
#     inference_base_url  – The base URL (ending in `/v1`) of the endpoint.
#     inference_model_name – (optional) If provided, this model name will be
#                           sent to the endpoint instead of `Model.name`.
#
# You can now instantiate a prompted model like so:
#
#     model = art.Model(
#         name="gpt-4.1",
#         project="my-project",
#         inference_api_key=os.getenv("OPENAI_API_KEY"),
#         inference_base_url="https://api.openai.com/v1/",
#     )
#
# Or, if you're pointing at OpenRouter:
#
#     model = art.Model(
#         name="gemini-2.5-pro",
#         project="my-project",
#         inference_api_key=os.getenv("OPENROUTER_API_KEY"),
#         inference_base_url="https://openrouter.ai/api/v1",
#         inference_model_name="google/gemini-2.5-pro-preview-03-25",
#     )
#
# For trainable (`art.TrainableModel`) models these values will be populated
# automatically by `model.register(api)` so you generally don't need to think
# about them.
# ---------------------------------------------------------------------------


class Model(BaseModel):
    name: str
    project: str

    config: BaseModel | None = None

    # --- Inference connection information (populated automatically for
    #     TrainableModel or set manually for prompted / comparison models) ---

    inference_api_key: str | None = None
    inference_base_url: str | None = None
    # If set, this will be used instead of `self.name` when calling the
    # inference endpoint.
    inference_model_name: str | None = None

    _api: Optional["LocalAPI"] = None
    _s3_bucket: str | None = None
    _s3_prefix: str | None = None

    def api(self) -> "LocalAPI":
        if self._api is None:
            raise ValueError(
                "Model is not registered with the API. You must call `model.register()` first."
            )
        return self._api

    async def register(self, api: "LocalAPI") -> None:
        if self.config is not None:
            try:
                self.config.model_dump_json()
            except Exception as e:
                raise ValueError(
                    "The model config cannot be serialized to JSON. Please ensure that all fields are JSON serializable and try again."
                ) from e

        self._api = api
        await self._api.register(self)

    def openai_client(
        self,
    ) -> AsyncOpenAI:
        if self.inference_api_key is None or self.inference_base_url is None:
            raise ValueError(
                "OpenAI client not yet available. You must call `model.register()` first."
            )
        openai_client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=1200, connect=5.0),
                limits=httpx.Limits(
                    max_connections=100_000, max_keepalive_connections=100_000
                ),
            ),
        )
        patch_openai(openai_client)
        return openai_client

    # ------------------------------------------------------------------
    # Inference name helpers
    # ------------------------------------------------------------------

    def get_inference_name(self) -> str:
        """Return the name that should be sent to the inference endpoint.

        If ``inference_model_name`` is provided we use that, otherwise we fall
        back to the model's own ``name``.
        """
        return self.inference_model_name or self.name

    async def log(
        self,
        trajectories: Iterable[Trajectory] | Iterable[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectories or trajectory groups.

        Args:
            trajectories: A batch of trajectories or trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        if any(isinstance(t, Trajectory) for t in trajectories):
            trajectory_groups = [
                TrajectoryGroup(cast(Iterable[Trajectory], trajectories))
            ]
        else:
            trajectory_groups = cast(list[TrajectoryGroup], list(trajectories))
        await self.api()._log(
            self,
            trajectory_groups,
            split,
        )


# ---------------------------------------------------------------------------
# Trainable models
# ---------------------------------------------------------------------------


class TrainableModel(Model):
    base_model: str
    trainable: bool = True

    # The fields within `_internal_config` are unstable and subject to change.
    # Use at your own risk.
    _internal_config: dev.InternalModelConfig | None = None

    async def register(
        self,
        api: "LocalAPI",
        _openai_client_config: dev.OpenAIServerConfig | None = None,
    ) -> None:
        await super().register(api)
        base_url, api_key = await api._prepare_backend_for_training(
            self, _openai_client_config
        )

        # Populate the new top-level inference fields so that the rest of the
        # code (and any user code) can create an OpenAI client immediately.
        self.inference_base_url = base_url
        self.inference_api_key = api_key
        self.inference_model_name = self.name

    async def get_step(self) -> int:
        """
        Get the model's current training step.
        """
        return await self.api()._get_step(self)

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        await self.api()._delete_checkpoints(self, best_checkpoint_metric)

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        await self.api()._train_model(
            self, list(trajectory_groups), config, _config or {}
        )
