import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel
from typing import TYPE_CHECKING, cast, Generic, Iterable, Optional, overload, TypeVar
from typing_extensions import Never

from . import dev
from .openai import patch_openai
from .trajectories import Trajectory, TrajectoryGroup
from .types import TrainConfig

if TYPE_CHECKING:
    from art.backend import Backend


ModelConfig = TypeVar("ModelConfig", bound=BaseModel | None)


class Model(
    BaseModel,
    Generic[ModelConfig],
):
    """
    A model is an object that can be passed to your `rollout` function, and used
    to log completions. Additionally, a `TrainableModel`, which is a subclass of
    `Model`, can be used to train a model.

    The `Model` abstraction is useful for comparing prompted model performance
    to the performance of your trained models.

    You can instantiate a prompted model like so:

    ```python model = art.Model(
        name="gpt-4.1", project="my-project",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )
    ```

    Or, if you're pointing at OpenRouter:

    ```python model = art.Model(
        name="gemini-2.5-pro", project="my-project",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
        inference_model_name="google/gemini-2.5-pro-preview-03-25",
    )
    ```

    For trainable (`art.TrainableModel`) models the inference values will be
    populated automatically by `model.register(api)` so you generally don't need
    to think about them.
    """

    name: str
    project: str
    config: ModelConfig

    # --- Inference connection information (populated automatically for
    #     TrainableModel or set manually for prompted / comparison models) ---
    inference_api_key: str | None = None
    inference_base_url: str | None = None
    # If set, this will be used instead of `self.name` when calling the
    # inference endpoint.
    inference_model_name: str | None = None
    api_version: str | None = None

    _backend: Optional["Backend"] = None
    _s3_bucket: str | None = None
    _s3_prefix: str | None = None
    _openai_client: AsyncOpenAI | None = None

    def __init__(
        self,
        *,
        name: str,
        project: str,
        config: ModelConfig | None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        api_version: str | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            config=config,
            inference_api_key=inference_api_key,
            inference_base_url=inference_base_url,
            inference_model_name=inference_model_name,
            api_version=api_version,
            **kwargs,
        )

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        config: None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        api_version: str | None = None,
    ) -> "Model[None]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        config: ModelConfig,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        api_version: str | None = None,
    ) -> "Model[ModelConfig]": ...

    def __new__(
        cls,
        *args,
        **kwargs,
    ) -> "Model[ModelConfig] | Model[None]":
        return super().__new__(cls)

    @property
    def trainable(self) -> bool:
        return False

    def backend(self) -> "Backend":
        if self._backend is None:
            raise ValueError(
                "Model is not registered with the Backend. You must call `model.register()` first."
            )
        return self._backend

    async def register(self, backend: "Backend") -> None:
        if self.config is not None:
            try:
                self.config.model_dump_json()
            except Exception as e:
                raise ValueError(
                    "The model config cannot be serialized to JSON. Please ensure that all fields are JSON serializable and try again."
                ) from e

        self._backend = backend
        await self._backend.register(self)

    def openai_client(
        self,
    ) -> AsyncOpenAI:
        if self._openai_client is not None:
            return self._openai_client

        if self.inference_api_key is None or self.inference_base_url is None:
            if self.trainable:
                raise ValueError(
                    "OpenAI client not yet available on this trainable model. You must call `model.register()` first."
                )
            else:
                raise ValueError(
                    "In order to create an OpenAI client you must provide an `inference_api_key` and `inference_base_url`."
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
        self._openai_client = openai_client

        return self._openai_client

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
        trajectories: Iterable[Trajectory | BaseException] | Iterable[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectories or trajectory groups.

        Args:
            trajectories: A batch of trajectories or trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        if any(isinstance(t, Trajectory) for t in trajectories) or any(
            isinstance(t, BaseException) for t in trajectories
        ):
            trajectory_groups = [
                TrajectoryGroup(
                    cast(Iterable[Trajectory | BaseException], trajectories)
                )
            ]
        else:
            trajectory_groups = cast(list[TrajectoryGroup], list(trajectories))
        await self.backend()._log(
            self,
            trajectory_groups,
            split,
        )


# ---------------------------------------------------------------------------
# Trainable models
# ---------------------------------------------------------------------------


class TrainableModel(Model[ModelConfig], Generic[ModelConfig]):
    base_model: str

    # The fields within `_internal_config` are unstable and subject to change.
    # Use at your own risk.
    _internal_config: dev.InternalModelConfig | None = None

    def __init__(
        self,
        *,
        name: str,
        project: str,
        config: ModelConfig | None = None,
        base_model: str,
        _internal_config: dev.InternalModelConfig | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            config=config,
            base_model=base_model,  # type: ignore
            **kwargs,
        )
        if _internal_config is not None:
            # Bypass BaseModel __setattr__ to allow setting private attr
            object.__setattr__(self, "_internal_config", _internal_config)

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        config: None = None,
        base_model: str,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[None]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        config: ModelConfig,
        base_model: str,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[ModelConfig]": ...

    def __new__(
        cls,
        *args,
        **kwargs,
    ) -> "TrainableModel[ModelConfig] | TrainableModel[None]":
        return super().__new__(cls)  # type: ignore

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["_internal_config"] = self._internal_config
        return data

    @property
    def trainable(self) -> bool:
        return True

    async def register(
        self,
        backend: "Backend",
        _openai_client_config: dev.OpenAIServerConfig | None = None,
    ) -> None:
        await super().register(backend)
        base_url, api_key = await backend._prepare_backend_for_training(
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
        return await self.backend()._get_step(self)

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        await self.backend()._delete_checkpoints(
            self, best_checkpoint_metric, benchmark_smoothing=1.0
        )

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        async for _ in self.backend()._train_model(
            self, list(trajectory_groups), config, _config or {}, verbose
        ):
            pass
