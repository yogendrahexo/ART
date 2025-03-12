import asyncio
from openai import AsyncOpenAI
import os
import torch
from transformers import AutoTokenizer
import wandb
from wandb.sdk.wandb_run import Run

from ..api import API
from ..model import Model
from ..types import BaseModel, Trajectory, TuneConfig, Verbosity
from .grpo import GRPO
from .pack import packed_tensors_from_tokenized_results, plot_packed_tensors
from .model_configs import model_configs
from .recipe import ComponentConfig, TuneRecipeConfig
from .tokenize import tokenize_trajectory_groups
from .tune import get_iteration, get_last_iteration_dir, get_last_tune_metrics, tune
from .vllm import kill_vllm_workers, start_vllm, vLLM


class LocalAPI(API):

    def __init__(
        self,
        *,
        path: str = "./.art",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        self._path = path
        os.makedirs(self._path, exist_ok=True)
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._vllm: vLLM | None = None
        self._wandb_runs: dict[str, Run] = {}

    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
        os.makedirs(self._get_output_dir(name), exist_ok=True)
        return Model(api=self, name=name, base_model=base_model)

    def _get_output_dir(self, model_name: str) -> str:
        return f"{self._path}/models/{model_name}"

    async def _get_iteration(self, model: Model) -> int:
        return self.__get_iteration(model)

    def __get_iteration(self, model: Model) -> int:
        return get_iteration(self._get_output_dir(model.name))

    async def _get_openai_client(
        self, model: Model, estimated_token_usage: int, verbosity: Verbosity
    ) -> tuple[AsyncOpenAI, asyncio.Semaphore]:
        self._vllm = await start_vllm(
            get_last_iteration_dir(self._get_output_dir(model.name))
            or model.base_model,
            model.name,
            max_concurrent_requests=4096,
            named_arguments=dict(
                block_size=32,
                disable_log_requests=True,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                enforce_eager=True,
                gpu_memory_utilization=0.95,
                max_num_seqs=4096,
                max_num_batched_tokens=16384,
                num_scheduler_steps=16,
                preemption_mode="swap",
                return_tokens_as_token_ids=True,
                swap_space=80,
                tensor_parallel_size=torch.cuda.device_count(),
            ),
            timeout=360 + 15 * torch.cuda.device_count(),
            verbosity=verbosity,
        )
        return self._vllm.client, asyncio.Semaphore(
            int(self._vllm.max_concurrent_tokens / estimated_token_usage)
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        await client.close()
        if self._vllm:
            self._vllm.process.terminate()
            kill_vllm_workers()

    async def _save(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory]],
        name: str = "val",
    ) -> None:
        # Collect all metrics (including reward) across all trajectories
        all_metrics: dict[str, list[float]] = {"reward": []}

        for group in trajectory_groups:
            for trajectory in group:
                # Add reward metric
                all_metrics["reward"].append(trajectory.reward)

                # Collect other custom metrics
                for metric, value in trajectory.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        # Calculate averages for all metrics
        averages = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        self._log_wandb_data(model, averages, name)

    async def _tune_model(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory]],
        config: TuneConfig,
    ) -> None:
        await self._save(model, trajectory_groups, "train")
        tokenizer = AutoTokenizer.from_pretrained(model.base_model)
        tokenized_results = list(
            tokenize_trajectory_groups(tokenizer, trajectory_groups)
        )
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            config.sequence_length,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
        )
        if config.plot_tensors:
            plot_packed_tensors(packed_tensors)
        elif config.verbosity > 0:
            print(f"Packed tensors with shape: {packed_tensors['tokens'].shape}")
        model_config = model_configs[model.base_model]()
        optimizer = ComponentConfig(
            (
                "torch.optim.AdamW"
                if torch.cuda.device_count() > 1
                else "bitsandbytes.optim.PagedAdam8bit"
            ),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        if torch.cuda.device_count() > 1:
            optimizer.fused = True
        await tune(
            base_model=model.base_model,
            output_dir=self._get_output_dir(model.name),
            packed_tensors=packed_tensors,
            model=model_config.tune_model,
            model_type=model_config.tune_model_type,
            config=TuneRecipeConfig(
                optimizer=optimizer,
                loss=ComponentConfig(
                    GRPO,
                    clip_epsilon=config.clip_epsilon,
                    entropy_coef=config.entropy_coef,
                    kl_coef=config.kl_coef,
                ),
                shuffle=True,
                batch_size=32768 // config.sequence_length,
                fsdp_cpu_offload=True,
                enable_activation_checkpointing=True,
                enable_activation_offloading=True,
                custom_sharded_layers=["tok_embeddings", "output"],
                num_output_chunks=model_config.tune_num_output_chunks,
                compile=True,
            ),
            verbosity=config.verbosity,
        )
        self._log_wandb_data(
            model,
            get_last_tune_metrics(self._get_output_dir(model.name)),
            "train",
        )

    def _log_wandb_data(
        self, model: Model, data: dict[str, float], namespace: str
    ) -> None:
        # Initialize wandb only if we don't have an active run for this model
        if model.name not in self._wandb_runs or not wandb.run:
            run = wandb.init(
                entity=self._wandb_entity,
                project=self._wandb_project,
                id=model.name,
                name=model.name,
                resume="allow",
            )
            self._wandb_runs[model.name] = run

        # Add namespacing if needed
        data = (
            {f"{namespace}/{metric}": value for metric, value in data.items()}
            if namespace
            else data
        )

        # Log the data
        self._wandb_runs[model.name].log(
            data,
            step=self.__get_iteration(model),
        )
