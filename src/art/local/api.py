from openai import AsyncOpenAI
import os
import torch
from transformers import AutoTokenizer

from ..api import API
from ..model import Model
from ..types import BaseModel, Trajectory, TuneConfig, Verbosity
from .grpo import GRPO
from .pack import packed_tensors_from_tokenized_results, plot_packed_tensors
from .model_configs import model_configs
from .recipe import ComponentConfig, TuneRecipeConfig
from .tokenize import tokenize_trajectory_groups
from .tune import get_last_iteration_dir, tune
from .vllm import kill_vllm_workers, start_vllm, vLLM


class LocalAPI(API):

    def __init__(self, *, path: str = "./.art") -> None:
        self._path = path
        os.makedirs(self._path, exist_ok=True)
        self._vllm: vLLM | None = None

    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
        os.makedirs(f"{self._path}/models/{name}", exist_ok=True)
        return Model(api=self, name=name, base_model=base_model)

    async def _get_openai_client(
        self, model: Model, verbosity: Verbosity
    ) -> AsyncOpenAI:
        self._vllm = await start_vllm(
            get_last_iteration_dir(f"{self._path}/models/{model.name}")
            or model.base_model,
            model.name,
            max_concurrent_requests=4096,
            env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
            named_arguments=dict(
                block_size=32,
                disable_log_requests=True,
                enable_prefix_caching=True,
                enforce_eager=True,
                gpu_memory_utilization=0.95,
                max_model_len=16384,
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
        return self._vllm.client

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        await client.close()
        if self._vllm:
            self._vllm.process.terminate()
            kill_vllm_workers()

    async def _save_eval(
        self, model: Model, trajectory_groups: list[list[Trajectory]]
    ) -> None: ...

    async def _tune_model(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory]],
        config: TuneConfig,
    ) -> None:
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
            model.base_model,
            f"{self._path}/models/{model.name}",
            packed_tensors,
            model_config.tune_model,
            model_config.tune_model_type,
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
