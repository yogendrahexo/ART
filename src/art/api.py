from dataclasses import dataclass
import torch

from .model import Model
from .openai import AsyncOpenAI
from .types import Verbosity
from .vllm import kill_vllm_workers, start_vllm, vLLM


@dataclass
class API:
    _vllm: vLLM | None = None

    async def get_or_create_model(self, name: str, base_model: str) -> Model:
        return Model(api=self, name=name, base_model=base_model)

    async def _get_openai_client(
        self, model: Model, verbosity: Verbosity
    ) -> AsyncOpenAI:
        self._vllm = await start_vllm(
            model.base_model,
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
        return AsyncOpenAI(
            model=model.name, api_key="default", base_url="http://localhost:8000/v1"
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        await client.close()
        if self._vllm:
            self._vllm.process.terminate()
            kill_vllm_workers()
