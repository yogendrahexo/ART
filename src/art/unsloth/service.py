import asyncio
import functools
from pydantic import BaseModel
import torch
import traceback
from typing import Awaitable, Callable, cast, ParamSpec, TYPE_CHECKING, TypeVar

from .. import types
from .checkpoints import get_iteration, get_last_iteration_dir
from ..config.model import ModelConfig
from ..config.openai_server import get_openai_server_config, OpenAIServerConfig
from .model import get_model_and_tokenizer
from .pack import DiskPackedTensors, packed_tensors_from_dir, PackedTensors
from .train import get_trainer, train
from .vllm import openai_server_task

T = TypeVar("T")
P = ParamSpec("P")

if TYPE_CHECKING:
    from peft.peft_model import PeftModel
    from transformers import PreTrainedTokenizerBase
    from trl import GRPOTrainer


def catch_and_print_errors(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T]]:
    """
    Decorator that catches, prints, and reraises any errors that occur in the wrapped function.
    """

    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await cast(Awaitable[T], func(*args, **kwargs))
        except Exception as e:
            if __name__ == "__main__":
                print(f"Error in {func.__name__}: {e}")
                traceback.print_exc()
            raise

    return async_wrapper


class StartOpenaiServer(BaseModel):
    tool_use: bool


class TuneInputs(PackedTensors):
    config: types.TuneConfig


class ModelService(BaseModel):
    host: str
    port: int
    model_name: str
    base_model: types.BaseModel
    config: ModelConfig
    output_dir: str
    process: asyncio.subprocess.Process | None = None
    _openai_server_task: asyncio.Task[None] | None = None
    _train_task: asyncio.Task[None] | None = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @catch_and_print_errors
    async def start_openai_server(
        self, tool_use: bool, config: OpenAIServerConfig | None
    ) -> None:
        peft_model, _ = self.model_and_tokenizer
        lora_path = get_last_iteration_dir(self.output_dir)
        if lora_path is None:
            lora_path = f"{self.output_dir}/0000"
            self.trainer.save_model(lora_path)
        await self.stop_openai_server()
        self._openai_server_task = openai_server_task(
            model=peft_model,
            config=get_openai_server_config(
                model_name=self.model_name,
                base_model=self.base_model,
                tool_use=tool_use,
                lora_path=lora_path,
                config=config,
            ),
        )
        done, _ = await asyncio.wait([self._openai_server_task], timeout=1.0)
        for task in done:
            task.result()

    @catch_and_print_errors
    async def stop_openai_server(self) -> None:
        if self._openai_server_task:
            self._openai_server_task.cancel()
            self._openai_server_task = None

    @catch_and_print_errors
    async def tune(
        self, disk_packed_tensors: DiskPackedTensors, config: types.TuneConfig
    ) -> None:
        from unsloth_zoo.training_utils import set_training, unset_training  # type: ignore

        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        await self.inputs_queue.join()
        model, _ = self.model_and_tokenizer
        set_training(model)
        trainer = self.trainer
        for i in range(packed_tensors["tokens"].shape[0]):
            self.inputs_queue.put_nowait(
                TuneInputs(
                    **{
                        k: v[i : i + 1]
                        for k, v in packed_tensors.items()
                        if isinstance(v, torch.Tensor)
                    },
                    config=config,
                )
            )
        if self._train_task is None:
            self._train_task = asyncio.create_task(train(trainer, self.inputs_queue))
        (done,), _ = await asyncio.wait(
            [self._train_task, asyncio.create_task(self.inputs_queue.join())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # unset_training(peft_model)
        if exception := done.exception():
            raise exception
        # Save the new lora
        iteration_dir = f"{self.output_dir}/{get_iteration(self.output_dir) + 1:04d}"
        trainer.save_model(iteration_dir)
        # Swap in the new lora
        lora_request = model.load_lora(
            iteration_dir,
            load_tensors=True,
        )
        lora_request.lora_int_id = 1
        lora_request.lora_name = self.model_name
        model.vllm_engine.engine.remove_lora(1)
        model.vllm_engine.engine.add_lora(lora_request)

    @functools.cached_property
    def model_and_tokenizer(self) -> tuple["PeftModel", "PreTrainedTokenizerBase"]:
        return get_model_and_tokenizer(self.config)

    @functools.cached_property
    def inputs_queue(self) -> asyncio.Queue[TuneInputs]:
        return asyncio.Queue()

    @functools.cached_property
    def trainer(self) -> "GRPOTrainer":
        import unsloth  # type: ignore
        from trl import GRPOConfig

        peft_model, tokenizer = self.model_and_tokenizer
        self._trainer = get_trainer(
            model=peft_model,
            tokenizer=tokenizer,
            args=self.config.train_args or GRPOConfig(),
            inputs_queue=self.inputs_queue,
        )
        return self._trainer
