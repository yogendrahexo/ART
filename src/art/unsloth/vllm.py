from argparse import Namespace
import asyncio
from contextlib import asynccontextmanager
import re
from transformers import AutoModelForCausalLM
from typing import AsyncIterator
from unsloth.models import FastLanguageModel
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

from .. import UVICORN_LOGGING_CONFIG_PATH


build_async_engine_client = api_server.build_async_engine_client


def max_concurrent_tokens() -> int:
    with open("./logs/vllm.log", "r") as f:
        matches = re.findall(
            r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x",
            f.read(),
        )
        return int(int(matches[-1][0]) * float(matches[-1][1]))


def openai_server_task(
    model: AutoModelForCausalLM, model_name: str, tool_use: bool
) -> asyncio.Task:
    @asynccontextmanager
    async def yield_unsloth_async_engine_client(
        _: Namespace,
    ) -> AsyncIterator[EngineClient]:
        yield getattr(model, "vllm_engine")

    api_server.build_async_engine_client = yield_unsloth_async_engine_client
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = [
        "--api-key",
        "default",
        "--disable-log-requests",
        "--num-scheduler-steps",
        "4",
        "--served-model-name",
        model_name,
    ]
    if tool_use:
        args.extend(["--enable-auto-tool-choice", "--tool-call-parser", "hermes"])
    namespace = parser.parse_args(args)
    validate_parsed_serve_args(namespace)
    return asyncio.create_task(
        api_server.run_server(namespace, log_config=UVICORN_LOGGING_CONFIG_PATH)
    )
