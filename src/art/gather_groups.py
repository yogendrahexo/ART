import asyncio
from collections import Counter
import contextvars
import contextlib
from dataclasses import dataclass, field
from typing import Any, Coroutine, Iterable, Iterator, TypeVar


from .tqdm import tqdm
from .types import Trajectory


async def gather_groups(
    groups: Iterable[Iterable[Coroutine[Any, Any, Trajectory]]],
    *,
    pbar_desc: str | None = None,
) -> list[list[Trajectory]]:
    groups = [list(g) for g in groups]
    context = GroupsContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=sum(len(g) for g in groups))
    )
    with set_groups_context(context):
        result_groups = await asyncio.gather(
            *[asyncio.gather(*[wrap_coroutine(c) for c in g]) for g in groups]
        )
    if context.pbar is not None:
        context.pbar.close()
    return result_groups


async def wrap_coroutine(coro: Coroutine[Any, Any, Trajectory]) -> Trajectory:
    result = await coro
    context = get_groups_context()
    context.total_metrics["reward"] += result.metrics["reward"]  # type: ignore
    for metric in result.metrics:
        context.total_metrics[metric] += result.metrics[metric]  # type: ignore
    context.total_metric_reports.update(result.metrics)
    if context.pbar is not None:
        context.pbar.update(1)
        context.pbar.set_postfix(
            {
                metric: context.total_metrics[metric]
                / context.total_metric_reports[metric]
                for metric in context.total_metrics
            }
        )
    return result


@dataclass
class GroupsContext:
    pbar: tqdm.tqdm | None = None
    total_metrics: Counter[str] = field(default_factory=Counter)
    total_metric_reports: Counter[str] = field(default_factory=Counter)


groups_context_var = contextvars.ContextVar("groups_context", default=GroupsContext())


@contextlib.contextmanager
def set_groups_context(context: GroupsContext) -> Iterator[None]:
    token = groups_context_var.set(context)
    try:
        yield
    finally:
        groups_context_var.reset(token)


def get_groups_context() -> GroupsContext:
    return groups_context_var.get()
