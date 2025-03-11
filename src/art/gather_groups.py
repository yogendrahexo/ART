import asyncio
import contextvars
import contextlib
from dataclasses import dataclass
from typing import Any, Coroutine, Iterable, Iterator, TypeVar


from .tqdm import tqdm


T = TypeVar("T")


async def gather_groups(
    groups: Iterable[Iterable[Coroutine[Any, Any, T]]],
    *,
    pbar_desc: str | None = None,
) -> list[list[T]]:
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


async def wrap_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    result = await coro
    context = get_groups_context()
    if context.pbar is not None:
        context.pbar.update(1)
    return result


@dataclass
class GroupsContext:
    pbar: tqdm.tqdm | None = None


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
