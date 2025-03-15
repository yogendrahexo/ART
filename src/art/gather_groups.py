import asyncio
from collections import Counter
import contextvars
import contextlib
from dataclasses import dataclass, field
from typing import Any, Coroutine, Iterable, Iterator, Literal, overload, TypeVar


from .tqdm import tqdm


T = TypeVar("T")


@overload
async def gather_groups(
    groups: Iterable[Iterable[Coroutine[Any, Any, T]]],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: Literal[True] = True,
) -> list[list[T | BaseException]]: ...


@overload
async def gather_groups(
    groups: Iterable[Iterable[Coroutine[Any, Any, T]]],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: Literal[False],
) -> list[list[T]]: ...


async def gather_groups(
    groups: Iterable[Iterable[Coroutine[Any, Any, T]]],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: bool = True,
) -> list[list[T | BaseException]] | list[list[T]]:
    groups = [list(g) for g in groups]
    context = GroupsContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=sum(len(g) for g in groups)),
        pbar_total_completion_tokens=pbar_total_completion_tokens,
    )
    with set_groups_context(context):
        result_groups = await asyncio.gather(
            *[
                asyncio.gather(
                    *[wrap_coroutine(c) for c in g], return_exceptions=return_exceptions
                )
                for g in groups
            ]
        )
    if context.pbar is not None:
        context.pbar.close()
    return result_groups


async def wrap_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    context = get_groups_context()
    try:
        result = await coro
    except BaseException as e:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        raise e
    else:
        context.update_pbar(n=1)
        return result


@dataclass
class GroupsContext:
    pbar: tqdm.tqdm | None = None
    metric_sums: Counter[str] = field(default_factory=Counter)
    metric_divisors: Counter[str] = field(default_factory=Counter)
    pbar_total_completion_tokens: bool = False

    def update_pbar(self, n: int) -> None:
        if self.pbar is not None:
            self.pbar.update(n)
            postfix = {}
            for metric in self.metric_sums:
                sum = self.metric_sums[metric]
                divisor = max(1, self.metric_divisors[metric])
                if isinstance(sum, int):
                    postfix[metric] = int(sum / divisor)
                else:
                    postfix[metric] = sum / divisor
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_completion_tokens",
            ):
                if key in postfix:
                    postfix[key] = postfix.pop(key)
            self.pbar.set_postfix(postfix)


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
