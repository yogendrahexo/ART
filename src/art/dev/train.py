from typing_extensions import TypedDict


class TrainConfig(TypedDict, total=False):
    epsilon: float  # clip epsilon, using the same name as TRL
    logprob_calculation_chunk_size: int
