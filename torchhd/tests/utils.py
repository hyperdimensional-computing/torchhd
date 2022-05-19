from typing import Union

number = Union[float, int]


def between(value: number, low: number, high: number) -> bool:
    return low < value < high
