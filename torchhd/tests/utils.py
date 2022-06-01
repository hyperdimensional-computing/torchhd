from typing import Union
import torch

number = Union[float, int]


def between(value: number, low: number, high: number) -> bool:
    return low < value < high


def within(value: number, target: number, delta: number) -> bool:
    return between(value, target - delta, target + delta)


torch_float_dtypes = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

torch_complex_dtypes = {
    torch.complex64,
    torch.complex128,
}

torch_int_dtypes = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

torch_dtypes = {
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
}


def supported_dtype(dtype: torch.dtype) -> bool:
    not_supported = dtype in torch_complex_dtypes or dtype == torch.uint8
    return not not_supported
