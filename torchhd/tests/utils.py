from typing import Union, Type
import torch
import torchhd

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
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
}


def supported_dtype(dtype: torch.dtype, model: Type[torchhd.VSATensor]) -> bool:
    if not issubclass(model, torchhd.VSATensor):
        raise ValueError("Must provide a VSATensor class")

    if model == torchhd.BSCTensor:
        return dtype not in {
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
        }

    elif model == torchhd.MAPTensor:
        return dtype not in {torch.uint8, torch.bool, torch.float16, torch.bfloat16}

    elif model == torchhd.HRRTensor:
        return dtype in {torch.float32, torch.float64}

    elif model == torchhd.FHRRTensor:
        return dtype in {torch.complex64, torch.complex128}

    return False


VSATensors = [
    torchhd.BSCTensor,
    torchhd.MAPTensor,
    torchhd.HRRTensor,
    torchhd.FHRRTensor,
]
