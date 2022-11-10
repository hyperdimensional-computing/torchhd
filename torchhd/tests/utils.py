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


def supported_dtype(dtype: torch.dtype, model: Type[torchhd.VSA_Model]) -> bool:
    if not issubclass(model, torchhd.VSA_Model):
        raise ValueError("Must provide a VSA_Model class")

    if model == torchhd.BSC:
        return dtype not in {
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
        }

    elif model == torchhd.MAP:
        return dtype not in {torch.uint8, torch.bool, torch.float16, torch.bfloat16}

    elif model == torchhd.HRR:
        return dtype in {torch.float32, torch.float64}

    elif model == torchhd.FHRR:
        return dtype in {torch.complex64, torch.complex128}

    return False


vsa_models = [
    torchhd.BSC,
    torchhd.MAP,
    # torchhd.HRR,
    torchhd.FHRR,
]
