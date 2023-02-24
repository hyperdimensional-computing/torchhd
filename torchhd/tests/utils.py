from typing import Union, Type
import torch
import torchhd
from torchhd.types import VSAOptions

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


def supported_dtype(
    dtype: torch.dtype, vsa: Union[Type[torchhd.VSATensor], VSAOptions]
) -> bool:
    if isinstance(vsa, str):
        vsa_tensor = torchhd.functional.get_vsa_tensor_class(vsa)
    else:
        vsa_tensor = vsa

    if not issubclass(vsa_tensor, torchhd.VSATensor):
        raise ValueError("Must provide a VSATensor class")

    if vsa_tensor == torchhd.BSCTensor:
        return dtype not in {
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
        }

    elif vsa_tensor == torchhd.MAPTensor:
        return dtype not in {torch.uint8, torch.bool, torch.float16, torch.bfloat16}

    elif vsa_tensor == torchhd.HRRTensor:
        return dtype in {torch.float32, torch.float64}

    elif vsa_tensor == torchhd.FHRRTensor:
        return dtype in {torch.complex64, torch.complex128}

    return False


vsa_tensors = [
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
]
