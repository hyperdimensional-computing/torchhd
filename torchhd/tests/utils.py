#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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

    return dtype in vsa_tensor.supported_dtypes


vsa_tensors = [
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
    "BSBC",
    "VTB",
    "MCR",
]
