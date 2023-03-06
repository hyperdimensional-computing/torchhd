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
import pytest
import torch

import torchhd
from torchhd import functional

from ..utils import (
    torch_dtypes,
    supported_dtype,
    vsa_tensors,
)

seed = 2147483644


class Testidentity:
    @pytest.mark.parametrize("n", [1, 3, 55])
    @pytest.mark.parametrize("d", [84, 10])
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_shape(self, n, d, vsa):
        hv = functional.identity(n, d, vsa)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            with pytest.raises(ValueError):
                functional.identity(3, 26, vsa, dtype=dtype)

            return

        hv = functional.identity(8, 26, vsa, dtype=dtype)
        assert hv.requires_grad == False
        assert hv.dim() == 2
        assert hv.size(0) == 8
        assert hv.size(1) == 26

        if vsa == "BSC":
            assert torch.all(hv == False).item()

        elif vsa == "HRR":
            hv = functional.identity(8, 26, vsa, dtype=dtype)
            x = torch.fft.fft(hv)
            assert torch.allclose(x, torch.full_like(x, 1.0))

        else:
            hv = functional.identity(8, 26, vsa, dtype=dtype)
            assert torch.all(hv == 1.0).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_device(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity(3, 52, vsa, device=device, dtype=dtype)
        assert hv.device.type == device.type

    def test_uses_default_dtype(self):
        hv = functional.identity(3, 52, "BSC")
        assert hv.dtype == torch.bool

        torch.set_default_dtype(torch.float32)
        hv = functional.identity(3, 52, "MAP")
        assert hv.dtype == torch.float32
        hv = functional.identity(3, 52, "HRR")
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.identity(3, 52, "MAP")
        assert hv.dtype == torch.float64
        hv = functional.identity(3, 52, "HRR")
        assert hv.dtype == torch.float64

        hv = functional.identity(3, 52, "FHRR")
        assert hv.dtype == torch.complex64

    def test_requires_grad(self):
        hv = functional.identity(3, 52, "MAP", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.identity(3, 52, "HRR", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.identity(3, 52, "FHRR", requires_grad=True)
        assert hv.requires_grad == True
