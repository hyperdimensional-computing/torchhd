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


class Testrandom:
    @pytest.mark.parametrize("n", [1, 3, 55])
    @pytest.mark.parametrize("d", [84, 16])
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_shape(self, n, d, vsa):
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(n, d, vsa, block_size=64)

        elif vsa == "VTB" and d == 84:
            with pytest.raises(ValueError):
                hv = functional.random(n, d, vsa)

            return

        else:
            hv = functional.random(n, d, vsa)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_generator(self, vsa):
        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv1 = functional.random(20, 10000, vsa, generator=generator, block_size=64)
        else:
            hv1 = functional.random(20, 10000, vsa, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv2 = functional.random(20, 10000, vsa, generator=generator, block_size=64)
        else:
            hv2 = functional.random(20, 10000, vsa, generator=generator)
        assert torch.all(hv1 == hv2).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            with pytest.raises(ValueError):
                if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
                    functional.random(3, 25, vsa, dtype=dtype, block_size=64)
                else:
                    functional.random(3, 25, vsa, dtype=dtype)

            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                8, 25921, vsa, dtype=dtype, generator=generator, block_size=64
            )
        else:
            hv = functional.random(8, 25921, vsa, dtype=dtype, generator=generator)
        assert hv.requires_grad == False
        assert hv.dim() == 2
        assert hv.size(0) == 8
        assert hv.size(1) == 25921

        if vsa == "BSC":
            assert torch.all((hv == False) | (hv == True)).item()

        elif vsa == "MAP":
            assert torch.all((hv == -1) | (hv == 1)).item()

        elif vsa == "HRR":
            std, mean = torch.std_mean(hv)
            assert torch.allclose(
                mean, torch.tensor(0.0, dtype=mean.dtype), atol=0.0001
            )

        elif vsa == "VTB":
            mag = torch.norm(hv, dim=-1)
            assert torch.allclose(mag, torch.tensor(1.0, dtype=mag.dtype))

        elif vsa == "FHRR":
            mag = hv.abs()
            assert torch.allclose(mag, torch.tensor(1.0, dtype=mag.dtype))

        elif vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            assert torch.all((hv < 64) & (hv >= 0))

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.756, 1.0])
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_sparsity(self, sparsity, dtype):
        if not supported_dtype(dtype, "BSC"):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random(
            1000,
            10000,
            "BSC",
            generator=generator,
            dtype=dtype,
            sparsity=sparsity,
        )

        calc_sparsity = torch.sum(hv == False).div(10000 * 1000)
        assert torch.allclose(calc_sparsity, torch.tensor(sparsity), atol=0.005)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_orthogonality(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                100, 10000, vsa, dtype=dtype, generator=generator, block_size=1042
            )
        else:
            hv = functional.random(100, 10000, vsa, dtype=dtype, generator=generator)

        sims = functional.cosine_similarity(hv[0], hv[1:])
        assert torch.allclose(
            sims.mean(), torch.tensor(0.0, dtype=sims.dtype), atol=0.002
        )

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_device(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                3, 49, vsa, device=device, dtype=dtype, block_size=64
            )
        else:
            hv = functional.random(3, 49, vsa, device=device, dtype=dtype)
        assert hv.device.type == device.type

    def test_uses_default_dtype(self):
        hv = functional.random(3, 49, "BSC")
        assert hv.dtype == torch.bool

        torch.set_default_dtype(torch.float32)
        hv = functional.random(3, 49, "MAP")
        assert hv.dtype == torch.float32
        hv = functional.random(3, 49, "HRR")
        assert hv.dtype == torch.float32
        hv = functional.random(3, 49, "VTB")
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.random(3, 49, "MAP")
        assert hv.dtype == torch.float64
        hv = functional.random(3, 49, "HRR")
        assert hv.dtype == torch.float64
        hv = functional.random(3, 49, "VTB")
        assert hv.dtype == torch.float64

        hv = functional.random(3, 49, "FHRR")
        assert hv.dtype == torch.complex64

        hv = functional.random(3, 52, "BSBC", block_size=64)
        assert hv.dtype == torch.int64

    def test_requires_grad(self):
        hv = functional.random(3, 49, "MAP", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.random(3, 49, "HRR", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.random(3, 49, "VTB", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.random(3, 49, "FHRR", requires_grad=True)
        assert hv.requires_grad == True
