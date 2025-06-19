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
from torchhd import HRRTensor

from ..utils import torch_dtypes, supported_dtype, vsa_tensors

seed = 2147483644


class Testcircular:
    @pytest.mark.parametrize("n", [1, 3, 55])
    @pytest.mark.parametrize("d", [84, 16])
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_shape(self, n, d, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.circular(n, d, vsa, block_size=1024)
        else:
            hv = functional.circular(n, d, vsa)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_generator(self, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return

        generator = torch.Generator()
        generator.manual_seed(seed)
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv1 = functional.circular(
                20, 10000, vsa, generator=generator, block_size=1024
            )
        else:
            hv1 = functional.circular(20, 10000, vsa, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv2 = functional.circular(
                20, 10000, vsa, generator=generator, block_size=1024
            )
        else:
            hv2 = functional.circular(20, 10000, vsa, generator=generator)
        assert torch.all(hv1 == hv2).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            with pytest.raises(ValueError):
                if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
                    functional.circular(3, 26, vsa, dtype=dtype, block_size=1024)
                else:
                    functional.circular(3, 26, vsa, dtype=dtype)

            return

        if vsa == "HRR" or vsa == "VTB":
            with pytest.raises(ValueError):
                functional.circular(3, 26, vsa, dtype=dtype)

            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.circular(
                50, 26569, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            hv = functional.circular(50, 26569, vsa, dtype=dtype, generator=generator)
        assert hv.requires_grad == False
        assert hv.dim() == 2
        assert hv.size(0) == 50
        assert hv.size(1) == 26569

        if vsa == "BSC":
            assert torch.all((hv == False) | (hv == True)).item()

        elif vsa == "MAP":
            assert torch.all((hv == -1) | (hv == 1)).item()

        elif vsa == "FHRR":
            mag = hv.abs()
            assert torch.allclose(
                mag, torch.tensor(1.0, dtype=mag.dtype), rtol=0.0001, atol=0.0001
            )

        elif vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            assert torch.all((hv >= 0) | (hv < 1024)).item()

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.circular(
                8, 1000000, vsa, generator=generator, dtype=dtype, block_size=1024
            )
        else:
            hv = functional.circular(8, 1000000, vsa, generator=generator, dtype=dtype)

        for i in range(8-1):
            sims = functional.cosine_similarity(hv[0], hv)
            sims_diff = sims[:-1] - sims[1:]
            assert torch.all(
                sims_diff.sign() == torch.tensor([1, 1, 1, 1, -1, -1, -1])
            ), f"element #{i}: second half must get more similar"

            assert torch.allclose(
                sims_diff.abs(), torch.tensor(0.25, dtype=sims_diff.dtype), atol=0.005
            ), f"element #{i}: similarity decreases linearly"
            hv = torch.roll(hv,1,0)

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.756, 1.0])
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_sparsity(self, sparsity, dtype):
        if not supported_dtype(dtype, torchhd.BSCTensor):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.circular(
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
    def test_device(self, dtype):
        if not supported_dtype(dtype):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular(3, 49, device=device, dtype=dtype)
        assert hv.device.type == device.type

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_device(self, dtype, vsa):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "HRR" or vsa == "VTB":
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.circular(
                3, 52, vsa, device=device, dtype=dtype, block_size=1024
            )
        else:
            hv = functional.circular(3, 49, vsa, device=device, dtype=dtype)
        assert hv.device.type == device.type

    def test_uses_default_dtype(self):
        hv = functional.circular(3, 52, "BSC")
        assert hv.dtype == torch.bool

        torch.set_default_dtype(torch.float32)
        hv = functional.circular(3, 52, "MAP")
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.circular(3, 52, "MAP")
        assert hv.dtype == torch.float64

        hv = functional.circular(3, 52, "FHRR")
        assert hv.dtype == torch.complex64

        hv = functional.circular(3, 52, "BSBC", block_size=1024)
        assert hv.dtype == torch.int64

    def test_requires_grad(self):
        hv = functional.circular(3, 52, "MAP", requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.circular(3, 52, "FHRR", requires_grad=True)
        assert hv.requires_grad == True
