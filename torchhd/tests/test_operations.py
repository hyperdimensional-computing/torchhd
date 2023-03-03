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

from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestBind:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        hv = functional.empty(2, 10, vsa, dtype=dtype)
        res = functional.bind(hv[0], hv[1])
        if vsa == "BSC":
            assert torch.all(res == torch.logical_xor(hv[0], hv[1])).item()
        elif vsa == "FHRR" or vsa == "MAP":
            assert torch.all(res == torch.mul(hv[0], hv[1])).item()
        elif vsa == "HRR":
            from torch.fft import fft, ifft

            assert torch.all(res == ifft(torch.mul(fft(hv[0]), fft(hv[1])))).item()
        assert dtype == res.dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(2, 100, device=device)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert res.device == device


class TestBundle:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        hv = functional.random(2, 10, vsa, dtype=dtype)
        res = functional.bundle(hv[0], hv[1])

        if vsa == "BSC":
            hv[0] = torch.tensor(
                [False, False, True, False, False, True, True, True, False, False]
            )
            hv[1] = torch.tensor(
                [True, False, True, False, False, True, False, False, True, False]
            )

            res = functional.bundle(hv[0], hv[1])
            for i in range(10):
                assert (
                    (
                        hv[0][i].item() == hv[1][i].item()
                        and hv[1][i].item() == True
                        and res[i].item()
                    )
                    or (
                        hv[0][i].item() == hv[1][i].item()
                        and hv[1][i].item() == False
                        and not res[i].item()
                    )
                    or (hv[0][i].item() != hv[1][i].item())
                )

        if vsa == "MAP":
            hv[0] = torch.tensor([1, 1, -1, -1, 1, 1, 1, 1, -1, -1])
            hv[1] = torch.tensor([1, 1, -1, -1, -1, -1, -1, -1, 1, -1])

            res = functional.bundle(hv[0], hv[1])
            assert torch.all(
                res == torch.tensor([2, 2, -2, -2, 0, 0, 0, 0, 0, -2], dtype=dtype)
            ).item()
        if vsa == "FHRR":
            assert torch.all(res == hv[0].add(hv[1])).item()
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(2, 100, device=device)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv <= 2) & (hv >= -2)).item(), "values are between -2 and +2"
        assert res.device == device


class TestPermute:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        hv = functional.random(2, 100, vsa, dtype=dtype)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        if vsa == "BSC":
            assert torch.all((hv == 0) | (hv == 1)).item(), "values are either -1 or +1"
            assert torch.sum(res == hv[0]) != res.size(
                0
            ), "all element must not be the same"

            one_shift = functional.permute(hv[0])
            two_shift = functional.permute(hv[0], shifts=2)
            assert torch.sum(one_shift == two_shift) != res.size(
                0
            ), "all element must not be the same"

            hv = functional.random(1, 10000, vsa, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        if vsa == "MAP":
            assert torch.all(
                (hv == -1) | (hv == 1)
            ).item(), "values are either -1 or +1"
            assert torch.sum(res == hv[0]) != res.size(
                0
            ), "all element must not be the same"

            one_shift = functional.permute(hv[0])
            two_shift = functional.permute(hv[0], shifts=2)
            assert torch.sum(one_shift == two_shift) != res.size(
                0
            ), "all element must not be the same"

            hv = functional.random(1, 10000, vsa, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        if vsa == "HRR" or vsa == "FHRR":
            hv = functional.random(1, 10000, vsa, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(2, 100, device=device)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert res.device == device


class TestCleanup:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random(5, 100, vsa, dtype=dtype, generator=generator)
        noise = functional.random(1, 100, vsa, dtype=dtype, generator=generator)
        res = functional.cleanup(functional.bundle(hv[0], noise), hv)
        assert torch.all(hv[0] == res).item()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_threshold(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random(5, 100, vsa, dtype=dtype, generator=generator)
        noise = functional.random(1, 100, vsa, dtype=dtype, generator=generator)
        res = functional.cleanup(functional.bundle(hv[0], noise), hv, threshold=0.3)
        assert torch.all(hv[0] == res).item()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(5, 100, vsa, dtype=dtype)
        res = functional.cleanup(hv[0], hv)
        assert res.device == device


class TestRandsel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        a, b = functional.random(2, 1000, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, p=0, generator=generator)
        assert torch.all(a == res)

        a, b = functional.random(2, 1000, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, p=1, generator=generator)
        assert torch.all(b == res)

        a, b = functional.random(2, 1000, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, generator=generator)
        assert torch.all((b == res) | (a == res))
        assert res.dtype == dtype

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        a, b = functional.random(2, 100, vsa, dtype=dtype)
        res = functional.randsel(a, b)

        assert res.dtype == a.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert res.device == device


class TestMultiRandsel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        x = functional.random(4, 1000, vsa, dtype=dtype)

        res = functional.multirandsel(
            x, p=torch.tensor([0.0, 0.0, 1.0, 0.0]), generator=generator
        )
        assert torch.all(x[2] == res)

        x = functional.random(4, 1000, vsa, dtype=dtype)
        res = functional.multirandsel(
            x, p=torch.tensor([0.5, 0.0, 0.5, 0.0]), generator=generator
        )
        assert torch.all((x[0] == res) | (x[2] == res))

        x = functional.random(4, 1000, vsa, dtype=dtype)
        res = functional.multirandsel(x, generator=generator)
        assert torch.all((x[0] == res) | (x[1] == res) | (x[2] == res) | (x[3] == res))
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = functional.random(4, 100, device=device)
        res = functional.multirandsel(x)

        assert res.dtype == x.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert res.device == device


class TestRandomPermute:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        x = functional.random(4, 100)

        perm = functional.create_random_permute(100)

        assert torch.equal(x, perm(perm(x, 3), -3))
        assert torch.equal(x, perm(x, 0))
        assert torch.allclose(x.sort().values, perm(x, 5).sort().values)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = functional.random(4, 100, device=device)
        perm = functional.create_random_permute(100)
        assert perm(x).device == device