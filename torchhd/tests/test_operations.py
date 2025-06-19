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
import math
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

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.empty(2, 10, vsa, dtype=dtype, block_size=1024)
        else:
            hv = functional.empty(2, 16, vsa, dtype=dtype)
        res = functional.bind(hv[0], hv[1])
        if vsa == "BSC":
            assert torch.all(res == torch.logical_xor(hv[0], hv[1])).item()
        elif vsa == "FHRR" or vsa == "MAP" or vsa == "VTB":
            assert torch.all(res == torch.mul(hv[0], hv[1])).item()
        elif vsa == "HRR":
            from torch.fft import fft, ifft

            assert torch.all(res == ifft(torch.mul(fft(hv[0]), fft(hv[1])))).item()
        elif vsa == "BSBC":
            assert torch.all(res == ((hv[0] + hv[1]) % 1024))
        elif vsa == "MCR" or vsa == "CGR":
            assert torch.all(res == ((hv[0] + hv[1]) % 1024))
        assert dtype == res.dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(2, 100, device=device)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert res.device.type == device.type


class TestBundle:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(2, 10, vsa, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 16, vsa, dtype=dtype)
        res = functional.bundle(hv[0], hv[1])

        if vsa == "BSC":
            for i in range(10):
                assert (res[i].item() == hv[0][i].item()) or (
                    res[i].item() == hv[1][i].item()
                )

        if vsa == "MAP":
            x = torch.tensor([1, 1, -1, -1, 1, 1, 1, 1, -1, -1], dtype=dtype)
            y = torch.tensor([1, 1, -1, -1, -1, -1, -1, -1, 1, -1], dtype=dtype)

            res = functional.bundle(x, y)
            assert torch.all(
                res == torch.tensor([2, 2, -2, -2, 0, 0, 0, 0, 0, -2], dtype=dtype)
            ).item()

        if vsa == "FHRR":
            assert torch.all(res == hv[0].add(hv[1])).item()

        if vsa == "VTB":
            assert torch.all(res == hv[0].add(hv[1])).item()

        if vsa == "BSBC":
            for i in range(10):
                assert (res[i].item() == hv[0][i].item()) or (
                    res[i].item() == hv[1][i].item()
                )

        if vsa == "MCR":
            x = torch.tensor([1, 3, 5, 7, 9, 0, 2, 4, 6, 8], dtype=dtype)
            x = functional.ensure_vsa_tensor(x,'MCR')
            x.block_size = 10
            y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
            y = functional.ensure_vsa_tensor(y,'MCR')
            y.block_size = 10

            res = functional.bundle(x, y)

            possible_values = [[0,1], [1,2], [3,4], [5], [6,7,1,2], [2,3,7,8], [4],[5,6], [7], [8,9]]
            for i in range(10):
                assert (res[i].item() in possible_values[i])

        if vsa == "CGR":
            x = torch.tensor([1, 3, 5, 7, 9, 0, 2, 4, 6, 8], dtype=dtype)
            x = functional.ensure_vsa_tensor(x,'CGR')
            x.block_size = 10
            y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
            y = functional.ensure_vsa_tensor(y,'CGR')
            y.block_size = 10

            res = functional.bundle(x, y)

            possible_values = [[1,0], [3,1], [5,2], [7,3], [9,4], [0,5], [2,6], [4,7], [6,8], [8,9]]
            for i in range(10):
                assert (res[i].item() in possible_values[i])

        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(2, 100, device=device)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv <= 2) & (hv >= -2)).item(), "values are between -2 and +2"
        assert res.device.type == device.type


class TestPermute:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(2, 100, vsa, dtype=dtype, block_size=1024)
        else:
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

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(1, 10000, vsa, dtype=dtype, block_size=1024)
        else:
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
        assert res.device.type == device.type


class TestNormalize:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(12, 900, vsa, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(12, 900, vsa, dtype=dtype)

        bundle = functional.multibundle(hv)
        res = functional.normalize(bundle)

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 900

        if vsa == "BSBC" or vsa == "BSC":
            assert torch.all(bundle == res), "all elements must be the same"

        if vsa == "MAP":
            assert torch.all(
                (res == -1) | (res == 1)
            ).item(), "values are either -1 or +1"

        if vsa == "hrr" or vsa == "vtb":
            norm = torch.norm(res, p=2, dim=-1)
            assert torch.allclose(norm, torch.ones_like(norm))

        if vsa == "fhrr":
            norm = torch.norm(res, p=2, dim=-1)
            assert torch.allclose(norm, torch.full_like(norm, math.sqrt(900)))
            assert torch.allclose(res.angle(), bundle.angle())

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(4, 100, device=device).multibundle()
        res = functional.normalize(hv)

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((res == -1) | (res == 1)).item(), "values are either -1 or +1"
        assert res.device.type == device.type


class TestCleanup:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                5, 100, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            hv = functional.random(5, 100, vsa, dtype=dtype, generator=generator)
        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            noise = functional.random(
                1, 100, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
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

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                5, 100, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            hv = functional.random(5, 100, vsa, dtype=dtype, generator=generator)
        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            noise = functional.random(
                1, 100, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            noise = functional.random(1, 100, vsa, dtype=dtype, generator=generator)
        res = functional.cleanup(functional.bundle(hv[0], noise), hv, threshold=0.3)
        assert torch.all(hv[0] == res).item()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            hv = functional.random(
                5, 100, vsa, dtype=dtype, device=device, block_size=1024
            )
        else:
            hv = functional.random(5, 100, vsa, dtype=dtype, device=device)
        res = functional.cleanup(hv[0], hv)
        assert res.device.type == device.type


class TestRandsel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            a, b = functional.random(
                2, 1000, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            a, b = functional.random(2, 1024, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, p=0, generator=generator)
        assert torch.all(b == res)

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            a, b = functional.random(
                2, 1000, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            a, b = functional.random(2, 1024, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, p=1, generator=generator)
        assert torch.all(a == res)

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            a, b = functional.random(
                2, 1000, vsa, dtype=dtype, generator=generator, block_size=1024
            )
        else:
            a, b = functional.random(2, 1024, vsa, dtype=dtype, generator=generator)
        res = functional.randsel(a, b, generator=generator)
        assert torch.all((b == res) | (a == res))
        assert res.dtype == dtype

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            a, b = functional.random(
                2, 100, vsa, dtype=dtype, device=device, block_size=1024
            )
        else:
            a, b = functional.random(2, 100, vsa, dtype=dtype, device=device)
        res = functional.randsel(a, b)

        assert res.dtype == a.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert res.device.type == device.type


class TestMultiRandsel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            x = functional.random(4, 1000, vsa, dtype=dtype, block_size=1024)
        else:
            x = functional.random(4, 1024, vsa, dtype=dtype)

        res = functional.multirandsel(
            x, p=torch.tensor([0.0, 0.0, 1.0, 0.0]), generator=generator
        )
        assert torch.all(x[2] == res)

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            x = functional.random(4, 1000, vsa, dtype=dtype, block_size=1024)
        else:
            x = functional.random(4, 1024, vsa, dtype=dtype)
        res = functional.multirandsel(
            x, p=torch.tensor([0.5, 0.0, 0.5, 0.0]), generator=generator
        )
        assert torch.all((x[0] == res) | (x[2] == res))

        if  vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            x = functional.random(4, 1000, vsa, dtype=dtype, block_size=1024)
        else:
            x = functional.random(4, 1024, vsa, dtype=dtype)
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
        assert res.device.type == device.type


class TestRandomPermute:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            x = functional.random(4, 100, vsa, block_size=1024)
        else:
            x = functional.random(4, 100, vsa)

        perm = functional.create_random_permute(100)

        assert torch.equal(x, perm(perm(x, 3), -3))
        assert torch.equal(x, perm(x, 0))
        if not torch.is_complex(x):
            assert torch.allclose(x.sort().values, perm(x, 5).sort().values)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = functional.random(4, 100, device=device)
        perm = functional.create_random_permute(100)
        perm = perm.to(device)
        assert perm(x).device.type == device.type


class TestResonator:
    def test_shape(self):
        X = functional.random(5, 100)
        Y = functional.random(5, 100)
        Z = functional.random(5, 100)
        domains = torch.stack((X, Y, Z), dim=0)

        x_hat = functional.multiset(X)
        y_hat = functional.multiset(Y)
        z_hat = functional.multiset(Z)
        estimates = torch.stack((x_hat, y_hat, z_hat), dim=0)

        # Create the combined symbol
        s = X[0].bind(Y[1]).bind(Z[3])

        # resonator step
        new_estimates = functional.resonator(s, estimates, domains)
        assert new_estimates.shape == estimates.shape
        assert new_estimates.dtype == estimates.dtype
