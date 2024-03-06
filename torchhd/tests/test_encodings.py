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

from torchhd import functional
from torchhd.tensors.bsc import BSCTensor
from torchhd.tensors.map import MAPTensor

from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestMultiset:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSC":
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                ],
                dtype=dtype,
            ).as_subclass(BSCTensor)
            res = functional.multiset(hv)
            assert torch.all(
                res
                == torch.tensor(
                    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                    dtype=dtype,
                )
            ).item()

        elif vsa == "MAP":
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                    [1, 1, -1, -1, -1, 1, -1, 1, -1, 1],
                ],
                dtype=dtype,
            ).as_subclass(MAPTensor)
            res = functional.multiset(hv)
            assert torch.all(
                res == torch.tensor([3, 3, -5, 1, 1, 1, 1, 1, -1, 1], dtype=dtype)
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.multiset(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.multiset(hv)
        assert res.device.type == device.type


class TestMultibind:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        if vsa == "BSC":
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                ],
                dtype=dtype,
            ).as_subclass(BSCTensor)
            res = functional.multibind(hv)
            assert torch.all(
                res
                == torch.tensor(
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                    dtype=dtype,
                )
            ).item()

        elif vsa == "MAP":
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                    [1, 1, -1, -1, -1, 1, -1, 1, -1, 1],
                ],
                dtype=dtype,
            ).as_subclass(MAPTensor)
            res = functional.multibind(hv)
            assert torch.all(
                res == torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, -1, 1], dtype=dtype)
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        if dtype in {torch.float16}:
            return

        res = functional.multibind(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.multibind(hv)
        assert res.device.type == device.type


class TestCrossProduct:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAPTensor)
        res = functional.cross_product(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]]).as_subclass(
            MAPTensor
        )
        b = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]]).as_subclass(MAPTensor)
        res = functional.cross_product(a, b)
        assert torch.all(res == torch.tensor([0, 2, 0, 6])).item()
        assert res.dtype == a.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.cross_product(hv, hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.cross_product(hv, hv)
        assert res.device.type == device.type


class TestNgrams:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAPTensor)
        res = functional.ngrams(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAPTensor)
        res = functional.ngrams(hv)
        assert torch.all(res == torch.tensor([0, -2, -2, 0])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.ngrams(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.ngrams(hv)
        assert res.device.type == device.type


class TestHashTable:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAPTensor)
        res = functional.hash_table(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]]).as_subclass(MAPTensor)
        b = torch.tensor([[-1, 1, 1, 1], [-1, 1, 1, -1]]).as_subclass(MAPTensor)
        res = functional.hash_table(a, b)
        assert torch.all(res == torch.tensor([0, -2, 0, 0])).item()
        assert res.dtype == a.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.hash_table(hv, hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.hash_table(hv, hv)
        assert res.device.type == device.type


class TestBundleSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAPTensor)
        res = functional.bundle_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAPTensor)
        res = functional.bundle_sequence(hv)
        assert torch.all(res == torch.tensor([0, 0, 2, 0])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.bundle_sequence(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.bundle_sequence(hv)
        assert res.device.type == device.type


class TestBindSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAPTensor)
        res = functional.bind_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAPTensor)
        res = functional.bind_sequence(hv)
        assert torch.all(res == torch.tensor([1, 1, -1, 1])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAPTensor)

        if dtype in {torch.float16}:
            return

        res = functional.bind_sequence(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random(11, 10000, device=device)
        res = functional.bind_sequence(hv)
        assert res.device.type == device.type


class TestGraph:
    def test_value(self):
        hv = torch.zeros(2, 4, 1000).as_subclass(MAPTensor)
        res = functional.graph(hv)
        assert torch.all(res == 0).item()

        g = torch.tensor(
            [
                [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]],
                [[-1, -1, 1, 1], [-1, 1, 1, 1], [1, -1, -1, 1]],
            ]
        ).as_subclass(MAPTensor)
        res = functional.graph(g)
        assert torch.all(res == torch.tensor([-1, -1, -1, 3])).item()
        assert res.dtype == g.dtype

        res = functional.graph(g, directed=True)
        assert torch.all(res == torch.tensor([-1, 3, 1, 1])).item()
        assert res.dtype == g.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(5, 2, 23, 1000, dtype=dtype).as_subclass(MAPTensor)

        res = functional.graph(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = torch.zeros(5, 2, 23, 1000, device=device).as_subclass(MAPTensor)
        res = functional.graph(hv)
        assert res.device.type == device.type
