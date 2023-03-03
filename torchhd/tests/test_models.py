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
import torch.nn.functional as F
import torchhd
from torchhd import models
from torchhd import MAPTensor

from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestCentroid:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_initialization(self, dtype):
        if dtype not in MAPTensor.supported_dtypes:
            return

        model = models.Centroid(1245, 12, dtype=dtype)
        assert torch.allclose(model.weight, torch.zeros(12, 1245, dtype=dtype))
        assert model.weight.dtype == dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.Centroid(1245, 12, dtype=dtype, device=device)
        assert torch.allclose(model.weight, torch.zeros(12, 1245, dtype=dtype))
        assert model.weight.dtype == dtype
        assert model.weight.device.type == device.type

    def test_add(self):
        samples = torch.randn(4, 12)
        targets = torch.tensor([0, 1, 2, 2])

        model = models.Centroid(12, 3)
        model.add(samples, targets)

        c = samples[:-1].clone()
        c[-1] += samples[-1]

        assert torch.allclose(model(samples), torchhd.cos(samples, c))
        assert torch.allclose(model(samples, dot=True), torchhd.dot(samples, c))

        model.normalize()
        print(model(samples, dot=True))
        print(torchhd.cos(samples, c))
        assert torch.allclose(
            model(samples, dot=True), torchhd.dot(samples, F.normalize(c))
        )

    def test_add_online(self):
        samples = torch.randn(10, 12)
        targets = torch.randint(0, 3, (10,))

        model = models.Centroid(12, 3)
        model.add_online(samples, targets)

        logits = model(samples)
        assert logits.shape == (10, 3)


class TestIntRVFL:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_initialization(self, dtype):
        if dtype not in MAPTensor.supported_dtypes:
            return

        model = models.IntRVFL(5, 1245, 12, dtype=dtype)
        assert torch.allclose(model.weight, torch.zeros(12, 1245, dtype=dtype))
        assert model.weight.dtype == dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.IntRVFL(5, 1245, 12, dtype=dtype, device=device)
        assert torch.allclose(model.weight, torch.zeros(12, 1245, dtype=dtype))
        assert model.weight.dtype == dtype
        assert model.weight.device.type == device.type

    def test_fit_ridge_regression(self):
        samples = torch.randn(10, 12)
        targets = torch.randint(0, 3, (10,))

        model = models.IntRVFL(12, 1245, 3)
        model.fit_ridge_regression(samples, targets)

        logits = model(samples)
        assert logits.shape == (10, 3)
