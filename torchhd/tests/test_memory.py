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
from torchhd import memory
from torchhd import MAPTensor

from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestSparseDistributed:
    def test_shape(self):
        mem = memory.SparseDistributed(1000, 67, 123)

        keys = torchhd.random(1, 67).squeeze(0)
        values = torchhd.random(1, 123).squeeze(0)

        mem.write(keys, values)

        read = mem.read(keys).sign()

        assert read.shape == values.shape

        if torch.allclose(read, values):
            pass
        elif torch.allclose(read, torch.zeros_like(values)):
            pass
        else:
            assert False, "must be either the value or zero"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mem = memory.SparseDistributed(1000, 35, 74, kappa=3)
        mem = mem.to(device)

        keys = torchhd.random(5, 35, device=device)
        values = torchhd.random(5, 74, device=device)

        mem.write(keys, values)

        read = mem.read(keys).sign()

        assert read.device.type == device.type
        assert read.shape == values.shape


class TestHopfieldFn:
    def test_shape(self):
        items = torchhd.random(1, 67)

        read = memory.hopfield(items, items).sign()

        assert read.shape == items.shape
        assert torch.allclose(items, read)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        items = torchhd.random(10, 67, device=device)

        read = memory.hopfield(items, items, kappa=4).sign()

        assert read.device.type == device.type
        assert read.shape == items.shape


class TestHopfieldClass:
    def test_shape(self):
        mem = memory.Hopfield(67)

        items = torchhd.random(1, 67).squeeze(0)

        mem.write(items)

        read = mem.read(items).sign()

        assert read.shape == items.shape
        assert torch.allclose(read, items)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mem = memory.Hopfield(67, kappa=4)
        mem = mem.to(device)

        items = torchhd.random(10, 67, device=device)

        mem.write(items)
        read = mem.read(items)

        assert read.device.type == device.type
        assert read.shape == items.shape


class TestModernHopfield:
    def test_shape(self):
        items = torchhd.random(1, 67)

        read = memory.modern_hopfield(items.squeeze(0), items).sign()

        assert read.shape == (67,)
        assert torch.allclose(items, read)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        items = torchhd.random(10, 67, device=device)

        read = memory.modern_hopfield(items, items).sign()

        assert read.device.type == device.type
        assert read.shape == items.shape


class TestAttention:
    def test_shape(self):
        keys = torchhd.random(1, 67)
        values = torchhd.random(1, 123)

        read = memory.attention(keys.squeeze(0), keys, values).sign()

        assert read.shape == (123,)
        assert torch.allclose(values, read)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        keys = torchhd.random(5, 35, device=device)
        values = torchhd.random(5, 74, device=device)

        read = memory.attention(keys, keys, values).sign()

        assert read.device.type == device.type
        assert read.shape == values.shape
