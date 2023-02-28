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


class TestMapRange:
    def test_value(self):
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5])
        res = functional.map_range(x, 0, 1, 0, 10)
        assert torch.all(res == torch.tensor([-10.0, 0.0, 1.0, 5.0])).item()

        x = torch.tensor([-1.0, 0.0, 0.1, 0.5])
        res = functional.map_range(x, 0, 1, 1, 0)
        assert torch.all(res == torch.tensor([2.0, 1.0, 0.9, 0.5])).item()

    def test_dtype(self):
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], dtype=torch.float)
        res = functional.map_range(x, 0, 1, 0, 10)
        assert res.dtype == torch.float

        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], dtype=torch.float64)
        res = functional.map_range(x, 0, 1, 0, 10)
        assert res.dtype == torch.float64

        x = torch.tensor([0, 1, 2, 5], dtype=torch.int)
        with pytest.raises(ValueError):
            res = functional.map_range(x, 0, 5, 0, 10)

        x = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
        with pytest.raises(ValueError):
            res = functional.map_range(x, 0, 5, 0, 10)

        x = torch.tensor([0, 1, 0, 1], dtype=torch.complex64)
        with pytest.raises(ValueError):
            res = functional.map_range(x, 0, 5, 0, 10)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], device=device)
        res = functional.map_range(x, 0, 1, 0, 10)
        assert res.device == device


class TestValueToIndex:
    def test_value(self):
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5])
        res = functional.value_to_index(x, 0, 1, 11)
        assert torch.all(res == torch.LongTensor([-10, 0, 1, 5])).item()

        x = torch.tensor([-1.0, 0.0, 0.1, 0.5])
        res = functional.value_to_index(x, 1, 0, 2)
        assert torch.all(res == torch.tensor([2, 1, 1, 0])).item()

        x = torch.tensor([0.0, 0.45, 0.55, 4.55])
        res = functional.value_to_index(x, 0, 5, 6)
        assert torch.all(res == torch.tensor([0, 0, 1, 5])).item()

    def test_dtype(self):
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], dtype=torch.float)
        res = functional.value_to_index(x, 0, 1, 10)
        assert res.dtype == torch.long

        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], dtype=torch.float64)
        res = functional.value_to_index(x, 0, 1, 10)
        assert res.dtype == torch.long

        x = torch.tensor([0, 1, 2, 5], dtype=torch.int)
        res = functional.value_to_index(x, 0, 1, 10)
        assert res.dtype == torch.long

        x = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
        res = functional.value_to_index(x, 0, 1, 10)
        assert res.dtype == torch.long

        x = torch.tensor([0, 1, 0, 1], dtype=torch.complex64)
        with pytest.raises(ValueError):
            res = functional.value_to_index(x, 0, 1, 10)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([-1.0, 0.0, 0.1, 0.5], device=device)
        res = functional.value_to_index(x, 0, 1, 10)
        assert res.device == device


class TestIndexToValue:
    def test_value(self):
        x = torch.LongTensor([0, 1, 5, 8])
        res = functional.index_to_value(x, 9, 0, 1)
        assert torch.all(res == torch.tensor([0.0, 0.125, 0.625, 1.0])).item()

        x = torch.LongTensor([-3, 2, 6])
        res = functional.index_to_value(x, 5, 0, 2)
        assert torch.all(res == torch.tensor([-1.5, 1.0, 3.0])).item()

    def test_dtype(self):
        x = torch.LongTensor([0, 1, 5, 8])
        res = functional.index_to_value(x, 9, 0, 1)
        assert res.dtype == torch.float

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.LongTensor([0, 1, 5, 8], device=device)
        res = functional.index_to_value(x, 9, 0, 1)
        assert res.device == device
