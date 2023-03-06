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

from torchhd import VSATensor

class TestVSATensor:
    def test_empty(self):
        with pytest.raises(NotImplementedError):
            VSATensor.empty(4, 525)

    def test_identity(self):
        with pytest.raises(NotImplementedError):
            VSATensor.identity(4, 525)

    def test_random(self):
        with pytest.raises(NotImplementedError):
            VSATensor.random(4, 525)

    def test_bundle(self):
        a = torch.randn(100).as_subclass(VSATensor)
        b = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.bundle(b)

    def test_multibundle(self):
        a = torch.randn(10, 100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.multibundle()

    def test_bind(self):
        a = torch.randn(100).as_subclass(VSATensor)
        b = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.bind(b)

    def test_multibind(self):
        a = torch.randn(10, 100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.multibind()

    def test_inverse(self):
        a = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.inverse()

    def test_negative(self):
        a = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.negative()

    def test_permute(self):
        a = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.permute()

    def test_dot_similarity(self):
        a = torch.randn(100).as_subclass(VSATensor)
        b = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.dot_similarity(b)

    def test_cosine_similarity(self):
        a = torch.randn(100).as_subclass(VSATensor)
        b = torch.randn(100).as_subclass(VSATensor)

        with pytest.raises(NotImplementedError):
            a.cosine_similarity(b)
