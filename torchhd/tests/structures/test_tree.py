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
import string

from torchhd import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestTree:
    def test_creation_dim(self):
        T = structures.Tree(10000)
        assert torch.allclose(T.value, torch.zeros(10000))

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add_leaf(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        T = structures.Tree(10000)
        T.add_leaf(hv[0], ["l", "l"])
        assert (
            torch.argmax(
                functional.cosine_similarity(T.get_leaf(["l", "l"]), hv)
            ).item()
            == 0
        )
        T.add_leaf(hv[1], ["l", "r"])
        assert (
            torch.argmax(
                functional.cosine_similarity(T.get_leaf(["l", "r"]), hv)
            ).item()
            == 1
        )

    def test_get_leaf(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        T = structures.Tree(10000)
        T.add_leaf(hv[0], ["l", "l"])
        assert (
            torch.argmax(
                functional.cosine_similarity(T.get_leaf(["l", "l"]), hv)
            ).item()
            == 0
        )
        T.add_leaf(hv[1], ["l", "r"])
        assert (
            torch.argmax(
                functional.cosine_similarity(T.get_leaf(["l", "r"]), hv)
            ).item()
            == 1
        )

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 10, generator=generator)
        T = structures.Tree(10)

        T.add_leaf(hv[0], ["l", "l"])
        T.add_leaf(hv[1], ["l", "r"])

        T.clear()
        assert torch.allclose(
            T.value, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
