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


class TestMultiset:
    def test_creation_dim(self):
        M = structures.Multiset(10000)
        assert torch.equal(M.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 10000, generator=generator)
        multiset = functional.multiset(keys_hv)

        M = structures.Multiset(multiset)
        assert torch.equal(M.value, multiset)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset(4)

        M.add(keys_hv[0])
        assert torch.equal(M.value, torch.tensor([1.0, -1.0, 1.0, 1.0]))

        M.add(keys_hv[1])
        assert torch.equal(M.value, torch.tensor([2.0, 0.0, 0.0, 2.0]))

        M.add(keys_hv[2])
        assert torch.equal(M.value, torch.tensor([3.0, 1.0, 1.0, 1.0]))

    def test_remove(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset(4)

        M.add(keys_hv[0])
        M.add(keys_hv[1])

        assert M.contains(keys_hv[0]) > torch.tensor([0.5])

        M.remove(keys_hv[0])
        assert M.contains(keys_hv[0]) < torch.tensor([0.1])
        assert M.contains(keys_hv[1]) > torch.tensor([0.5])
        assert M.remove(keys_hv[0]) is None

    def test_contains(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset(4)

        M.add(keys_hv[0])
        M.add(keys_hv[0])
        M.add(keys_hv[0])
        M.add(keys_hv[1])
        assert M.contains(keys_hv[0]) > torch.tensor([0.8])
        M.remove(keys_hv[0])
        assert M.contains(keys_hv[0]) > torch.tensor([0.8])
        M.remove(keys_hv[0])
        assert M.contains(keys_hv[0]) > torch.tensor([0.7])
        M.remove(keys_hv[0])
        assert M.contains(keys_hv[0]) < torch.tensor([0.1])
        M.remove(keys_hv[1])
        assert M.contains(keys_hv[1]) < torch.tensor([0.1])

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset(4)

        M.add(keys_hv[0])
        M.add(keys_hv[0])
        M.add(keys_hv[1])

        assert len(M) == 3
        M.remove(keys_hv[0])

        assert len(M) == 2

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset(4)

        M.add(keys_hv[0])
        M.add(keys_hv[0])
        M.add(keys_hv[1])

        M.clear()

        assert M.contains(keys_hv[0]) < torch.tensor([0.1])
        assert M.contains(keys_hv[1]) < torch.tensor([0.1])

        M.add(keys_hv[0])
        assert M.contains(keys_hv[0]) > torch.tensor([0.8])

    def test_from_ngrams(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 3, generator=generator)
        M = structures.Multiset.from_ngrams(keys_hv)

        assert torch.equal(M.value, torch.tensor([0.0, 4.0, 0.0]))

    def test_from_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random(len(letters), 4, generator=generator)
        M = structures.Multiset.from_tensor(keys_hv)
        assert torch.equal(M.value, torch.tensor([2.0, 10.0, 4.0, 2.0]))
