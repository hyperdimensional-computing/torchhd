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


class TestBindSequence:
    def test_creation_dim(self):
        S = structures.BindSequence(10000)
        assert torch.equal(S.value, torch.ones(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)

        S = structures.BindSequence(hv[0])
        assert torch.equal(S.value, hv[0])

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_append(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5

    def test_appendleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.appendleft(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5

    def test_pop(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        S.append(hv[1])
        S.pop(hv[1])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5
        S.pop(hv[0])
        S.append(hv[2])
        assert functional.cosine_similarity(S.value, hv)[2] > 0.5
        S.append(hv[3])
        S.pop(hv[3])
        assert functional.cosine_similarity(S.value, hv)[2] > 0.5

    def test_popleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.appendleft(hv[0])
        S.appendleft(hv[1])
        S.popleft(hv[1])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5
        S.popleft(hv[0])
        S.appendleft(hv[2])
        assert functional.cosine_similarity(S.value, hv)[2] > 0.5
        S.appendleft(hv[3])
        S.popleft(hv[3])
        assert functional.cosine_similarity(S.value, hv)[2] > 0.5

    def test_replace(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5
        S.replace(0, hv[0], hv[1])
        assert functional.cosine_similarity(S.value, hv)[1] > 0.5

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        assert len(S) == 4
        S.pop(hv[0])
        S.pop(hv[0])
        S.pop(hv[0])
        assert len(S) == 1
        S.pop(hv[0])
        assert len(S) == 0
        S.append(hv[0])
        assert len(S) == 1

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        assert len(S) == 4
        S.clear()
        assert len(S) == 0
