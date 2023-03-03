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
from torchhd import MAPTensor

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestBundleSequence:
    def test_creation_dim(self):
        S = structures.BundleSequence(10000)
        assert torch.allclose(S.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        seq = functional.bundle(hv[1], functional.permute(hv[0], shifts=1))

        S = structures.BundleSequence(seq)
        assert torch.allclose(S.value, seq)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_append(self):
        hv = MAPTensor([[ 1., -1.,  1.,  1.,  1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1.,  1., -1.,  1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1.,  1., -1.],
           [ 1., -1., -1.,  1., -1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1., -1., -1.,  1.],
           [ 1.,  1.,  1.,  1., -1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1.,  1.,  1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1.,  1., -1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [-1., -1.,  1., -1.,  1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1.,  1., -1., -1.,  1.]])
        S = structures.BundleSequence(5)

        S.append(hv[0])
        assert torch.allclose(S.value, MAPTensor([ 1., -1.,  1.,  1.,  1.]))

        S.append(hv[1])
        assert torch.allclose(S.value, MAPTensor([0., 0., 0., 2., 2.]))

        S.append(hv[2])
        assert torch.allclose(S.value, MAPTensor([ 3.,  1.,  1., -1.,  3.]))

    def test_appendleft(self):
        hv = MAPTensor([[ 1., -1.,  1.,  1.,  1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1.,  1., -1.,  1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1.,  1., -1.],
           [ 1., -1., -1.,  1., -1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1., -1., -1.,  1.],
           [ 1.,  1.,  1.,  1., -1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1.,  1.,  1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1.,  1., -1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [-1., -1.,  1., -1.,  1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1.,  1., -1., -1.,  1.]])
        S = structures.BundleSequence(5)

        S.appendleft(hv[0])
        assert torch.allclose(S.value, MAPTensor([ 1., -1.,  1.,  1.,  1.]))

        S.appendleft(hv[1])
        assert torch.allclose(S.value, MAPTensor([ 2., -2.,  0.,  2.,  2.]))

        S.appendleft(hv[2])
        assert torch.allclose(S.value, MAPTensor([ 1., -1.,  1.,  3.,  3.]))

    def test_pop(self):
        hv = MAPTensor([[ 1., -1.,  1.,  1.,  1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1.,  1., -1.,  1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1.,  1., -1.],
           [ 1., -1., -1.,  1., -1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1., -1., -1.,  1.],
           [ 1.,  1.,  1.,  1., -1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1.,  1.,  1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1.,  1., -1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [-1., -1.,  1., -1.,  1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1.,  1., -1., -1.,  1.]])
        S = structures.BundleSequence(5)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        S.pop(hv[2])
        assert torch.allclose(S.value, MAPTensor([0., 0., 0., 2., 2.]))

        S.pop(hv[1])
        assert torch.allclose(S.value, MAPTensor([ 1., -1.,  1.,  1.,  1.]))

        S.pop(hv[0])
        assert torch.allclose(S.value, MAPTensor([0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_popleft(self):
        hv = MAPTensor([[ 1., -1.,  1.,  1.,  1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1.,  1., -1.,  1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1., -1., -1.,  1., -1.],
           [ 1., -1., -1.,  1., -1.],
           [ 1.,  1.,  1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [ 1.,  1., -1., -1.,  1.],
           [ 1.,  1.,  1.,  1., -1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1.,  1.,  1., -1.],
           [-1., -1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.],
           [-1.,  1., -1.,  1., -1.],
           [-1.,  1., -1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [-1., -1.,  1., -1.,  1.],
           [ 1.,  1., -1.,  1.,  1.],
           [-1., -1., -1., -1.,  1.],
           [-1.,  1., -1., -1.,  1.]])
        S = structures.BundleSequence(5)

        S.appendleft(hv[0])
        S.appendleft(hv[1])
        S.appendleft(hv[2])

        S.popleft(hv[2])
        assert torch.allclose(S.value, MAPTensor([ 2., -2.,  0.,  2.,  2.]))

        S.popleft(hv[1])
        assert torch.allclose(S.value, MAPTensor([ 1., -1.,  1.,  1.,  1.]))

        S.popleft(hv[0])
        assert torch.allclose(S.value, MAPTensor([0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_replace(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BundleSequence(10000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])
        S.append(hv[3])
        S.append(hv[4])
        S.append(hv[5])
        S.append(hv[6])

        assert functional.cosine_similarity(S[2], hv)[2] > 0.35
        S.replace(2, hv[2], hv[6])
        assert functional.cosine_similarity(S[2], hv)[2] < 0.35
        assert functional.cosine_similarity(S[2], hv)[6] > 0.35

        S2 = structures.BundleSequence.from_tensor(hv[:7])
        assert functional.cosine_similarity(S2[2], hv)[2] > 0.35
        S2.replace(2, hv[2], hv[6])
        assert functional.cosine_similarity(S2[2], hv)[2] < 0.35
        assert functional.cosine_similarity(S2[2], hv)[6] > 0.35

    def test_concat(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 1000, generator=generator)
        S = structures.BundleSequence(1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        S2 = structures.BundleSequence(1000)
        S2.append(hv[0])
        S2.append(hv[1])
        S2.append(hv[2])

        assert len(S) == 3
        assert len(S2) == 3
        S = S.concat(S2)
        assert len(S) == 6

        assert torch.argmax(functional.cosine_similarity(S[0], hv)).item() == 0
        assert torch.argmax(functional.cosine_similarity(S[1], hv)).item() == 1
        assert torch.argmax(functional.cosine_similarity(S[2], hv)).item() == 2
        assert torch.argmax(functional.cosine_similarity(S[3], hv)).item() == 0
        assert torch.argmax(functional.cosine_similarity(S[4], hv)).item() == 1
        assert torch.argmax(functional.cosine_similarity(S[5], hv)).item() == 2

        SS = structures.BundleSequence(1000)

        SS.appendleft(hv[0])
        SS.appendleft(hv[1])
        SS.appendleft(hv[2])

        SS2 = structures.BundleSequence(1000)
        SS2.appendleft(hv[0])
        SS2.appendleft(hv[1])
        SS2.appendleft(hv[2])

        SS = SS.concat(SS2)

        assert torch.argmax(functional.cosine_similarity(SS[0], hv)).item() == 2
        assert torch.argmax(functional.cosine_similarity(SS[1], hv)).item() == 1
        assert torch.argmax(functional.cosine_similarity(SS[2], hv)).item() == 0
        assert torch.argmax(functional.cosine_similarity(SS[3], hv)).item() == 2
        assert torch.argmax(functional.cosine_similarity(SS[4], hv)).item() == 1
        assert torch.argmax(functional.cosine_similarity(SS[5], hv)).item() == 0

    def test_getitem(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 1000, generator=generator)
        S = structures.BundleSequence(1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert torch.argmax(functional.cosine_similarity(S[0], hv)).item() == 0

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 1000, generator=generator)
        S = structures.BundleSequence(1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert len(S) == 3
        S.pop(hv[2])

        assert len(S) == 2

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 1000, generator=generator)
        S = structures.BundleSequence(1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert len(S) == 3
        S.clear()
        assert len(S) == 0
        S.append(hv[0])
        assert len(S) == 1

    def test_from_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.BundleSequence.from_tensor(hv)

        assert torch.argmax(functional.cosine_similarity(S[3], hv)).item() == 3
        assert torch.argmax(functional.cosine_similarity(S[5], hv)).item() == 5
        assert torch.argmax(functional.cosine_similarity(S[1], hv)).item() == 1
