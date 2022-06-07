import pytest
import torch
import string

from torchhd import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestBundleSequence:
    def test_creation_dim(self):
        S = structures.BundleSequence(10000)
        assert torch.equal(S.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        seq = functional.bundle(hv[1], functional.permute(hv[0], shifts=1))

        S = structures.BundleSequence(seq)
        assert torch.equal(S.value, seq)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_append(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.BundleSequence(4)

        S.append(hv[0])
        assert torch.equal(S.value, torch.tensor([1.0, -1.0, 1.0, 1.0]))

        S.append(hv[1])
        assert torch.equal(S.value, torch.tensor([2.0, 2.0, -2.0, 2.0]))

        S.append(hv[2])
        assert torch.equal(S.value, torch.tensor([3.0, 3.0, 3.0, -3.0]))

    def test_appendleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.BundleSequence(4)

        S.appendleft(hv[0])
        assert torch.equal(S.value, torch.tensor([1.0, -1.0, 1.0, 1.0]))

        S.appendleft(hv[1])
        assert torch.equal(S.value, torch.tensor([2.0, 0.0, 2.0, 0.0]))

        S.appendleft(hv[2])
        assert torch.equal(S.value, torch.tensor([3.0, -1.0, 3.0, 1.0]))

    def test_pop(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.BundleSequence(4)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        S.pop(hv[2])
        assert torch.equal(S.value, torch.tensor([2.0, 2.0, -2.0, 2.0]))

        S.pop(hv[1])
        assert torch.equal(S.value, torch.tensor([1.0, -1.0, 1.0, 1.0]))

        S.pop(hv[0])
        assert torch.equal(S.value, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    def test_popleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.BundleSequence(4)

        S.appendleft(hv[0])
        S.appendleft(hv[1])
        S.appendleft(hv[2])

        S.popleft(hv[2])
        assert torch.equal(S.value, torch.tensor([2.0, 0.0, 2.0, 0.0]))

        S.popleft(hv[1])
        assert torch.equal(S.value, torch.tensor([1.0, -1.0, 1.0, 1.0]))

        S.popleft(hv[0])
        assert torch.equal(S.value, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    def test_replace(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(8, 1000, generator=generator)
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
        hv = functional.random_hv(8, 1000, generator=generator)
        S = structures.BundleSequence(1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert torch.argmax(functional.cosine_similarity(S[0], hv)).item() == 0

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(8, 1000, generator=generator)
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
        hv = functional.random_hv(8, 1000, generator=generator)
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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.BundleSequence.from_tensor(hv)

        assert torch.argmax(functional.cosine_similarity(S[3], hv)).item() == 3
        assert torch.argmax(functional.cosine_similarity(S[5], hv)).item() == 5
        assert torch.argmax(functional.cosine_similarity(S[1], hv)).item() == 1
