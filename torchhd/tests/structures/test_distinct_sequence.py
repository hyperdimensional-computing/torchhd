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
        hv = functional.random_hv(len(letters), 10000, generator=generator)

        S = structures.BindSequence(hv[0])
        assert torch.equal(S.value, hv[0])

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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5

    def test_appendleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.appendleft(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5

    def test_pop(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        assert functional.cosine_similarity(S.value, hv)[0] > 0.5
        S.replace(0, hv[0], hv[1])
        assert functional.cosine_similarity(S.value, hv)[1] > 0.5

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.BindSequence(10000)
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        S.append(hv[0])
        assert len(S) == 4
        S.clear()
        assert len(S) == 0
