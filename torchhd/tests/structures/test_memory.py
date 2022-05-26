import pytest
import torch
import string

from torchhd import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestMemory:
    def test_creation(self):
        M = structures.Memory()

        assert M.keys == []
        assert M.values == []

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert torch.equal(M.keys[0], keys_hv[0])
        assert torch.equal(M.keys[1], keys_hv[1])
        assert torch.equal(M.keys[2], keys_hv[2])
        assert M.values[0] == letters[0]
        assert M.values[1] == letters[1]
        assert M.values[2] == letters[2]

    def test_index(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert M.index(keys_hv[0]) == 0
        assert M.index(keys_hv[1]) == 1
        assert M.index(keys_hv[2]) == 2

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert len(M) == 3
        del M[keys_hv[0]]

        assert len(M) == 2

        M.add(keys_hv[0], letters[0])
        assert len(M) == 3

    def test_getitem(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert M[keys_hv[0]][1] == letters[0]
        assert M[keys_hv[1]][1] == letters[1]
        assert M[keys_hv[2]][1] == letters[2]

    def test_setitem(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert len(M) == 3
        assert M[keys_hv[0]][1] == letters[0]
        assert M[keys_hv[1]][1] == letters[1]
        assert M[keys_hv[2]][1] == letters[2]

        M[keys_hv[0]] = letters[3]
        assert len(M) == 3
        assert M[keys_hv[0]][1] == letters[3]

    def test_delitem(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator)

        M = structures.Memory()
        M.add(keys_hv[0], letters[0])
        M.add(keys_hv[1], letters[1])
        M.add(keys_hv[2], letters[2])

        assert len(M) == 3
        assert M[keys_hv[0]][1] == letters[0]
        assert M[keys_hv[1]][1] == letters[1]
        assert M[keys_hv[2]][1] == letters[2]

        del M[keys_hv[0]]
        try:
            M[keys_hv[0]]
        except IndexError:
            assert True

        assert M[keys_hv[1]][1] == letters[1]
        assert M[keys_hv[2]][1] == letters[2]
        assert len(M) == 2
