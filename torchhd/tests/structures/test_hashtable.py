import pytest
import torch
import string

from torchhd import structures, functional

seed_key = 2147483644
seed_value = 2147483622
letters = list(string.ascii_lowercase)


class TestHashtable:
    def test_creation_dim(self):
        H = structures.HashTable(10000)
        assert torch.equal(H.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        hash_v1 = functional.bind(keys_hv[0], values_hv[0])
        hash_v2 = functional.bind(keys_hv[1], values_hv[1])
        hasht = functional.bundle(hash_v1, hash_v2)

        H = structures.HashTable(hasht)
        assert torch.equal(H.value, hasht)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed_key)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed_key)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])

        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(True),
        )
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[1]], values_hv) > 0.5)[1],
            torch.tensor(True),
        )

    def test_remove(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])

        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(True),
        )

        H.remove(keys_hv[0], values_hv[0])
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) < 0.2)[0],
            torch.tensor(True),
        )

    def test_get(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])
        H.add(keys_hv[2], values_hv[2])

        assert torch.equal(
            (functional.cosine_similarity(H.get(keys_hv[0]), values_hv) > 0.5)[0],
            torch.tensor(True),
        )
        assert torch.equal(
            (functional.cosine_similarity(H.get(keys_hv[1]), values_hv) > 0.5)[1],
            torch.tensor(True),
        )
        assert torch.equal(
            (functional.cosine_similarity(H.get(keys_hv[2]), values_hv) > 0.5)[2],
            torch.tensor(True),
        )
        assert torch.equal(
            torch.all(
                (functional.cosine_similarity(H.get(values_hv[2]), values_hv) > 0.5)
                == False
            ),
            torch.tensor(True),
        )

    def test_getitem(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])
        H.add(keys_hv[2], values_hv[2])

        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(True),
        )
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[1]], values_hv) > 0.5)[1],
            torch.tensor(True),
        )
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[2]], values_hv) > 0.5)[2],
            torch.tensor(True),
        )
        assert torch.equal(
            torch.all(
                (functional.cosine_similarity(H[values_hv[2]], values_hv) > 0.5)
                == False
            ),
            torch.tensor(True),
        )

    def test_replace(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])
        H.add(keys_hv[2], values_hv[2])

        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(True),
        )
        H.replace(keys_hv[0], values_hv[0], values_hv[1])
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[1],
            torch.tensor(True),
        )

    def test_length(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])
        H.add(keys_hv[2], values_hv[2])

        assert len(H) == 3
        H.remove(keys_hv[0], values_hv[0])

        assert len(H) == 2

    def test_clear(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(len(letters), 10000, generator=generator_key)
        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(len(letters), 10000, generator=generator_value)

        H = structures.HashTable(10000)
        H.add(keys_hv[0], values_hv[0])
        H.add(keys_hv[1], values_hv[1])
        H.add(keys_hv[2], values_hv[2])
        assert len(H) == 3
        H.clear()
        assert len(H) == 0
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(False),
        )
        H.add(keys_hv[0], values_hv[0])
        assert torch.equal(
            (functional.cosine_similarity(H[keys_hv[0]], values_hv) > 0.5)[0],
            torch.tensor(True),
        )

    def test_from_tensor(self):
        generator_key = torch.Generator()
        generator_key.manual_seed(seed_key)
        keys_hv = functional.random_hv(2, 3, generator=generator_key)

        generator_value = torch.Generator()
        generator_value.manual_seed(seed_value)
        values_hv = functional.random_hv(2, 3, generator=generator_value)

        H = structures.HashTable.from_tensors(keys_hv, values_hv)
        assert torch.equal(H.value, torch.tensor([2.0, 0.0, 0.0]))
