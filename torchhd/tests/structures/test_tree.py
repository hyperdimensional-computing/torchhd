import pytest
import torch
import string

from torchhd import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestTree:
    def test_creation_dim(self):
        T = structures.Tree(10000)
        assert torch.equal(T.value, torch.zeros(10000))

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add_leaf(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(len(letters), 10000, generator=generator)
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
        hv = functional.random_hv(8, 10, generator=generator)
        T = structures.Tree(10)

        T.add_leaf(hv[0], ["l", "l"])
        T.add_leaf(hv[1], ["l", "r"])

        T.clear()
        assert torch.equal(
            T.value, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
