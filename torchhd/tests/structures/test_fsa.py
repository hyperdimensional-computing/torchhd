import pytest
import torch
import string

from torchhd import structures, functional

seed = 2147483644
seed1 = 2147483643
letters = list(string.ascii_lowercase)


class TestFSA:
    def test_creation_dim(self):
        F = structures.FiniteStateAutomata(10000)
        assert torch.equal(F.value, torch.zeros(10000))

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add_transition(self):
        generator = torch.Generator()
        generator1 = torch.Generator()
        generator.manual_seed(seed)
        generator1.manual_seed(seed1)
        tokens = functional.random_hv(10, 10, generator=generator)
        actions = functional.random_hv(10, 10, generator=generator1)

        F = structures.FiniteStateAutomata(10)

        F.add_transition(tokens[0], actions[1], actions[2])
        assert torch.equal(
            F.value,
            torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0]),
        )
        F.add_transition(tokens[1], actions[1], actions[3])
        assert torch.equal(
            F.value, torch.tensor([0.0, 0.0, -2.0, 2.0, 0.0, 2.0, 0.0, -2.0, -2.0, 0.0])
        )
        F.add_transition(tokens[2], actions[1], actions[3])
        assert torch.equal(
            F.value,
            torch.tensor([1.0, 1.0, -3.0, 1.0, 1.0, 3.0, -1.0, -1.0, -1.0, 1.0]),
        )

    def test_transition(self):
        generator = torch.Generator()
        generator1 = torch.Generator()
        generator.manual_seed(seed)
        generator1.manual_seed(seed1)
        tokens = functional.random_hv(10, 10, generator=generator)
        states = functional.random_hv(10, 10, generator=generator1)

        F = structures.FiniteStateAutomata(10)

        F.add_transition(tokens[0], states[1], states[2])
        F.add_transition(tokens[1], states[1], states[3])
        F.add_transition(tokens[2], states[1], states[5])

        assert (
            torch.argmax(
                functional.cosine_similarity(F.transition(states[1], tokens[0]), states)
            ).item()
            == 2
        )
        assert (
            torch.argmax(
                functional.cosine_similarity(F.transition(states[1], tokens[1]), states)
            ).item()
            == 3
        )
        assert (
            torch.argmax(
                functional.cosine_similarity(F.transition(states[1], tokens[2]), states)
            ).item()
            == 5
        )

    def test_clear(self):
        generator = torch.Generator()
        generator1 = torch.Generator()
        generator.manual_seed(seed)
        generator1.manual_seed(seed1)
        tokens = functional.random_hv(10, 10, generator=generator)
        states = functional.random_hv(10, 10, generator=generator1)

        F = structures.FiniteStateAutomata(10)

        F.add_transition(tokens[0], states[1], states[2])
        F.add_transition(tokens[1], states[1], states[3])
        F.add_transition(tokens[2], states[1], states[5])

        F.clear()
        assert torch.equal(
            F.value, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
