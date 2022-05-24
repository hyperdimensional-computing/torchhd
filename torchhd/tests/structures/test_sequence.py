import pytest
import torch
import string

from ... import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestSequence:
    def test_creation_dim(self):
        S = structures.Sequence(dim_or_input=10000)
        assert torch.equal(S.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        seq = functional.bundle(hv[1], functional.permute(hv[0], shifts=1))

        S = structures.Sequence(dim_or_input=seq)
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
        S = structures.Sequence(dim_or_input=4)

        S.append(hv[0])
        assert torch.equal(S.value, torch.tensor([1., -1., 1., 1.]))

        S.append(hv[1])
        assert torch.equal(S.value, torch.tensor([2., 2., -2., 2.]))

        S.append(hv[2])
        assert torch.equal(S.value, torch.tensor([3., 3., 3., -3.]))

    def test_appendleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.Sequence(dim_or_input=4)

        S.appendleft(hv[0])
        assert torch.equal(S.value, torch.tensor([1., -1., 1., 1.]))

        S.appendleft(hv[1])
        assert torch.equal(S.value, torch.tensor([2., 0., 2., 0.]))

        S.appendleft(hv[2])
        assert torch.equal(S.value, torch.tensor([3., -1., 3., 1.]))

    def test_pop(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.Sequence(dim_or_input=4)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        S.pop(hv[2])
        assert torch.equal(S.value, torch.tensor([2., 2., -2., 2.]))

        S.pop(hv[1])
        assert torch.equal(S.value, torch.tensor([1., -1., 1., 1.]))

        S.pop(hv[0])
        assert torch.equal(S.value, torch.tensor([0., 0., 0., 0.]))

    def test_popleft(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 4, generator=generator)
        S = structures.Sequence(dim_or_input=4)

        S.appendleft(hv[0])
        S.appendleft(hv[1])
        S.appendleft(hv[2])

        S.popleft(hv[2])
        assert torch.equal(S.value, torch.tensor([2., 0., 2., 0.]))

        S.popleft(hv[1])
        assert torch.equal(S.value, torch.tensor([1., -1., 1., 1.]))

        S.popleft(hv[0])
        assert torch.equal(S.value, torch.tensor([0., 0., 0., 0.]))

    def test_replace(self):
        """ NOT WORKING """
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        S = structures.Sequence(dim_or_input=10000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])
        S.append(hv[3])
        S.append(hv[4])
        S.append(hv[5])
        S.append(hv[6])

        print(functional.cosine_similarity(S[2], hv))
        """tensor([-5.7600e-03, -5.5296e-03, 3.6933e-01, 2.3040e-04, -5.0688e-03,
                -8.6785e-03, 4.6080e-04, 7.6800e-03, 2.1735e-02, -5.4528e-03,
                1.0061e-02, -1.2595e-02, -8.2945e-03, 2.0659e-02, 4.0704e-03,
                2.6880e-03, 1.2134e-02, -4.6848e-03, -3.3792e-03, 6.1440e-03,
                -4.2240e-03, 6.0672e-03, 6.1440e-04, -7.9872e-03, 1.2058e-02,
                -8.0641e-03])"""
        S.replace(2, hv[2], hv[6])
        print(functional.cosine_similarity(S[2], hv))
        """tensor([-3.9211e-03, -9.5323e-03, 3.2261e-01, -9.4646e-03, -1.0141e-03,
                -2.5690e-03, 4.1239e-03, 3.0422e-03, 1.3791e-02, -4.1915e-03,
                3.9211e-03, -6.2872e-03, -6.4224e-03, 2.1633e-02, 9.8703e-03,
                3.6506e-03, 1.4400e-02, -4.1915e-03, 2.0281e-04, 1.2912e-02,
                -4.8675e-03, 2.8394e-03, 2.2310e-03, -1.1696e-02, 1.4467e-02,
                -8.6534e-03])"""

        hv1 = functional.random_hv(10, 10000)
        S2 = structures.Sequence.from_tensor(hv1)
        print(functional.cosine_similarity(S2[2], hv1))
        """tensor([ 0.0065,  0.0105,  0.0023,  0.0025,  0.0022, -0.0155,  0.0092, -0.0094,
         0.0195,  0.0183])"""
        S2.replace(2, hv1[2], hv1[5])
        print(functional.cosine_similarity(S2[2], hv1))
        """tensor([ 0.0071,  0.0081,  0.0038,  0.0091,  0.0062, -0.0171,  0.0068, -0.0085,
         0.0168,  0.0205])"""

    def test_concat(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(8, 1000, generator=generator)
        S = structures.Sequence(dim_or_input=1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        S2 = structures.Sequence(dim_or_input=1000)
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

        SS = structures.Sequence(dim_or_input=1000)

        SS.appendleft(hv[0])
        SS.appendleft(hv[1])
        SS.appendleft(hv[2])

        SS2 = structures.Sequence(dim_or_input=1000)
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
        S = structures.Sequence(dim_or_input=1000)

        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert torch.argmax(functional.cosine_similarity(S[0], hv)).item() == 0

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(8, 1000, generator=generator)
        S = structures.Sequence(dim_or_input=1000)


        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert len(S) == 3
        S.pop(hv[2])

        assert len(S) == 2

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(8, 1000, generator=generator)
        S = structures.Sequence(dim_or_input=1000)


        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        assert len(S) == 3
        S.clear()
        assert len(S) == 0
        S.append(hv[0])
        assert len(S) == 1

    def test_from_tensor(self):
        """ NOT WORKING """

        """generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        seq = functional.bundle(hv[1], functional.permute(hv[0], shifts=1))

        S = structures.Sequence.from_tensor(seq)
        assert torch.equal(S.value, seq)"""
