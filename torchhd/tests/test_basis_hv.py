import pytest
import torch

from torchhd import functional

from .utils import between


class TestIdentity_hv:
    def test_shape(self):
        hv = functional.identity_hv(13, 2556)
        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

        hv = functional.identity_hv(101, 85390)
        assert hv.dim() == 2
        assert hv.size(0) == 101
        assert hv.size(1) == 85390

    def test_value(self):
        hv = functional.identity_hv(100, 10000)
        assert torch.all(hv == 1.0).item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(3, 52, device=device)
        assert hv.device == device

    def test_dtype(self):
        hv = functional.identity_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.identity_hv(3, 532, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.identity_hv(865, 452, dtype=torch.long)
        assert hv.dtype == torch.long

        hv = functional.identity_hv(32, 582, dtype=torch.float)
        assert hv.dtype == torch.float

        hv = functional.identity_hv(6, 1252, dtype=torch.float64)
        assert hv.dtype == torch.float64

        with pytest.raises(NotImplementedError):
            hv = functional.identity_hv(3, 52, dtype=torch.bool)

        with pytest.raises(NotImplementedError):
            hv = functional.identity_hv(3, 520, dtype=torch.complex64)

        with pytest.raises(ValueError):
            hv = functional.identity_hv(9, 520, dtype=torch.uint8)

    def test_requires_grad(self):
        hv = functional.identity_hv(3, 52, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.identity_hv(3, 52, requires_grad=True)
        assert hv.requires_grad == True

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(
            6, 10000, dtype=torch.float16, requires_grad=True, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float16
        assert hv.device == device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(
            63, 3567, dtype=torch.long, requires_grad=False, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 63
        assert hv.size(1) == 3567
        assert hv.requires_grad == False
        assert hv.dtype == torch.long
        assert hv.device == device


class TestRandom_hv:
    def test_shape(self):
        hv = functional.random_hv(13, 2556)
        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

        hv = functional.random_hv(13, 2556)
        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.random_hv(20, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv2 = functional.random_hv(20, 10000, generator=generator)
        assert torch.all(hv1 == hv2).item()

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(100, 10000, generator=generator)
        assert torch.all((hv == -1) | (hv == 1)).item()

    def test_sparsity(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(100, 10000, generator=generator, sparsity=1)
        assert torch.all(hv == 1).item()

        hv = functional.random_hv(100, 10000, generator=generator, sparsity=0)
        assert torch.all(hv == -1).item()

        hv = functional.random_hv(100, 10000, generator=generator, sparsity=0.5)
        assert between(torch.sum(hv == -1).div(10000 * 100).item(), 0.499, 0.501)

    def test_orthogonality(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        sims = [None] * 1000
        for i in range(1000):
            hv = functional.random_hv(2, 10000, generator=generator)
            sims[i] = functional.cosine_similarity(hv[0], hv[1].unsqueeze(0))

        sims = torch.cat(sims)
        assert between(
            sims.mean().item(), -0.001, 0.001
        ), "similarity is approximately 0"
        assert sims.std().item() < 0.01

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(3, 52, device=device)
        assert hv.device == device

    def test_dtype(self):
        hv = functional.random_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.random_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.random_hv(3, 52, dtype=torch.long)
        assert hv.dtype == torch.long

        hv = functional.random_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

        hv = functional.random_hv(3, 52, dtype=torch.float64)
        assert hv.dtype == torch.float64

        with pytest.raises(NotImplementedError):
            hv = functional.random_hv(3, 52, dtype=torch.bool)

        with pytest.raises(NotImplementedError):
            hv = functional.random_hv(3, 52, dtype=torch.complex128)

        with pytest.raises(ValueError):
            hv = functional.random_hv(3, 52, dtype=torch.uint8)

    def test_requires_grad(self):
        hv = functional.random_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.random_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            # Only floating point values can require gradients
            hv = functional.random_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(
            84,
            2403,
            dtype=torch.long,
            requires_grad=False,
            device=device,
            sparsity=0.8,
        )
        assert hv.dim() == 2
        assert hv.size(0) == 84
        assert hv.size(1) == 2403
        assert hv.requires_grad == False
        assert hv.dtype == torch.long
        assert hv.device == device


class TestLevel_hv:
    def test_shape(self):
        hv = functional.level_hv(13, 2556)
        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

        hv = functional.level_hv(130, 3530)
        assert hv.dim() == 2
        assert hv.size(0) == 130
        assert hv.size(1) == 3530

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.level_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv2 = functional.level_hv(60, 10000, generator=generator)

        assert torch.all(hv1 == hv2).item()

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.level_hv(50, 10000, generator=generator)
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"

        # look at the similarity profile w.r.t. the first hypervector
        sims = functional.cosine_similarity(hv[0], hv)
        sims_diff = sims[:-1] - sims[1:]
        assert torch.all(sims_diff > 0).item(), "similarity must be decreasing"

        sims = [None] * 1000
        for i in range(1000):
            hv = functional.level_hv(5, 10000, generator=generator)
            sims[i] = functional.cosine_similarity(hv[0], hv)

        sims = torch.vstack(sims)
        sims_diff = torch.mean(sims[:, :-1] - sims[:, 1:], dim=0)
        assert torch.all(
            (0.249 < sims_diff) & (sims_diff < 0.251)
        ).item(), "similarity decreases linearly"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(3, 52, device=device)
        assert hv.device == device

    def test_dtype(self):
        hv = functional.level_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.level_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.level_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

        hv = functional.level_hv(3, 52, dtype=torch.long)
        assert hv.dtype == torch.long

        hv = functional.level_hv(3, 52, dtype=torch.float64)
        assert hv.dtype == torch.float64

        with pytest.raises(NotImplementedError):
            functional.level_hv(3, 26, dtype=torch.bool)

        with pytest.raises(NotImplementedError):
            functional.level_hv(3, 26, dtype=torch.complex128)

        with pytest.raises(ValueError):
            functional.level_hv(3, 26, dtype=torch.uint8)

    def test_requires_grad(self):
        hv = functional.level_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.level_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            # Cannot require gradients for integer or boolean types
            hv = functional.level_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(
            145, 1526, dtype=torch.long, requires_grad=False, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 145
        assert hv.size(1) == 1526
        assert hv.requires_grad == False
        assert hv.dtype == torch.long
        assert hv.device == device


class TestCircular_hv:
    def test_shape(self):
        hv = functional.circular_hv(13, 2556)
        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

        hv = functional.circular_hv(724, 9345)
        assert hv.dim() == 2
        assert hv.size(0) == 724
        assert hv.size(1) == 9345

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.circular_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv2 = functional.circular_hv(60, 10000, generator=generator)

        assert torch.all(hv1 == hv2).item()

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.circular_hv(50, 10000, generator=generator)
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"

        sims = [None] * 1000
        for i in range(1000):
            hv = functional.circular_hv(8, 10000, generator=generator)
            sims[i] = functional.cosine_similarity(hv[0], hv)

        sims = torch.vstack(sims)
        sims_diff = torch.mean(sims[:, :-1] - sims[:, 1:], dim=0)
        assert torch.all(
            sims_diff.sign() == torch.tensor([1, 1, 1, 1, -1, -1, -1])
        ), "second half must get more similar"

        abs_sims_diff = sims_diff.abs()
        assert torch.all(
            (0.249 < abs_sims_diff) & (abs_sims_diff < 0.251)
        ).item(), "similarity decreases linearly"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(3, 52, device=device)
        assert hv.device == device

    def test_dtype(self):
        hv = functional.circular_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.circular_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.circular_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

        hv = functional.circular_hv(3, 52, dtype=torch.long)
        assert hv.dtype == torch.long

        hv = functional.circular_hv(3, 52, dtype=torch.float64)
        assert hv.dtype == torch.float64

        with pytest.raises(NotImplementedError):
            functional.circular_hv(2, 62, dtype=torch.bool)

        with pytest.raises(NotImplementedError):
            functional.circular_hv(2, 62, dtype=torch.complex128)

        with pytest.raises(ValueError):
            functional.circular_hv(2, 62, dtype=torch.uint8)

    def test_requires_grad(self):
        hv = functional.circular_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.circular_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            # Integers and Booleans do not allow gradients
            hv = functional.circular_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(
            8, 24256, dtype=torch.long, requires_grad=False, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 8
        assert hv.size(1) == 24256
        assert hv.requires_grad == False
        assert hv.dtype == torch.long
        assert hv.device == device
