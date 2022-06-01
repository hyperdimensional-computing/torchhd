import pytest
import torch

from torchhd import functional

from .utils import (
    between,
    within,
    torch_dtypes,
    torch_float_dtypes,
    torch_int_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)

seed = 2147483644


class TestIdentity_hv:
    @pytest.mark.parametrize("n", [1, 5564])
    @pytest.mark.parametrize("d", [8425, 10])
    def test_shape(self, n, d):
        hv = functional.identity_hv(n, d)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        if dtype == torch.bool:
            hv = functional.identity_hv(100, 10000, dtype=dtype)
            assert torch.all(hv == False).item()

            return

        hv = functional.identity_hv(100, 10000, dtype=dtype)
        assert torch.all(hv == 1.0).item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(3, 52, device=device)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.identity_hv(3, 26, dtype=dtype)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.identity_hv(3, 26, dtype=dtype)

            return

        hv = functional.identity_hv(3, 52, dtype=dtype)
        assert hv.dtype == dtype

    def test_uses_default_dtype(self):
        hv = functional.identity_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        hv = functional.identity_hv(3, 52)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.identity_hv(3, 52)
        assert hv.dtype == torch.float64

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(
            63, 3567, dtype=torch.bool, requires_grad=False, device=device
        )
        assert hv.dim() == 2
        assert hv.size(0) == 63
        assert hv.size(1) == 3567
        assert hv.requires_grad == False
        assert hv.dtype == torch.bool
        assert hv.device == device


class TestRandom_hv:
    @pytest.mark.parametrize("n", [1, 5564])
    @pytest.mark.parametrize("d", [8425, 10])
    def test_shape(self, n, d):
        hv = functional.random_hv(n, d)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(20, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv2 = functional.random_hv(20, 10000, generator=generator)
        assert torch.all(hv1 == hv2).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if dtype == torch.bool:
            hv = functional.random_hv(100, 10000, dtype=dtype)
            assert torch.all((hv == False) | (hv == True)).item()

            return

        hv = functional.random_hv(100, 10000, generator=generator)
        assert torch.all((hv == -1) | (hv == 1)).item()

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.5, 0.756, 1.0])
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_sparsity(self, sparsity, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if dtype == torch.bool:
            hv = functional.random_hv(
                1000, 10000, generator=generator, dtype=dtype, sparsity=sparsity
            )
            calc_sparsity = torch.sum(hv == False).div(10000 * 1000).item()
            assert within(calc_sparsity, sparsity, 0.001)

            return

        hv = functional.random_hv(
            1000, 10000, generator=generator, dtype=dtype, sparsity=sparsity
        )
        calc_sparsity = torch.sum(hv == 1).div(10000 * 1000).item()
        assert within(calc_sparsity, sparsity, 0.001)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_orthogonality(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        sims = [None] * 100
        for i in range(100):
            hv = functional.random_hv(2, 10000, generator=generator)
            sims[i] = functional.cosine_similarity(hv[0], hv[1].unsqueeze(0))

        sims = torch.cat(sims)
        assert within(sims.mean().item(), 0, 0.001)
        assert sims.std().item() < 0.01

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(3, 52, device=device)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.random_hv(3, 26, dtype=dtype)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.random_hv(3, 26, dtype=dtype)

            return

        hv = functional.random_hv(3, 52, dtype=dtype)
        assert hv.dtype == dtype

    def test_uses_default_dtype(self):
        hv = functional.random_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        hv = functional.random_hv(3, 52)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.random_hv(3, 52)
        assert hv.dtype == torch.float64

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
    @pytest.mark.parametrize("n", [1, 2, 3, 5564])
    @pytest.mark.parametrize("d", [8425, 10])
    def test_shape(self, n, d):
        hv = functional.level_hv(n, d)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.level_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.level_hv(60, 10000, generator=generator)

        assert torch.all(hv1 == hv2).item()

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(seed)

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

    def test_sparsity(self):
        generator = torch.Generator()
        generator.manual_seed(2147287646)

        hv = functional.level_hv(1000, 10000, generator=generator, sparsity=1)
        assert torch.all(hv == 1).item()

        hv = functional.level_hv(1000, 10000, generator=generator, sparsity=0)
        assert torch.all(hv == -1).item()

        sparsity = [None] * 100
        for i in range(100):
            hv = functional.level_hv(100, 10000, generator=generator, sparsity=0.5)
            sparsity[i] = torch.sum(hv == -1).div(10000 * 100)
        assert between(torch.vstack(sparsity).mean().item(), 0.499, 0.501)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(3, 52, device=device)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.level_hv(3, 26, dtype=dtype)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.level_hv(3, 26, dtype=dtype)

            return

        hv = functional.level_hv(3, 52, dtype=dtype)
        assert hv.dtype == dtype

    def test_uses_default_dtype(self):
        hv = functional.level_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        hv = functional.level_hv(3, 52)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.level_hv(3, 52)
        assert hv.dtype == torch.float64

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
    @pytest.mark.parametrize("n", [1, 2, 3, 5564])
    @pytest.mark.parametrize("d", [8425, 10])
    def test_shape(self, n, d):
        hv = functional.circular_hv(n, d)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.circular_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.circular_hv(60, 10000, generator=generator)

        assert torch.all(hv1 == hv2).item()

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(seed)

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

    def test_sparsity(self):
        generator = torch.Generator()
        generator.manual_seed(2147487649)

        hv = functional.circular_hv(100, 10000, generator=generator, sparsity=1)
        assert torch.all(hv == 1).item()

        hv = functional.circular_hv(100, 10000, generator=generator, sparsity=0)
        assert torch.all(hv == -1).item()

        sparsity = [None] * 100
        for i in range(100):
            hv = functional.circular_hv(100, 10000, generator=generator, sparsity=0.5)
            sparsity[i] = torch.sum(hv == -1).div(10000 * 100)
        assert between(torch.vstack(sparsity).mean().item(), 0.499, 0.501)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(3, 52, device=device)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.circular_hv(3, 26, dtype=dtype)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.circular_hv(3, 26, dtype=dtype)

            return

        hv = functional.circular_hv(3, 52, dtype=dtype)
        assert hv.dtype == dtype

    def test_uses_default_dtype(self):
        hv = functional.circular_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        hv = functional.circular_hv(3, 52)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.circular_hv(3, 52)
        assert hv.dtype == torch.float64

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
