import pytest
import torch

from torchhd import functional

from ..utils import (
    within,
    torch_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)

seed = 2147483644


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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.circular_hv(50, 10000, generator=generator, dtype=dtype)
        if dtype == torch.bool:
            assert torch.all(
                (hv == True) | (hv == False)
            ).item(), "values are either 1 or 0"
        elif dtype in torch_complex_dtypes:
            magnitudes = hv.abs()
            assert torch.allclose(
                magnitudes, torch.tensor(1.0, dtype=magnitudes.dtype)
            ), "magnitude must be 1"
        else:
            assert torch.all(
                (hv == -1) | (hv == 1)
            ).item(), "values are either -1 or +1"

        hv = functional.circular_hv(8, 1000000, generator=generator, dtype=dtype)
        if dtype in torch_complex_dtypes:
            sims = functional.cosine_similarity(hv[0], hv)
            sims_diff = sims[:-1] - sims[1:]

            assert torch.all(
                sims_diff.sign() == torch.tensor([1, 1, 1, 1, -1, -1, -1])
            ), "second half must get more similar"

            abs_sims_diff = sims_diff.abs()
            assert torch.all(
                (0.247 < abs_sims_diff) & (abs_sims_diff < 0.253)
            ).item(), "similarity changes linearly"
        else:
            sims = functional.hamming_similarity(hv[0], hv).float() / 1000000
            sims_diff = sims[:-1] - sims[1:]

            assert torch.all(
                sims_diff.sign() == torch.tensor([1, 1, 1, 1, -1, -1, -1])
            ), "second half must get more similar"

            abs_sims_diff = sims_diff.abs()
            assert torch.all(
                (0.124 < abs_sims_diff) & (abs_sims_diff < 0.126)
            ).item(), "similarity changes linearly"

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.756, 1.0])
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_sparsity(self, sparsity, dtype):
        if not supported_dtype(dtype):
            return

        if dtype in torch_complex_dtypes:
            # Complex hypervectors don't support sparsity.
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.circular_hv(
            1000, 10000, generator=generator, dtype=dtype, sparsity=sparsity
        )

        if dtype == torch.bool:
            calc_sparsity = torch.sum(hv == False).div(10000 * 1000).item()
        else:
            calc_sparsity = torch.sum(hv == 1).div(10000 * 1000).item()

        assert within(calc_sparsity, sparsity, 0.01)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, dtype):
        if not supported_dtype(dtype):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(3, 52, device=device, dtype=dtype)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
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
