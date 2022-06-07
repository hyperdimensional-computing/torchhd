import pytest
import torch

from torchhd import functional

from ..utils import (
    torch_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)

seed = 2147483644


class TestIdentity_hv:
    @pytest.mark.parametrize("n", [1, 3, 5564])
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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, dtype):
        if not supported_dtype(dtype):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(3, 52, device=device, dtype=dtype)
        assert hv.device == device

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
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
