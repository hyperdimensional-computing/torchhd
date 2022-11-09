import pytest
import torch

import torchhd
from torchhd import functional

from ..utils import (
    torch_dtypes,
    supported_dtype,
    vsa_models,
)

seed = 2147483644


class TestEmpty_hv:
    @pytest.mark.parametrize("n", [1, 3, 55])
    @pytest.mark.parametrize("d", [84, 10])
    @pytest.mark.parametrize("model", vsa_models)
    def test_shape(self, n, d, model):
        hv = functional.empty_hv(n, d, model)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("model", vsa_models)
    def test_value(self, dtype, model):

        if not supported_dtype(dtype, model):
            with pytest.raises(ValueError):
                functional.empty_hv(3, 26, model, dtype=dtype)

            return

        hv = functional.empty_hv(8, 26, model, dtype=dtype)
        assert hv.requires_grad == False
        assert hv.dim() == 2
        assert hv.size(0) == 8
        assert hv.size(1) == 26

        if model == torchhd.BSC:
            assert torch.all((hv == False) | (hv == True)).item()

        else:
            hv = functional.empty_hv(8, 26, model, dtype=dtype)
            assert torch.all(hv == 0.0).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("model", vsa_models)
    def test_device(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.empty_hv(3, 52, model, device=device, dtype=dtype)
        assert hv.device == device

    def test_uses_default_dtype(self):
        hv = functional.empty_hv(3, 52, torchhd.BSC)
        assert hv.dtype == torch.bool

        torch.set_default_dtype(torch.float32)
        hv = functional.empty_hv(3, 52, torchhd.MAP)
        assert hv.dtype == torch.float32
        hv = functional.empty_hv(3, 52, torchhd.HRR)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.empty_hv(3, 52, torchhd.MAP)
        assert hv.dtype == torch.float64
        hv = functional.empty_hv(3, 52, torchhd.HRR)
        assert hv.dtype == torch.float64

        hv = functional.empty_hv(3, 52, torchhd.FHRR)
        assert hv.dtype == torch.complex64

    def test_requires_grad(self):
        hv = functional.empty_hv(3, 52, torchhd.MAP, requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.empty_hv(3, 52, torchhd.HRR, requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.empty_hv(3, 52, torchhd.FHRR, requires_grad=True)
        assert hv.requires_grad == True
