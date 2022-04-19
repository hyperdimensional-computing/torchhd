import pytest
import torch

from .. import functional


class TestIdentity_hv:
    def test_shape(self):
        hv = functional.identity_hv(13, 2556)

        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_value(self):
        hv = functional.identity_hv(4, 85)
        assert (hv == 1).min().item()

    def test_out(self):
        buffer = torch.empty(3, 52)
        hv = functional.identity_hv(3, 52, out=buffer)

        assert buffer.data_ptr() == hv.data_ptr()
        assert hv.dim() == 2
        assert hv.size(0) == 3
        assert hv.size(1) == 52

    def test_dtype(self):
        hv = functional.identity_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.identity_hv(3, 52, dtype=torch.bool)
        assert hv.dtype == torch.bool

        hv = functional.identity_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

    def test_requires_grad(self):
        hv = functional.identity_hv(3, 52, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.identity_hv(3, 52, requires_grad=True)
        assert hv.requires_grad == True

    def test_integration(self):
        buffer = torch.empty(6, 10000, dtype=torch.float16)
        hv = functional.identity_hv(
            6, 10000, out=buffer, dtype=torch.float16, requires_grad=True
        )

        assert buffer.data_ptr() == hv.data_ptr()
        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float16

        with pytest.raises(RuntimeError):
            buffer = torch.empty(6, 10000, dtype=torch.float)
            hv = functional.identity_hv(6, 10000, out=buffer, dtype=torch.float16)
