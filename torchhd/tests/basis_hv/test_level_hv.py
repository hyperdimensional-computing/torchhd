import pytest
import torch

import torchhd
from torchhd import functional

from ..utils import torch_dtypes, supported_dtype, VSATensors

seed = 2147483644


class Testlevel:
    @pytest.mark.parametrize("n", [1, 3, 55])
    @pytest.mark.parametrize("d", [84, 10])
    @pytest.mark.parametrize("model", VSATensors)
    def test_shape(self, n, d, model):
        hv = functional.level(n, d, model)

        assert hv.dim() == 2
        assert hv.size(0) == n
        assert hv.size(1) == d

    @pytest.mark.parametrize("model", VSATensors)
    def test_generator(self, model):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.level(20, 10000, model, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv2 = functional.level(20, 10000, model, generator=generator)
        assert torch.all(hv1 == hv2).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("model", VSATensors)
    def test_value(self, dtype, model):
        if not supported_dtype(dtype, model):
            with pytest.raises(ValueError):
                functional.level(3, 26, model, dtype=dtype)

            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.level(50, 26000, model, dtype=dtype, generator=generator)
        assert hv.requires_grad == False
        assert hv.dim() == 2
        assert hv.size(0) == 50
        assert hv.size(1) == 26000

        if model == torchhd.BSCTensor:
            assert torch.all((hv == False) | (hv == True)).item()

        elif model == torchhd.MAPTensor:
            assert torch.all((hv == -1) | (hv == 1)).item()

        elif model == torchhd.HRRTensor:
            std, mean = torch.std_mean(hv)
            assert torch.allclose(
                mean, torch.tensor(0.0, dtype=mean.dtype), atol=0.0001
            )

        elif model == torchhd.FHRRTensor:
            mag = hv.abs()
            assert torch.allclose(mag, torch.tensor(1.0, dtype=mag.dtype))

        sims = functional.cosine_similarity(hv[0], hv)
        sims_diff = sims[:-1] - sims[1:]
        assert torch.all(sims_diff > 0).item(), "similarity must be decreasing"

        hv = functional.level(5, 1000000, model, generator=generator, dtype=dtype)
        sims = functional.cosine_similarity(hv[0], hv)
        sims_diff = sims[:-1] - sims[1:]
        assert torch.allclose(
            sims_diff, torch.tensor(0.25, dtype=sims_diff.dtype), atol=0.005
        ), "similarity decreases linearly"

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.756, 1.0])
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_sparsity(self, sparsity, dtype):
        if not supported_dtype(dtype, torchhd.BSCTensor):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.level(
            1000,
            10000,
            torchhd.BSCTensor,
            generator=generator,
            dtype=dtype,
            sparsity=sparsity,
        )

        calc_sparsity = torch.sum(hv == False).div(10000 * 1000)
        assert torch.allclose(calc_sparsity, torch.tensor(sparsity), atol=0.005)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    @pytest.mark.parametrize("model", VSATensors)
    def test_device(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level(3, 52, model, device=device, dtype=dtype)
        assert hv.device == device

    def test_uses_default_dtype(self):
        hv = functional.level(3, 52, torchhd.BSCTensor)
        assert hv.dtype == torch.bool

        torch.set_default_dtype(torch.float32)
        hv = functional.level(3, 52, torchhd.MAPTensor)
        assert hv.dtype == torch.float32
        hv = functional.level(3, 52, torchhd.HRRTensor)
        assert hv.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        hv = functional.level(3, 52, torchhd.MAPTensor)
        assert hv.dtype == torch.float64
        hv = functional.level(3, 52, torchhd.HRRTensor)
        assert hv.dtype == torch.float64

        hv = functional.level(3, 52, torchhd.FHRRTensor)
        assert hv.dtype == torch.complex64

    def test_requires_grad(self):
        hv = functional.level(3, 52, torchhd.MAPTensor, requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.level(3, 52, torchhd.HRRTensor, requires_grad=True)
        assert hv.requires_grad == True

        hv = functional.level(3, 52, torchhd.FHRRTensor, requires_grad=True)
        assert hv.requires_grad == True
