import pytest
import torch
import torchhd
from torchhd import functional

from .utils import (
    torch_dtypes,
    vsa_models,
    supported_dtype,
)


class TestBind:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        hv = functional.empty_hv(2, 10, model, dtype=dtype)
        res = functional.bind(hv[0], hv[1])
        if model == torchhd.BSC:
            assert torch.all(
                res
                == torch.logical_xor(hv[0], hv[1])
            ).item()
        elif model == torchhd.FHRR or model == torchhd.MAP:
            assert torch.all(
                res == torch.mul(hv[0], hv[1])
            ).item()
        elif model == torchhd.HRR:
            from torch.fft import fft, ifft
            assert torch.all(
                res == ifft(torch.mul(fft(hv[0]), fft(hv[1])))
            ).item()
        assert dtype == res.dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert res.device == device


class TestBundle:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, model, dtype):
        if not supported_dtype(dtype, model):
            return

        hv = functional.random_hv(2, 10, model, dtype=dtype)
        res = functional.bundle(hv[0], hv[1])

        if model == torchhd.BSC:
            hv[0] = torch.tensor([False, False, True, False, False, True, True, True, False, False])
            hv[1] = torch.tensor([True, False, True, False, False, True, False, False, True, False])

            res = functional.bundle(hv[0], hv[1])
            for i in range(10):
                assert (hv[0][i].item() == hv[1][i].item() and hv[1][i].item() == True and res[i].item()) or (hv[0][i].item() == hv[1][i].item() and hv[1][i].item() == False and not res[i].item()) or (hv[0][i].item() != hv[1][i].item())

        if model == torchhd.MAP:
            hv[0] = torch.tensor([1, 1, -1, -1, 1, 1, 1, 1, -1, -1])
            hv[1] = torch.tensor([1, 1, -1, -1, -1, -1, -1, -1, 1, -1])

            res = functional.bundle(hv[0], hv[1])
            assert torch.all(
                res == torch.tensor([2, 2, -2, -2, 0, 0, 0, 0, 0, -2], dtype=dtype)
            ).item()
        if model == torchhd.FHRR:
            assert torch.all(
                res == hv[0].add(hv[1])
            ).item()
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv <= 2) & (hv >= -2)).item(), "values are between -2 and +2"
        assert res.device == device


class TestPermute:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        hv = functional.random_hv(2, 100, model, dtype=dtype)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        if model == torchhd.BSC:
            assert torch.all((hv == 0) | (hv == 1)).item(), "values are either -1 or +1"
            assert torch.sum(res == hv[0]) != res.size(
                0
            ), "all element must not be the same"

            one_shift = functional.permute(hv[0])
            two_shift = functional.permute(hv[0], shifts=2)
            assert torch.sum(one_shift == two_shift) != res.size(
                0
            ), "all element must not be the same"

            hv = functional.random_hv(1, 10000, model, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        if model == torchhd.MAP:
            assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
            assert torch.sum(res == hv[0]) != res.size(
                0
            ), "all element must not be the same"

            one_shift = functional.permute(hv[0])
            two_shift = functional.permute(hv[0], shifts=2)
            assert torch.sum(one_shift == two_shift) != res.size(
                0
            ), "all element must not be the same"

            hv = functional.random_hv(1, 10000, model, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        if model == torchhd.HRR or model == torchhd.FHRR:
            hv = functional.random_hv(1, 10000, model, dtype=dtype)
            a = functional.permute(hv, shifts=5)
            b = functional.permute(a, shifts=-5)
            assert torch.all(hv == b).item(), "can undo shifts"
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert res.device == device


class TestCleanup:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(5, 100, model, dtype=dtype)
        noise = functional.random_hv(1, 100, model, dtype=dtype)
        res = functional.cleanup(functional.bundle(hv[0], noise), hv)
        assert torch.all(hv[0] == res).item()

    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_threshold(self, dtype, model):
        if not supported_dtype(dtype, model):
            return

        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(5, 100, model, dtype=dtype)
        noise = functional.random_hv(1, 100, model, dtype=dtype)
        res = functional.cleanup(functional.bundle(hv[0], noise), hv, threshold=0.3)
        assert torch.all(hv[0] == res).item()

    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, model, dtype):
        if not supported_dtype(dtype, model):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(5, 100, model, dtype=dtype)
        res = functional.cleanup(hv[0], hv)
        assert res.device == device


class TestRandsel:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, model, dtype):
        if not supported_dtype(dtype, model):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        a, b = functional.random_hv(2, 1000, model, dtype=dtype)
        res = functional.randsel(a, b, p=0)
        assert torch.all(a == res)

        a, b = functional.random_hv(2, 1000, model, dtype=dtype)
        res = functional.randsel(a, b, p=1)
        assert torch.all(b == res)

        a, b = functional.random_hv(2, 1000, model, dtype=dtype)
        res = functional.randsel(a, b)
        assert torch.all((b == res) | (a == res))
        assert res.dtype == dtype

    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, model, dtype):
        if not supported_dtype(dtype, model):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        a, b = functional.random_hv(2, 100, model, dtype=dtype)
        res = functional.randsel(a, b)

        assert res.dtype == a.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert res.device == device


class TestMultiRandsel:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, model, dtype):
        if not supported_dtype(dtype, model):
            return
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        x = functional.random_hv(4, 1000, model, dtype=dtype)

        res = functional.multirandsel(
            x, p=torch.tensor([0.0, 0.0, 1.0, 0.0]), generator=generator
        )
        assert torch.all(x[2] == res)

        x = functional.random_hv(4, 1000, model, dtype=dtype)
        res = functional.multirandsel(
            x, p=torch.tensor([0.5, 0.0, 0.5, 0.0]), generator=generator
        )
        assert torch.all((x[0] == res) | (x[2] == res))

        x = functional.random_hv(4, 1000, model, dtype=dtype)
        res = functional.multirandsel(x, generator=generator)
        assert torch.all((x[0] == res) | (x[1] == res) | (x[2] == res) | (x[3] == res))
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = functional.random_hv(4, 100, device=device)
        res = functional.multirandsel(x)

        assert res.dtype == x.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert res.device == device
