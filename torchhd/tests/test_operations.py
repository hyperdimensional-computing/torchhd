import pytest
import torch

from torchhd import functional

from .utils import (
    between,
    torch_dtypes,
    torch_float_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)


class TestBind:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        if dtype == torch.bool:
            hv = torch.tensor(
                [
                    [False, False, True, False, False, True, True, True, False, False],
                    [True, False, True, False, False, True, False, False, True, False],
                ],
                dtype=dtype,
            )
            res = functional.bind(hv[0], hv[1])
            assert torch.all(
                res
                == torch.tensor(
                    [True, False, False, False, False, False, True, True, True, False]
                )
            ).item()
        else:
            hv = torch.tensor(
                [
                    [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1],
                    [1, 1, -1, 1, -1, 1, -1, -1, 1, -1],
                ],
                dtype=dtype,
            )
            res = functional.bind(hv[0], hv[1])
            assert torch.all(
                res == torch.tensor([-1, 1, -1, -1, 1, -1, 1, -1, -1, -1])
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.bind(hv[0], hv[1])

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.bind(hv[0], hv[1])

            return

        res = functional.bind(hv[0], hv[1])
        assert res.dtype == dtype

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
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        if dtype == torch.bool:
            hv = torch.tensor(
                [
                    [False, False, True, False, False, True, True, True, False, False],
                    [True, False, True, False, False, True, False, False, True, False],
                ],
                dtype=dtype,
            )
            res = functional.bundle(hv[0], hv[1])
            assert torch.all(
                res
                == torch.tensor(
                    [
                        False,
                        False,
                        True,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    dtype=dtype,
                )
            ).item()

            tie = torch.tensor(
                [[False, True, False, False, False, True, False, True, True, False]],
                dtype=dtype,
            )
            res = functional.bundle(hv[0], hv[1], tie=tie)
            assert torch.all(
                res
                == torch.tensor(
                    [False, False, True, False, False, True, False, True, True, False],
                    dtype=dtype,
                )
            ).item()
        else:
            hv = torch.tensor(
                [
                    [1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                    [1, 1, -1, -1, -1, -1, -1, -1, 1, -1],
                ],
                dtype=dtype,
            )
            res = functional.bundle(hv[0], hv[1])
            assert torch.all(
                res == torch.tensor([2, 2, -2, -2, 0, 0, 0, 0, 0, -2], dtype=dtype)
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.bundle(hv[0], hv[1])

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.bundle(hv[0], hv[1])

            return

        res = functional.bundle(hv[0], hv[1])
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
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"
        assert torch.sum(res == hv[0]) != res.size(
            0
        ), "all element must not be the same"

        one_shift = functional.permute(hv[0])
        two_shift = functional.permute(hv[0], shifts=2)
        assert torch.sum(one_shift == two_shift) != res.size(
            0
        ), "all element must not be the same"

        hv = functional.random_hv(1, 10000)
        a = functional.permute(hv, shifts=5)
        b = functional.permute(a, shifts=-5)
        assert torch.all(hv == b).item(), "can undo shifts"

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.permute(hv[0])

            return

        res = functional.permute(hv[0])
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
    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(5, 100, generator=generator)
        noise = functional.random_hv(1, 100, sparsity=0.95, generator=generator)
        res = functional.cleanup(functional.bind(hv[0], noise), hv)
        assert torch.all(hv[0] == res).item()

        hv = functional.random_hv(8, 100, generator=generator)
        noise = functional.random_hv(1, 100, sparsity=0.95, generator=generator)
        res = functional.cleanup(functional.bind(hv[3], noise), hv)
        assert torch.all(hv[3] == res).item()

    def test_threshold(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)

        hv = functional.random_hv(5, 100, generator=generator)
        res = functional.cleanup(hv[0], hv, threshold=0.5)
        assert torch.all(hv[0] == res).item()

        hv = functional.random_hv(5, 100, generator=generator)
        noise = functional.random_hv(1, 100, sparsity=0.95, generator=generator)
        res = functional.cleanup(hv[0], hv, threshold=1.0)
        assert torch.all(hv[0] == res).item()

        hv = functional.random_hv(5, 100, generator=generator)
        noise = functional.random_hv(1, 100, sparsity=0.95, generator=generator)
        with pytest.raises(KeyError):
            res = functional.cleanup(functional.bind(hv[0], noise), hv, threshold=1.0)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.cleanup(hv[0], hv, threshold=-1)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.cleanup(hv[0], hv, threshold=-1)

            return

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(5, 100, device=device)
        res = functional.cleanup(hv[0], hv)
        assert res.device == device
