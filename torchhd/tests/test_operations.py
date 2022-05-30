import pytest
import torch

from torchhd import functional


class TestBind:
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"

        a = torch.tensor([-1, -1, +1, +1])
        b = torch.tensor([-1, +1, -1, +1])
        res = functional.bind(a, b)
        expect = torch.tensor([+1, -1, -1, +1])
        assert torch.all(res == expect).item()

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"

        hv = torch.zeros(2, 10000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            functional.bind(hv[0], hv[1])

        hv = torch.zeros(2, 10000, dtype=torch.complex64)
        with pytest.raises(NotImplementedError):
            functional.bind(hv[0], hv[1])

        hv = torch.zeros(2, 10000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            functional.bind(hv[0], hv[1])

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
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv <= 2) & (hv >= -2)).item(), "values are between -2 and +2"

        a = torch.tensor([-1, -1, +1, +1])
        b = torch.tensor([-1, +1, -1, +1])
        res = functional.bundle(a, b)
        expect = torch.tensor([-2, 0, 0, +2])
        assert torch.all(res == expect).item()

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv <= 2) & (hv >= -2)).item(), "values are between -2 and +2"

        hv = torch.zeros(2, 10000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            functional.bundle(hv[0], hv[1])

        hv = torch.zeros(2, 10000, dtype=torch.complex64)
        with pytest.raises(NotImplementedError):
            functional.bundle(hv[0], hv[1])

        hv = torch.zeros(2, 10000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            functional.bundle(hv[0], hv[1])

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

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert torch.all((hv == -1) | (hv == 1)).item(), "values are either -1 or +1"

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

    def test_dtype(self):
        hv = functional.random_hv(5, 100)
        res = functional.cleanup(hv[0], hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(5, 100, dtype=torch.float)
        res = functional.cleanup(hv[0], hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(5, 100, dtype=torch.int)
        res = functional.cleanup(hv[0], hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(5, 100, dtype=torch.long)
        res = functional.cleanup(hv[0], hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(5, 100, dtype=torch.float64)
        res = functional.cleanup(hv[0], hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.cleanup(hv[0], hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.cleanup(hv[0], hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.cleanup(hv[0], hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(5, 100, device=device)
        res = functional.cleanup(hv[0], hv)
        assert res.device == device
