from re import A
import pytest
import torch

from torchhd import functional

from .utils import between


class TestMultiset:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.multiset(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1]])
        res = functional.multiset(hv)
        assert torch.all(res == torch.tensor([-1, -1, -1, 3])).item()

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.multiset(hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.multiset(hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.multiset(hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.multiset(hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.multiset(hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.multiset(hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.multiset(hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.multiset(hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.multiset(hv)
        assert res.device == device


class TestMultibind:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.multibind(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]])
        res = functional.multibind(hv)
        assert torch.all(res == torch.tensor([1, 1, -1, 1])).item()
        assert res.dtype == hv.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.multibind(hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.multibind(hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.multibind(hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.multibind(hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.multibind(hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.multibind(hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.multibind(hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.multibind(hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.multibind(hv)
        assert res.device == device


class TestCrossProduct:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.cross_product(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]])
        b = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]])
        res = functional.cross_product(a, b)
        assert torch.all(res == torch.tensor([0, 2, 0, 6])).item()
        assert res.dtype == a.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.cross_product(hv, hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.cross_product(hv, hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.cross_product(hv, hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.cross_product(hv, hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.cross_product(hv, hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.cross_product(hv, hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.cross_product(hv, hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.cross_product(hv, hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.cross_product(hv, hv)
        assert res.device == device


class TestNgrams:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.ngrams(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        )
        res = functional.ngrams(hv)
        assert torch.all(res == torch.tensor([0, -2, -2, 0])).item()
        assert res.dtype == hv.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.ngrams(hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.ngrams(hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.ngrams(hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.ngrams(hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.ngrams(hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.ngrams(hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.ngrams(hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.ngrams(hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.ngrams(hv)
        assert res.device == device


class TestHashTable:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.hash_table(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]])
        b = torch.tensor([[-1, 1, 1, 1], [-1, 1, 1, -1]])
        res = functional.hash_table(a, b)
        assert torch.all(res == torch.tensor([0, -2, 0, 0])).item()
        assert res.dtype == a.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.hash_table(hv, hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.hash_table(hv, hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.hash_table(hv, hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.hash_table(hv, hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.hash_table(hv, hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.hash_table(hv, hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.hash_table(hv, hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.hash_table(hv, hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.hash_table(hv, hv)
        assert res.device == device


class TestBundleSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.bundle_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        )
        res = functional.bundle_sequence(hv)
        assert torch.all(res == torch.tensor([0, 0, 2, 0])).item()
        assert res.dtype == hv.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.bundle_sequence(hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.bundle_sequence(hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.bundle_sequence(hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.bundle_sequence(hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.bundle_sequence(hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.bundle_sequence(hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.bundle_sequence(hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.bundle_sequence(hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.bundle_sequence(hv)
        assert res.device == device


class TestBindSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.bind_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        )
        res = functional.bind_sequence(hv)
        assert torch.all(res == torch.tensor([1, 1, -1, 1])).item()
        assert res.dtype == hv.dtype

    def test_dtype(self):
        hv = functional.random_hv(11, 10000)
        res = functional.bind_sequence(hv)
        assert res.dtype == torch.get_default_dtype()

        hv = functional.random_hv(11, 10000, dtype=torch.float)
        res = functional.bind_sequence(hv)
        assert res.dtype == torch.float

        hv = functional.random_hv(11, 10000, dtype=torch.int)
        res = functional.bind_sequence(hv)
        assert res.dtype == torch.int

        hv = functional.random_hv(11, 10000, dtype=torch.long)
        res = functional.bind_sequence(hv)
        assert res.dtype == torch.long

        hv = functional.random_hv(11, 10000, dtype=torch.float64)
        res = functional.bind_sequence(hv)
        assert res.dtype == torch.float64

        hv = torch.zeros(4, 1000, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            res = functional.bind_sequence(hv)

        hv = torch.zeros(4, 1000, dtype=torch.complex128)
        with pytest.raises(NotImplementedError):
            res = functional.bind_sequence(hv)

        hv = torch.zeros(4, 1000, dtype=torch.uint8)
        with pytest.raises(ValueError):
            res = functional.bind_sequence(hv)

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.bind_sequence(hv)
        assert res.device == device
