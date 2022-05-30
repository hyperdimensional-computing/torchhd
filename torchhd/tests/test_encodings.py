from re import A, L
import pytest
import torch

from torchhd import functional

from .utils import between, torch_dtypes, torch_float_dtypes, torch_complex_dtypes


class TestMultiset:
    def test_value(self):
        hv = torch.zeros(4, 1000)
        res = functional.multiset(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1]])
        res = functional.multiset(hv)
        assert torch.all(res == torch.tensor([-1, -1, -1, 3])).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.multiset(hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.multiset(hv)

            return

        res = functional.multiset(hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.multibind(hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.multibind(hv)

            return

        if dtype in {torch.float16, torch.bfloat16}:
            # torch.product is not implemented on CPU for these dtypes
            with pytest.raises(RuntimeError):
                functional.multibind(hv)

            return         

        res = functional.multibind(hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.cross_product(hv, hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.cross_product(hv, hv)

            return

        res = functional.cross_product(hv, hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.ngrams(hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.ngrams(hv)

            return

        res = functional.ngrams(hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.hash_table(hv, hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.hash_table(hv, hv)

            return

        res = functional.hash_table(hv, hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.bundle_sequence(hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.bundle_sequence(hv)

            return

        res = functional.bundle_sequence(hv)
        assert res.dtype == dtype

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

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype)

        if dtype in torch_complex_dtypes:
            with pytest.raises(NotImplementedError):
                functional.bind_sequence(hv)

            return

        if dtype == torch.uint8:
            with pytest.raises(ValueError):
                functional.bind_sequence(hv)

            return

        if dtype in {torch.float16, torch.bfloat16}:
            # torch.product is not implemented on CPU for these dtypes
            with pytest.raises(RuntimeError):
                functional.multibind(hv)

            return  

        res = functional.bind_sequence(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.bind_sequence(hv)
        assert res.device == device
