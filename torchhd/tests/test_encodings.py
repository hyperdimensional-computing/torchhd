import pytest
import torch

from torchhd import functional
from torchhd.bsc import BSC
from torchhd.map import MAP

from .utils import (
    torch_dtypes,
    vsa_models,
    supported_dtype,
)


class TestMultiset:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, model, dtype):
        if not supported_dtype(dtype, model):
            return

        if model == BSC:
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                ],
                dtype=dtype,
            ).as_subclass(BSC)
            res = functional.multiset(hv)
            assert torch.all(
                res
                == torch.tensor(
                    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                    dtype=dtype,
                )
            ).item()

        elif model == MAP:
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                    [1, 1, -1, -1, -1, 1, -1, 1, -1, 1],
                ],
                dtype=dtype,
            ).as_subclass(MAP)
            res = functional.multiset(hv)
            assert torch.all(
                res == torch.tensor([3, 3, -5, 1, 1, 1, 1, 1, -1, 1], dtype=dtype)
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.multiset(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.multiset(hv)
        assert res.device == device


class TestMultibind:
    @pytest.mark.parametrize("model", vsa_models)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, model, dtype):
        if not supported_dtype(dtype, model):
            return

        if model == BSC:
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                ],
                dtype=dtype,
            ).as_subclass(BSC)
            res = functional.multibind(hv)
            assert torch.all(
                res
                == torch.tensor(
                    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                    dtype=dtype,
                )
            ).item()

        elif model == MAP:
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                    [1, 1, -1, -1, -1, 1, -1, 1, -1, 1],
                ],
                dtype=dtype,
            ).as_subclass(MAP)
            res = functional.multibind(hv)
            assert torch.all(
                res == torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, -1, 1], dtype=dtype)
            ).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

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
        hv = torch.zeros(4, 1000).as_subclass(MAP)
        res = functional.cross_product(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]]).as_subclass(MAP)
        b = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]]).as_subclass(MAP)
        res = functional.cross_product(a, b)
        assert torch.all(res == torch.tensor([0, 2, 0, 6])).item()
        assert res.dtype == a.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.cross_product(hv, hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.cross_product(hv, hv)
        assert res.device == device


class TestNgrams:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAP)
        res = functional.ngrams(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAP)
        res = functional.ngrams(hv)
        assert torch.all(res == torch.tensor([0, -2, -2, 0])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.ngrams(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.ngrams(hv)
        assert res.device == device


class TestHashTable:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAP)
        res = functional.hash_table(hv, hv)
        assert torch.all(res == 0).item()

        a = torch.tensor([[1, -1, -1, 1], [-1, -1, 1, 1]]).as_subclass(MAP)
        b = torch.tensor([[-1, 1, 1, 1], [-1, 1, 1, -1]]).as_subclass(MAP)
        res = functional.hash_table(a, b)
        assert torch.all(res == torch.tensor([0, -2, 0, 0])).item()
        assert res.dtype == a.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.hash_table(hv, hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.hash_table(hv, hv)
        assert res.device == device


class TestBundleSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAP)
        res = functional.bundle_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAP)
        res = functional.bundle_sequence(hv)
        assert torch.all(res == torch.tensor([0, 0, 2, 0])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.bundle_sequence(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(11, 10000, device=device)
        res = functional.bundle_sequence(hv)
        assert res.device == device


class TestBindSequence:
    def test_value(self):
        hv = torch.zeros(4, 1000).as_subclass(MAP)
        res = functional.bind_sequence(hv)
        assert torch.all(res == 0).item()

        hv = torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, -1]]
        ).as_subclass(MAP)
        res = functional.bind_sequence(hv)
        assert torch.all(res == torch.tensor([1, 1, -1, 1])).item()
        assert res.dtype == hv.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(23, 1000, dtype=dtype).as_subclass(MAP)

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


class TestGraph:
    def test_value(self):
        hv = torch.zeros(2, 4, 1000).as_subclass(MAP)
        res = functional.graph(hv)
        assert torch.all(res == 0).item()

        g = torch.tensor(
            [
                [[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, 1]],
                [[-1, -1, 1, 1], [-1, 1, 1, 1], [1, -1, -1, 1]],
            ]
        ).as_subclass(MAP)
        res = functional.graph(g)
        assert torch.all(res == torch.tensor([-1, -1, -1, 3])).item()
        assert res.dtype == g.dtype

        res = functional.graph(g, directed=True)
        assert torch.all(res == torch.tensor([-1, 3, 1, 1])).item()
        assert res.dtype == g.dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        hv = torch.zeros(5, 2, 23, 1000, dtype=dtype).as_subclass(MAP)

        res = functional.graph(hv)
        assert res.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = torch.zeros(5, 2, 23, 1000, device=device).as_subclass(MAP)
        res = functional.graph(hv)
        assert res.device == device
