import pytest
import torch

from torchhd import functional

from .utils import (
    torch_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)

seed = 2147483644


class TestDotSimilarity:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        hv1 = functional.random_hv(6, 100, generator=generator, dtype=dtype).view(
            2, 3, 100
        )
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if dtype == torch.bool:
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            )

            res = functional.dot_similarity(hv, hv)
            exp = torch.tensor([[10, 4], [4, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        elif dtype in torch_complex_dtypes:
            hv = torch.tensor(
                [
                    [
                        -0.2510 + 0.9680j,
                        0.0321 + 0.9995j,
                        -0.6063 - 0.7953j,
                        -0.4006 - 0.9162j,
                        0.4987 - 0.8667j,
                        -0.3252 - 0.9456j,
                        -0.2784 + 0.9605j,
                        -0.8563 + 0.5165j,
                        0.9061 + 0.4231j,
                        -0.3801 - 0.9250j,
                    ],
                    [
                        -0.9610 + 0.2766j,
                        0.9879 - 0.1551j,
                        -0.4111 - 0.9116j,
                        -0.8185 + 0.5744j,
                        -0.8123 + 0.5833j,
                        0.2966 + 0.9550j,
                        -0.9958 - 0.0915j,
                        0.8630 - 0.5052j,
                        -0.1480 - 0.9890j,
                        0.5285 - 0.8489j,
                    ],
                ],
                dtype=dtype,
            )

            res = functional.dot_similarity(hv, hv)
            out_dtype = torch.float if dtype == torch.complex64 else torch.double
            exp = torch.tensor([[10.0, -1.5274], [-1.5274, 10.0]], dtype=out_dtype)
            assert torch.allclose(res, exp)

        else:
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            )

            res = functional.dot_similarity(hv, hv)
            exp = torch.tensor([[10, -4], [-4, 10]], dtype=dtype)
            assert torch.all(res == exp).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(3, 100, generator=generator, dtype=dtype)

        similarity = functional.dot_similarity(hv, hv)

        if dtype == torch.bool:
            assert similarity.dtype == torch.long
        elif dtype == torch.complex64:
            assert similarity.dtype == torch.float
        elif dtype == torch.complex128:
            assert similarity.dtype == torch.double
        else:
            assert similarity.dtype == dtype

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(
            3, 100, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.dot_similarity(hv, hv)

        assert similarity.device == device


class TestCosSimilarity:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        hv1 = functional.random_hv(6, 100, generator=generator, dtype=dtype).view(
            2, 3, 100
        )
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if dtype == torch.bool:
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            )

            res = functional.cosine_similarity(hv, hv)
            exp = torch.tensor([[1, 0.4], [0.4, 1]], dtype=torch.float)
            assert torch.allclose(res, exp)

        elif dtype in torch_complex_dtypes:
            hv = torch.tensor(
                [
                    [
                        -0.2510 + 0.9680j,
                        0.0321 + 0.9995j,
                        -0.6063 - 0.7953j,
                        -0.4006 - 0.9162j,
                        0.4987 - 0.8667j,
                        -0.3252 - 0.9456j,
                        -0.2784 + 0.9605j,
                        -0.8563 + 0.5165j,
                        0.9061 + 0.4231j,
                        -0.3801 - 0.9250j,
                    ],
                    [
                        -0.9610 + 0.2766j,
                        0.9879 - 0.1551j,
                        -0.4111 - 0.9116j,
                        -0.8185 + 0.5744j,
                        -0.8123 + 0.5833j,
                        0.2966 + 0.9550j,
                        -0.9958 - 0.0915j,
                        0.8630 - 0.5052j,
                        -0.1480 - 0.9890j,
                        0.5285 - 0.8489j,
                    ],
                ],
                dtype=dtype,
            )

            res = functional.cosine_similarity(hv, hv)
            exp = torch.tensor([[1.0, -0.15274], [-0.15274, 1.0]], dtype=torch.float)
            assert torch.allclose(res, exp)

        else:
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            )

            res = functional.cosine_similarity(hv, hv)
            exp = torch.tensor([[1, -0.4], [-0.4, 1]], dtype=torch.float)
            assert torch.allclose(res, exp)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(3, 100, generator=generator, dtype=dtype)

        similarity = functional.cosine_similarity(hv, hv)

        assert similarity.dtype == torch.float

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, dtype):
        if not supported_dtype(dtype) or dtype == torch.half:
            return

        generator = torch.Generator()
        generator.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(
            3, 100, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.cosine_similarity(hv, hv)

        assert similarity.device == device


class TestHammingSimilarity:
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        hv = functional.random_hv(2, 100, generator=generator, dtype=dtype)
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        hv1 = functional.random_hv(6, 100, generator=generator, dtype=dtype).view(
            2, 3, 100
        )
        hv2 = functional.random_hv(4, 100, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if dtype == torch.bool:
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            )

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 7], [7, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        elif dtype in torch_complex_dtypes:
            hv = torch.tensor(
                [
                    [
                        -0.2510 + 0.9680j,
                        0.0321 + 0.9995j,
                        -0.6063 - 0.7953j,
                        -0.4006 - 0.9162j,
                        0.4987 - 0.8667j,
                        -0.3252 - 0.9456j,
                        -0.2784 + 0.9605j,
                        -0.8563 + 0.5165j,
                        0.9061 + 0.4231j,
                        -0.3801 - 0.9250j,
                    ],
                    [
                        -0.9610 + 0.2766j,
                        0.9879 - 0.1551j,
                        -0.4111 - 0.9116j,
                        -0.8185 + 0.5744j,
                        -0.8123 + 0.5833j,
                        0.2966 + 0.9550j,
                        -0.9958 - 0.0915j,
                        0.8630 - 0.5052j,
                        -0.1480 - 0.9890j,
                        0.5285 - 0.8489j,
                    ],
                ],
                dtype=dtype,
            )

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 0], [0, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        else:
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            )

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 3], [3, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random_hv(3, 100, generator=generator, dtype=dtype)

        similarity = functional.hamming_similarity(hv, hv)

        assert similarity.dtype == torch.long

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, dtype):
        if not supported_dtype(dtype):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(
            3, 100, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.hamming_similarity(hv, hv)

        assert similarity.device == device
