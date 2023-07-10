#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import pytest
import torch

from torchhd import functional
from torchhd import BSCTensor
from torchhd import FHRRTensor
from torchhd import MAPTensor
from torchhd import HRRTensor

from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)

seed = 2147483644


class TestDotSimilarity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        if vsa == "SBC":
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype, block_size=1024).view(2, 3, 100)
        else:
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype).view(2, 3, 100)
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.dot_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSC":
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            ).as_subclass(BSCTensor)

            res = functional.dot_similarity(hv, hv)
            exp = torch.tensor([[10, 4], [4, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        elif vsa == "FHRR":
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
            ).as_subclass(FHRRTensor)

            res = functional.dot_similarity(hv, hv)
            out_dtype = torch.float if dtype == torch.complex64 else torch.double
            exp = torch.tensor([[10.0, -1.5274], [-1.5274, 10.0]], dtype=out_dtype)
            assert torch.allclose(res, exp)

        elif vsa == "MAP":
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            ).as_subclass(MAPTensor)

            res = functional.dot_similarity(hv, hv)
            exp = torch.tensor([[10, -4], [-4, 10]], dtype=dtype)
            assert torch.all(res == exp).item()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random(3, 100, vsa, generator=generator, dtype=dtype)

        similarity = functional.dot_similarity(hv, hv)

        if vsa == "FHRR":
            if dtype == torch.complex64:
                assert similarity.dtype == torch.float
            elif dtype == torch.complex128:
                assert similarity.dtype == torch.double
        elif vsa == "HRR":
            assert similarity.dtype == dtype
        else:
            assert similarity.dtype == torch.get_default_dtype()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device)
        generator.manual_seed(seed)

        if vsa == "SBC":
            hv = functional.random(
            3, 100, vsa, generator=generator, dtype=dtype, device=device,block_size=1024)
        else:
            hv = functional.random(
            3, 100, vsa, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.dot_similarity(hv, hv)

        assert similarity.device.type == device.type


class TestCosSimilarity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        if vsa == "SBC":
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype, block_size=1024).view(2, 3, 100)
        else:
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype).view(2, 3, 100)
            
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.cosine_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSC":
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            ).as_subclass(BSCTensor)

            res = functional.cosine_similarity(hv, hv)
            exp = torch.tensor([[1, 0.4], [0.4, 1]], dtype=torch.float)
            assert torch.allclose(res, exp)

        elif vsa == "FHRR":
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
            ).as_subclass(FHRRTensor)

            res = functional.cosine_similarity(hv, hv)
            result_dtype = torch.float if dtype == torch.complex64 else torch.double
            exp = torch.tensor([[1.0, -0.15274], [-0.15274, 1.0]], dtype=result_dtype)
            assert torch.allclose(res, exp)

        elif vsa == "MAP":
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            ).as_subclass(MAPTensor)

            res = functional.cosine_similarity(hv, hv)
            exp = torch.tensor([[1, -0.4], [-0.4, 1]], dtype=torch.float)
            assert torch.allclose(res, exp)

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random(3, 100, vsa, generator=generator, dtype=dtype)

        similarity = functional.cosine_similarity(hv, hv)

        if vsa == "FHRR":
            if dtype == torch.complex64:
                assert similarity.dtype == torch.float
            elif dtype == torch.complex128:
                assert similarity.dtype == torch.double
        elif vsa == "HRR":
            assert similarity.dtype == dtype
        else:
            assert similarity.dtype == torch.get_default_dtype()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device)
        generator.manual_seed(seed)

        hv = functional.random(
            3, 100, vsa, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.cosine_similarity(hv, hv)

        assert similarity.device.type == device.type


class TestHammingSimilarity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_shape(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv[0], hv[1])
        assert similarity.shape == ()

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv[0], hv)
        assert similarity.shape == (2,)

        if vsa == "SBC":
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv = functional.random(2, 100, vsa, generator=generator, dtype=dtype)
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv, hv2)
        assert similarity.shape == (2, 4)

        if vsa == "SBC":
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype, block_size=1024).view(2, 3, 100)
        else:
            hv1 = functional.random(6, 100, vsa, generator=generator, dtype=dtype).view(2, 3, 100)
            
        if vsa == "SBC":
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype, block_size=1024)
        else:
            hv2 = functional.random(4, 100, vsa, generator=generator, dtype=dtype)
        similarity = functional.hamming_similarity(hv1, hv2)
        assert similarity.shape == (2, 3, 4)

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_value(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        if vsa == "BSC":
            hv = torch.tensor(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                ],
                dtype=dtype,
            ).as_subclass(BSCTensor)

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 7], [7, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        elif vsa == "FHRR":
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
            ).as_subclass(FHRRTensor)

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 0], [0, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

        elif vsa == "MAP":
            hv = torch.tensor(
                [
                    [1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
                ],
                dtype=dtype,
            ).as_subclass(MAPTensor)

            res = functional.hamming_similarity(hv, hv)
            exp = torch.tensor([[10, 3], [3, 10]], dtype=torch.long)
            assert torch.all(res == exp).item()

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random(3, 100, vsa, generator=generator, dtype=dtype)

        similarity = functional.hamming_similarity(hv, hv)

        assert similarity.dtype == torch.long

    @pytest.mark.parametrize("vsa", vsa_tensors)
    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_device(self, vsa, dtype):
        if not supported_dtype(dtype, vsa):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device)
        generator.manual_seed(seed)

        hv = functional.random(
            3, 100, vsa, generator=generator, dtype=dtype, device=device
        )

        similarity = functional.hamming_similarity(hv, hv)

        assert similarity.device.type == device.type
