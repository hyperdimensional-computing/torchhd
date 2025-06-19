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
import math

import torchhd
from torchhd import functional
from torchhd import embeddings
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.vtb import VTBTensor
from torchhd.tensors.fhrr import type_conversion as fhrr_type_conversion


from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestEmpty:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])

        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa == "MAP" or vsa == "HRR":
            assert emb(idx).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert (
                emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 10000
        embedding = 4
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Empty(embedding, dimensions, vsa=vsa)

        if vsa == "BSC":
            assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1
        elif vsa in {"MAP", "HRR", "VTB"}:
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 0.0 + 0.0j).item()

        emb.reset_parameters()

        if vsa == "BSC":
            assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1
        elif vsa in {"MAP", "HRR", "VTB"}:
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 0.0 + 0.0j).item()


class TestIdentity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        idx = torch.LongTensor([0, 1, 3])

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa in {"MAP", "HRR", "VTB"}:
            assert emb(idx).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert emb(idx).dtype in {torch.complex64, torch.complex32}

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 9
        embedding = 4

        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        if vsa == "BSC":
            assert torch.all(emb.weight == False).item()
        elif vsa == "MAP":
            assert torch.all(emb.weight == 1.0).item()
        elif vsa == "HRR":
            ten = HRRTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == "VTB":
            ten = VTBTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ) * 3 ** (-0.5)
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 1.0 + 0.0j).item()

        emb.reset_parameters()
        if vsa == "BSC":
            assert torch.all(emb.weight == False).item()
        elif vsa == "MAP":
            assert torch.all(emb.weight == 1.0).item()
        elif vsa == "HRR":
            ten = HRRTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == "VTB":
            ten = VTBTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ) * 3 ** (-0.5)
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 1.0 + 0.0j).item()


class TestRandom:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Random(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Random(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Random(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa in {"MAP", "HRR", "VTB"}:
            assert emb(idx).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert emb(idx).dtype in {torch.complex64, torch.complex32}

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 10000
        embedding = 4
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Random(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1

        emb.reset_parameters()
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1


class TestLevel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Level(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Level(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Level(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa in {"MAP", "HRR", "VTB"}:
            assert emb(idx).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert emb(idx).dtype in {torch.complex64, torch.complex32}

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 99856
        embedding = 4
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Level(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.67
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.65
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) < 0.01
        )

        emb.reset_parameters()
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.67
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.65
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) < 0.01
        )


class TestCircular:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return
        dimensions = 1024
        embedding = 10
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return
        dimensions = 4
        embedding = 6
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        angle = torch.tensor([0.0, 3.141, 6.282, 9.423])

        if vsa == "BSC":
            assert emb(angle).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(angle).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert (
                emb(angle).dtype == torch.complex64
                or emb(angle).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "HRR" or vsa == "VTB":
            return
        dimensions = 99856
        embedding = 4
        if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa, block_size=1024)
        else:
            emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.51
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.49
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) < 0.51
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) > 0.49
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.01
        )

        emb.reset_parameters()
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.51
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.49
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) < 0.51
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) > 0.49
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.01
        )


class TestThermometer:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        if vsa in {"HRR", "BSBC", "MCR", "CGR"}:
            return
        dimensions = 1024
        embedding = 10

        if vsa not in {"BSC", "MAP", "FHRR"}:
            with pytest.raises(ValueError):
                emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)

            return

        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 1024
        embedding = 10
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 6
        embedding = 4
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        angle = torch.rand(4)
        if vsa == "BSC":
            assert emb(angle).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(angle).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert (
                emb(angle).dtype == torch.complex64
                or emb(angle).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 99856
        embedding = 4
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) > 0.99
        )

        emb.reset_parameters()
        assert (
            abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) < 0.34
            and abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[2])) > 0.32
            or abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[3])) > 0.99
        )


class TestProjection:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_in_features(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1020
        out_features = 16
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert emb.in_features == in_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_out_features(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1020
        out_features = 16
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert emb.out_features == out_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1000
        out_features = 16
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        x = torch.randn(1, in_features)
        if vsa == "MAP" or vsa == "HRR":
            assert emb(x).dtype == torch.get_default_dtype()
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 100000
        out_features = 100
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1

        emb.reset_parameters()
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1


class TestSinusoid:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_in_features(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1000
        out_features = 16
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert emb.in_features == in_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_out_features(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1000
        out_features = 16
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert emb.out_features == out_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 1000
        out_features = 16
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        x = torch.randn(1, in_features)
        if vsa == "MAP" or vsa == "HRR":
            assert emb(x).dtype == torch.get_default_dtype()
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa in {"BSC", "FHRR", "BSBC", "MCR", "CGR"}:
            return
        in_features = 100000
        out_features = 16

        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1

        emb.reset_parameters()
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-1


class TestDensity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1024
        embedding = 16

        if vsa not in {"BSC", "MAP", "FHRR"}:
            if vsa == "BSBC" or vsa == "MCR" or vsa == "CGR":
                with pytest.raises(ValueError):
                    emb = embeddings.Density(
                        embedding, dimensions, vsa=vsa, block_size=1024
                    )
            else:
                with pytest.raises(ValueError):
                    emb = embeddings.Density(embedding, dimensions, vsa=vsa)

            return

        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        assert emb.density_encoding.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embedings(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 1024
        embedding = 16
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        assert emb.density_encoding.num_embeddings == dimensions + 1

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 1024
        embedding = 16
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        x = torch.randn(1, embedding)

        if vsa == "BSC":
            assert emb(x).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(x).dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert emb(x).dtype == torch.complex64 or emb(x).dtype == torch.complex32
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa not in {"BSC", "MAP", "FHRR"}:
            return

        dimensions = 1024
        embedding = 16

        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        assert (
            abs(
                torchhd.cosine_similarity(
                    emb.density_encoding.weight[0], emb.density_encoding.weight[1]
                )
            )
            > 0.99
        )

        emb.reset_parameters()
        assert (
            abs(
                torchhd.cosine_similarity(
                    emb.density_encoding.weight[0], emb.density_encoding.weight[1]
                )
            )
            > 0.99
        )


class TestFractionalPower:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_default_dtype(self, vsa):
        dimensions = 1000
        embedding = 16

        if vsa not in {"HRR", "FHRR"}:
            with pytest.raises(ValueError):
                embeddings.FractionalPower(embedding, dimensions, vsa=vsa)

            return

        emb = embeddings.FractionalPower(embedding, dimensions, vsa=vsa)
        x = torch.randn(2, embedding)
        y = emb(x)
        assert y.shape == (2, dimensions)

        if vsa == "HRR":
            assert y.dtype == torch.get_default_dtype()
        elif vsa == "FHRR":
            assert fhrr_type_conversion[y.dtype] == torch.get_default_dtype()
        else:
            return

    @pytest.mark.parametrize("dtype", torch_dtypes)
    def test_dtype(self, dtype):
        dimensions = 1456
        embedding = 2

        if dtype not in {torch.float32, torch.float64}:
            with pytest.raises(ValueError):
                embeddings.FractionalPower(
                    embedding, dimensions, vsa="HRR", dtype=dtype
                )
        else:
            emb = embeddings.FractionalPower(
                embedding, dimensions, vsa="HRR", dtype=dtype
            )

            x = torch.randn(13, embedding, dtype=dtype)
            y = emb(x)
            assert y.shape == (13, dimensions)
            assert y.dtype == dtype

        if dtype not in {torch.complex64, torch.complex128}:
            with pytest.raises(ValueError):
                embeddings.FractionalPower(
                    embedding, dimensions, vsa="FHRR", dtype=dtype
                )
        else:
            emb = embeddings.FractionalPower(
                embedding, dimensions, vsa="FHRR", dtype=dtype
            )

            x = torch.randn(13, embedding, dtype=fhrr_type_conversion[dtype])
            y = emb(x)
            assert y.shape == (13, dimensions)
            assert y.dtype == dtype

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        emb = embeddings.FractionalPower(35, 1000, "gaussian", device=device)

        x = torchhd.random(5, 35, device=device)
        y = emb(x)
        assert y.shape == (5, 1000)
        assert y.device.type == device.type

    def test_custom_dist_iid(self):
        kernel_shape = torch.distributions.Normal(0, 1)
        band = 3.0

        emb = embeddings.FractionalPower(3, 1000, kernel_shape, band)
        x = torch.randn(1, 3)
        y = emb(x)
        assert y.shape == (1, 1000)

    def test_custom_dist_2d(self):
        # Phase distribution for periodic Sinc kernel
        class HexDisc(torch.distributions.Categorical):
            def __init__(self):
                super().__init__(torch.ones(6))
                self.r = 1
                self.side = self.r * math.sqrt(3) / 2
                self.phases = torch.tensor(
                    [
                        [-self.r, 0.0],
                        [-self.r / 2, self.side],
                        [self.r / 2, self.side],
                        [self.r, 0.0],
                        [self.r / 2, -self.side],
                        [-self.r / 2, -self.side],
                    ]
                )

            def sample(self, sample_shape=torch.Size()):
                return self.phases[super().sample(sample_shape), :]

        kernel_shape = HexDisc()
        band = 3.0

        emb = embeddings.FractionalPower(2, 1000, kernel_shape, band)
        x = torch.randn(5, 2)
        y = emb(x)
        assert y.shape == (5, 1000)
