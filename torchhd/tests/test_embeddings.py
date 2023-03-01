import pytest
import torch

import torchhd
from torchhd import functional
from torchhd import embeddings
from torchhd.tensors.hrr import HRRTensor


from .utils import (
    torch_dtypes,
    vsa_tensors,
    supported_dtype,
)


class TestEmpty:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])

        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa == "MAP" or vsa == "HRR":
            assert emb(idx).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        if vsa == "BSC":
            assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-2
        elif vsa == "MAP":
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == "HRR":
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 0.0 + 0.0j).item()
        else:
            return


class TestIdentity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa == "MAP" or vsa == "HRR":
            assert emb(idx).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 6
        embedding = 4
        emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        if vsa == "BSC":
            assert torch.all(emb.weight == False).item()
        elif vsa == "MAP":
            assert torch.all(emb.weight == 1.0).item()
        elif vsa == "HRR":
            ten = HRRTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == "FHRR":
            assert torch.all(emb.weight == 1.0 + 0.0j).item()
        else:
            return


class TestRandom:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa == "MAP" or vsa == "HRR":
            assert emb(idx).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Random(embedding, dimensions, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-2


class TestLevel:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        dimensions = 1000
        embedding = 10
        emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        dimensions = 4
        embedding = 6
        emb = embeddings.Level(embedding, dimensions, vsa=vsa)
        idx = torch.LongTensor([0, 1, 3])
        if vsa == "BSC":
            assert emb(idx).dtype == torch.bool
        elif vsa == "MAP" or vsa == "HRR":
            assert emb(idx).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Level(embedding, dimensions, vsa=vsa)
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
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 4
        embedding = 6
        emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
        angle = torch.tensor([0.0, 3.141, 6.282, 9.423])

        if vsa == "BSC":
            assert emb(angle).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(angle).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(angle).dtype == torch.complex64
                or emb(angle).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Circular(embedding, dimensions, vsa=vsa)
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
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        assert emb.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embeddings(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        assert emb.num_embeddings == embedding

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 6
        embedding = 4
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
        angle = torch.rand(4)
        if vsa == "BSC":
            assert emb(angle).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(angle).dtype == torch.float
        elif vsa == "FHRR":
            assert (
                emb(angle).dtype == torch.complex64
                or emb(angle).dtype == torch.complex32
            )
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Thermometer(embedding, dimensions, vsa=vsa)
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
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert emb.in_features == in_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_out_features(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert emb.out_features == out_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        x = torch.randn(1, in_features)
        if vsa == "MAP" or vsa == "HRR":
            assert emb(x).dtype == torch.float
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 100000
        out_features = 100
        emb = embeddings.Projection(in_features, out_features, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-2


class TestSinusoid:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_in_features(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert emb.in_features == in_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_out_features(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert emb.out_features == out_features

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000
        out_features = 10
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        x = torch.randn(1, in_features)
        if vsa == "MAP" or vsa == "HRR":
            assert emb(x).dtype == torch.float
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "BSC" or vsa == "FHRR":
            return
        in_features = 1000000
        out_features = 10
        emb = embeddings.Sinusoid(in_features, out_features, vsa=vsa)
        assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-2


class TestDensity:
    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_embedding_dim(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        assert emb.density_encoding.embedding_dim == dimensions

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_num_embedings(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        assert emb.density_encoding.num_embeddings == dimensions + 1

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_dtype(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 1000
        embedding = 10
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        x = torch.randn(1, embedding)
        if vsa == "BSC":
            assert emb(x).dtype == torch.bool
        elif vsa == "MAP":
            assert emb(x).dtype == torch.float
        elif vsa == "FHRR":
            assert emb(x).dtype == torch.complex64 or emb(x).dtype == torch.complex32
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        if vsa == "HRR":
            return
        dimensions = 10000
        embedding = 10
        emb = embeddings.Density(embedding, dimensions, vsa=vsa)
        return (
            abs(
                torchhd.cosine_similarity(
                    emb.density_encoding.weight[0], emb.density_encoding.weight[1]
                )
            )
            > 0.99
        )
