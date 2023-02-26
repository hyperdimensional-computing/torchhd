import pytest
import torch

import torchhd
from torchhd import functional
from torchhd import embeddings
from torchhd.tensors.hrr import HRRTensor

from torchhd import types
from .utils import (
    torch_dtypes,
    torch_complex_dtypes,
    supported_dtype,
    vsa_tensors
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

        if vsa == 'BSC':
            assert emb(idx).dtype == torch.bool
        elif vsa == 'MAP' or vsa == 'HRR':
            assert emb(idx).dtype == torch.float
        elif vsa == 'FHRR':
            assert emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 1000000
        embedding = 4
        emb = embeddings.Empty(embedding, dimensions, vsa=vsa)
        if vsa == 'BSC':
            assert abs(torchhd.cosine_similarity(emb.weight[0], emb.weight[1])) < 0.5e-2
        elif vsa == 'MAP':
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == 'HRR':
            assert torch.all(emb.weight == 0.0).item()
        elif vsa == 'FHRR':
            assert torch.all(emb.weight == 0.+0.j).item()
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
        if vsa == 'BSC':
            assert emb(idx).dtype == torch.bool
        elif vsa == 'MAP' or vsa == 'HRR':
            assert emb(idx).dtype == torch.float
        elif vsa == 'FHRR':
            assert emb(idx).dtype == torch.complex64 or emb(idx).dtype == torch.complex32
        else:
            return

    @pytest.mark.parametrize("vsa", vsa_tensors)
    def test_value(self, vsa):
        dimensions = 6
        embedding = 4
        emb = embeddings.Identity(embedding, dimensions, vsa=vsa)
        if vsa == 'BSC':
            assert torch.all(emb.weight == False).item()
        elif vsa == 'MAP':
            assert torch.all(emb.weight == 1.0).item()
        elif vsa == 'HRR':
            ten = HRRTensor([[1., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0.]])
            assert torch.all(ten == emb.weight.data).item()
        elif vsa == 'FHRR':
            assert torch.all(emb.weight == 1.+0.j).item()
        else:
            return
