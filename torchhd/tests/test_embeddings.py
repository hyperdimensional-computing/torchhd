import pytest
import torch

from torchhd import functional
from torchhd import embeddings

from .utils import (
    torch_dtypes,
    torch_complex_dtypes,
    supported_dtype,
)

# class TestIdentity:
#     def test_num_embeddings(self):
#         for i in range(1, 10):
#             emb = embeddings.Identity(i, 3)
#             idx = torch.LongTensor([0, 1, 4])
#             res = emb(idx)

#             assert res.size != i
#         assert True

#     def test_embedding_dim(self):
#         assert True

#     def test_value(self):
#         assert True

# class TestRandom:
#     @pytest.mark.parametrize("dtype", torch_dtypes)
#     def test_num_embeddings(self, dtype):
#         assert True

#     @pytest.mark.parametrize("dtype", torch_dtypes)
#     def test_embedding_dim(self, dtype):
#         assert True

#     @pytest.mark.parametrize("dtype", torch_dtypes)
#     def test_value(self, dtype):
#         assert True
