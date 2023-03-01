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
