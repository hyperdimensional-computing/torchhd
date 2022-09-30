import pytest
import torch

from torchhd import functional
from torchhd import embeddings

# torch.float32,
# torch.float64,
# torch.complex64,
# torch.complex128,
# torch.float16,
# torch.bfloat16,
# torch.uint8,
# torch.int8,
# torch.int16,
# torch.int32,
# torch.int64,
# torch.bool,

# from .utils import (
#     torch_dtypes,
#     torch_complex_dtypes,
#     supported_dtype,
# )

for i in range(5, 20):
    emb = embeddings.Identity(i, 3)
    idx = torch.LongTensor([0, 1, 4])
    res = emb(idx)

    print("{0},{1}".format(res.size(dim=0),res.size(dim=1)))
