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
import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from scipy.stats import binom

import torchhd.functional as functional
from torchhd.tensors.base import VSATensor

__all__ = [
    "SparseDistributed",
    "hopfield",
    "Hopfield",
    "modern_hopfield",
    "attention",
]


class SparseDistributed(nn.Module):
    r"""`Sparse Distributed Memory <https://redwood.berkeley.edu/wp-content/uploads/2020/08/KanervaP_SDMrelated_models1993.pdf>`_

    The Sparse Distributed Memory (SDM) is specified by its (typically random) keys and their values.

    Args:
        memory_size (int): The number of memory key-value pairs.
        key_dim (int): The dimensionality of the key vectors.
        value_dim (int): The dimensionality of the value vectors.
        p (float, optional): The expected fraction of memory address that will contain any value. Default: ``0.000368``.
        kappa (int, optional): The maximum count for each memory cell, values are clipped between [-kappa, kappa]. Default: no clipping.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shapes:
        - Keys: :math:`(n, a)`
        - Values: :math:`(n, c)`

    Examples::
        >>> keys = torchhd.random(6, 512)
        >>> sdm = torchhd.memory.SparseDistributed(100000, 512, 512)
        >>> # use as associative memory
        >>> sdm.write(keys, keys)
        >>> read = sdm.read(keys).sign()
        >>> torchhd.cosine_similarity(read, keys)
        tensor([[ 1.0000,  0.0156, -0.0039, -0.0742,  0.0000, -0.0195],
                [ 0.0156,  1.0000, -0.0352, -0.0586,  0.0000, -0.0039],
                [-0.0039, -0.0352,  1.0000,  0.0156,  0.0820, -0.0234],
                [-0.0742, -0.0586,  0.0156,  1.0000, -0.0039,  0.0000],
                [ 0.0000,  0.0000,  0.0820, -0.0039,  1.0000,  0.0195],
                [-0.0195, -0.0039, -0.0234,  0.0000,  0.0195,  1.0000]])
    """

    memory_size: int
    key_dim: int
    value_dim: int
    keys: VSATensor
    values: VSATensor
    threshold: int
    kappa: Optional[int]

    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        value_dim: int,
        p: float = 0.000368,
        kappa: Optional[int] = None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> None:
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim

        radius = int(binom.ppf(p, key_dim, 0.5))
        self.threshold = key_dim - 2 * radius
        self.kappa = kappa

        keys = functional.random(memory_size, key_dim, dtype=dtype, device=device)
        self.keys = nn.Parameter(keys, requires_grad)

        values = functional.empty(memory_size, value_dim, device=device, dtype=dtype)
        self.values = nn.Parameter(values, requires_grad)

    def read(self, query: Tensor) -> VSATensor:
        r"""Read value from Sparse Distributed Memory whose key is most similar to the query.

        Args:
            query (Tensor): The query vector for the memory lookup.

        Shapes:
            - Query: :math:`(*, d)`
            - Result: :math:`(*, d)`

        """
        # first dims from query, last dim from value
        out_shape = tuple(query.shape[:-1]) + (self.value_dim,)

        if query.dim() == 1:
            query = query.unsqueeze(0)

        intermediate_shape = tuple(query.shape[:-1]) + (self.value_dim,)

        similarity = query @ self.keys.T
        is_active = similarity >= self.threshold

        # sparse matrix-vector multiplication
        r_indices, v_indices = is_active.nonzero().T
        read = query.new_zeros(intermediate_shape)
        read.index_add_(0, r_indices, self.values[v_indices])
        return read.view(out_shape)

    @torch.no_grad()
    def write(self, keys: Tensor, values: Tensor) -> None:
        r"""Write value to Sparse Distributed Memory at address.

        Args:
            address (Tensor): The address vector for the write to memory.
            value (Tensor): The value vector written to memory.

        Shapes:
            - Address: :math:`(*, d)`
            - Value: :math:`(*, d)`

        """

        if keys.dim() == 1:
            keys = keys.unsqueeze(0)

        if values.dim() == 1:
            values = values.unsqueeze(0)

        similarity = keys @ self.keys.T
        is_active = similarity >= self.threshold

        # sparse outer product and addition
        from_indices, to_indices = is_active.nonzero().T
        self.values.index_add_(0, to_indices, values[from_indices])

        if self.kappa is not None:
            self.values.clamp_(-self.kappa, self.kappa)


def hopfield(query: Tensor, memory: Tensor, kappa: int = None) -> Tensor:
    r"""`Classical Hopfield network <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/>`_

    Args:
        query (Tensor): The query vector for the memory lookup.
        memory (Tensor): The items of memory for the memory lookup.

    Shapes:
        - Query: :math:`(*, d)`
        - Memory: :math:`(n, d)`
        - Result: :math:`(*, d)`

    Examples::
        >>> items = torchhd.random(6, 512)
        >>> read = memory.hopfield(items, items).sign()
        >>> torchhd.cosine_similarity(read, items)
        tensor([[ 1.0000,  0.0156, -0.0039, -0.0742,  0.0000, -0.0195],
                [ 0.0156,  1.0000, -0.0352, -0.0586,  0.0000, -0.0039],
                [-0.0039, -0.0352,  1.0000,  0.0156,  0.0820, -0.0234],
                [-0.0742, -0.0586,  0.0156,  1.0000, -0.0039,  0.0000],
                [ 0.0000,  0.0000,  0.0820, -0.0039,  1.0000,  0.0195],
                [-0.0195, -0.0039, -0.0234,  0.0000,  0.0195,  1.0000]])


    """
    product = memory.T @ memory
    torch.diagonal(product).zero_()

    if kappa is not None:
        product.clamp_(-kappa, kappa)

    return query @ product


class Hopfield(nn.Module):
    r"""`Classical Hopfield network <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/>`_

    Args:
        vector_dim (int): The dimensionality of the vectors in the memory.
        kappa (int, optional): The maximum count for each memory cell, values are clipped between [-kappa, kappa]. Default: no clipping.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shapes:
        - Memory: :math:`(d, d)`

    Examples::
        >>> items = torchhd.random(6, 512)
        >>> hopfield = torchhd.memory.Hopfield(512)
        >>> hopfield.write(items)
        >>> read = hopfield.read(items).sign()
        >>> torchhd.cosine_similarity(read, items)
        tensor([[ 1.0000,  0.0156, -0.0039, -0.0742,  0.0000, -0.0195],
                [ 0.0156,  1.0000, -0.0352, -0.0586,  0.0000, -0.0039],
                [-0.0039, -0.0352,  1.0000,  0.0156,  0.0820, -0.0234],
                [-0.0742, -0.0586,  0.0156,  1.0000, -0.0039,  0.0000],
                [ 0.0000,  0.0000,  0.0820, -0.0039,  1.0000,  0.0195],
                [-0.0195, -0.0039, -0.0234,  0.0000,  0.0195,  1.0000]])
    """

    vector_dim: int
    memory: Tensor
    kappa: Optional[int]

    def __init__(
        self,
        vector_dim: int,
        kappa: Optional[int] = None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> None:
        super().__init__()

        self.vector_dim = vector_dim
        self.kappa = kappa

        memory = torch.zeros(
            self.vector_dim, self.vector_dim, device=device, dtype=dtype
        )
        self.memory = nn.Parameter(memory, requires_grad)

    def read(self, query: Tensor) -> Tensor:
        r"""Read value from Hopfield network at key most similar to the query.

        Args:
            query (Tensor): The query vector for the memory lookup.

        Shapes:
            - Query: :math:`(*, d)`
            - Result: :math:`(*, d)`

        """
        return query @ self.memory.T

    @torch.no_grad()
    def write(self, items: Tensor) -> Tensor:
        r"""Write items to Hopfield Memory.

        Args:
            items (Tensor): The item vectors to write to memory.

        Shapes:
            - Items: :math:`(*, d)`

        """

        if items.dim() == 1:
            items = items.unsqueeze(0)

        # Add the outer product to memory
        self.memory.add_(items.T @ items)
        torch.diagonal(self.memory).zero_()

        if self.kappa is not None:
            self.memory.clamp_(-self.kappa, self.kappa)


def modern_hopfield(query: Tensor, memory: Tensor) -> Tensor:
    r"""`Modern Hopfield network <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/>`_

    Also known as Dense Associative Memory.

    Args:
        query (Tensor): The query vector for the memory lookup.
        memory (Tensor): The items of memory for the memory lookup.

    Shapes:
        - Query: :math:`(*, d)`
        - Memory: :math:`(n, d)`
        - Result: :math:`(*, d)`


    Examples::
        >>> items = torchhd.random(6, 512)
        >>> read = memory.dense_associative(items, items).sign()
        >>> torchhd.cosine_similarity(read, items)
        tensor([[ 1.0000,  0.0469, -0.0117,  0.0039, -0.0313, -0.0078],
                [ 0.0469,  1.0000, -0.0352, -0.0039, -0.0391, -0.0078],
                [-0.0117, -0.0352,  1.0000,  0.0547,  0.0742, -0.0352],
                [ 0.0039, -0.0039,  0.0547,  1.0000,  0.0273,  0.0117],
                [-0.0313, -0.0391,  0.0742,  0.0273,  1.0000, -0.0547],
                [-0.0078, -0.0078, -0.0352,  0.0117, -0.0547,  1.0000]])
    """

    d = query.size(-1)

    query = query.unsqueeze(-2)
    repeat = [1 for _ in range(query.dim())]
    repeat[-2] = d

    pos_query = query.repeat(*repeat)
    torch.diagonal(pos_query, dim1=-2, dim2=-1).fill_(1)

    neg_query = query.repeat(*repeat)
    torch.diagonal(neg_query, dim1=-2, dim2=-1).fill_(-1)

    pos_energy = pos_query @ memory.T
    pos_energy = torch.logsumexp(pos_energy, dim=-1)

    neg_energy = neg_query @ memory.T
    neg_energy = torch.logsumexp(neg_energy, dim=-1)

    return pos_energy - neg_energy


def attention(
    query: Tensor, keys: Tensor, values: Tensor, beta: Optional[float] = None
) -> Tensor:
    r"""`Attention mechanism <https://arxiv.org/abs/1706.03762>`_

    Args:
        query (Tensor): The query vector to compare the similarity with the keys.
        keys (Tensor): The key vectors to compare with the query.
        values (Tensor): The value vectors containing retrievable values from memory.
        beta (float, optional): Temperature scalar for the attention weights before the softmax. Default: 1/sqrt(d)


    Shapes:
        - Query: :math:`(*, f)`
        - Keys: :math:`(n, f)`
        - Values: :math:`(n, g)`
        - Result: :math:`(*, g)`

    Examples::
        >>> items = torchhd.random(6, 512)
        >>> read = torchhd.memory.attention(items, items, items).sign()
        >>> torchhd.cosine_similarity(read, items)
        tensor([[ 1.0000,  0.0625,  0.0117, -0.0625, -0.0078, -0.0430],
                [ 0.0625,  1.0000, -0.0195,  0.0703,  0.0469,  0.0508],
                [ 0.0117, -0.0195,  1.0000,  0.0820,  0.0195,  0.0156],
                [-0.0625,  0.0703,  0.0820,  1.0000, -0.0547, -0.0195],
                [-0.0078,  0.0469,  0.0195, -0.0547,  1.0000, -0.0898],
                [-0.0430,  0.0508,  0.0156, -0.0195, -0.0898,  1.0000]])

    """
    if beta is None:
        d = query.size(-1)
        beta = 1 / math.sqrt(d)

    similarity = query @ keys.T
    scores = torch.softmax(beta * similarity, dim=-1)
    return scores @ values
