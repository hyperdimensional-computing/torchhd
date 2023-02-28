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
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from scipy.stats import binom

import torchhd.functional as functional
from torchhd.tensors.base import VSATensor

__all__ = [
    "SparseDistributed",
]


class SparseDistributed(nn.Module):
    r"""`Sparse Distributed Memory <https://redwood.berkeley.edu/wp-content/uploads/2020/08/KanervaP_SDMrelated_models1993.pdf>`_

    The Sparse Distributed Memory (SDM) is specified by its (typically random) addresses and its content.

    Args:
        num_addresses (int): The number of addresses in he memory.
        address_dim (int): The dimensionality of the address vectors.
        content_dim (int): The dimensionality of the content vectors.
        p (float, optional): The expected fraction of memory address that will contain any value. Default: ``0.000368``.
        kappa (int, optional): The maximum count for each memory cell, values are clipped between [-kappa, kappa]. Default: no clipping.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shapes:
        - Addresses: :math:`(n, d)`
        - Content: :math:`(n, d)`

    Examples::
        >>> address = torchhd.random(6, 512)
        >>> sdm = torchhd.memory.SparseDistributed(100000, 512, 512)
        >>> # use as associative memory
        >>> sdm.write(address, address)
        >>> read = sdm.read(address)
        >>> torchhd.cosine_similarity(read, address)
        MAPTensor([[ 1.0000,  0.0156, -0.0039, -0.0742,  0.0000, -0.0195],
                   [ 0.0156,  1.0000, -0.0352, -0.0586,  0.0000, -0.0039],
                   [-0.0039, -0.0352,  1.0000,  0.0156,  0.0820, -0.0234],
                   [-0.0742, -0.0586,  0.0156,  1.0000, -0.0039,  0.0000],
                   [ 0.0000,  0.0000,  0.0820, -0.0039,  1.0000,  0.0195],
                   [-0.0195, -0.0039, -0.0234,  0.0000,  0.0195,  1.0000]])
    """

    addresses: VSATensor
    content: VSATensor
    threshold: int
    kappa: Optional[int]

    def __init__(
        self,
        num_addresses: int,
        address_dim: int,
        content_dim: int,
        p: float = 0.000368,
        kappa: Optional[int] = None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> None:
        super().__init__()

        radius = int(binom.ppf(p, address_dim, 0.5))
        self.threshold = address_dim - 2 * radius
        self.kappa = kappa

        addresses = functional.random(
            num_addresses, address_dim, dtype=dtype, device=device
        )
        self.addresses = nn.Parameter(addresses, requires_grad)

        content = functional.empty(num_addresses, content_dim, device=device)
        self.content = nn.Parameter(content, requires_grad)

    def read(self, address: Tensor) -> Tensor:
        r"""Read value from Sparse Distributed Memory at address.

        Args:
            address (Tensor): The address vector for the memory lookup.

        Shapes:
            - Address: :math:`(*, d)`

        """

        if address.dim() == 1:
            address.unsqueeze_(0)

        similarity = functional.dot_similarity(address, self.addresses)
        is_active = similarity >= self.threshold

        # sparse matrix-vector multiplication
        r_indices, c_indices = is_active.nonzero().T
        read = torch.zeros_like(address)
        read.index_add_(0, r_indices, self.content[c_indices])
        return read.sign()

    def write(self, address: Tensor, value: Tensor) -> Tensor:
        r"""Write value to Sparse Distributed Memory at address.

        Args:
            address (Tensor): The address vector for the write to memory.
            value (Tensor): The value vector written to memory.

        Shapes:
            - Address: :math:`(*, d)`
            - Value: :math:`(*, d)`

        """

        if address.dim() == 1:
            address.unsqueeze_(0)

        if value.dim() == 1:
            value.unsqueeze_(0)

        similarity = functional.dot_similarity(address, self.addresses)
        is_active = similarity >= self.threshold

        # sparse outer product and addition
        v_indices, c_indices = is_active.nonzero().T
        self.content.index_add_(0, c_indices, value[v_indices])

        if self.kappa is not None:
            self.content.clamp_(-self.kappa, self.kappa)
