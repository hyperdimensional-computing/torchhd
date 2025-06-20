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
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Set

from torchhd.tensors.basemcr import BaseMCRTensor


class CGRTensor(BaseMCRTensor):
    r"""Cyclic Group Representation (CGR)

    First introduced in `Modular Composite Representation <https://link.springer.com/article/10.1007/s12559-013-9243-y>`_ and then better elaborated in `Understanding hyperdimensional computing for parallel single-pass learning <https://proceedings.neurips.cc/paper_files/paper/2022/file/080be5eb7e887319ff30c792c2cbc28c-Paper-Conference.pdf>`_, this model works with modular integer vectors. It works similar to the MCR class, but uses a bundling based on element-wise mode instead of addition of complex numbers.
    """

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        block_size: int,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "CGRTensor":
        return super().empty(
            num_vectors,
            dimensions,
            block_size=block_size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        block_size: int,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "CGRTensor":
        return super().identity(
            num_vectors,
            dimensions,
            block_size=block_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        block_size: int,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "CGRTensor":
        return super().random(
            num_vectors,
            dimensions,
            block_size=block_size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )


    def bundle(self, other: "CGRTensor") -> "CGRTensor":
        r"""Bundle the hypervector with majority voting. Ties might be broken at random. However, the expected result is that the tie representing the lowest value wins.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (CGR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.CGRTensor.random(2, 10, block_size=64)
            >>> a
            CGRTensor([32, 26, 22, 22, 34, 30,  2,  4, 40, 43])
            >>> b
            CGRTensor([32, 26, 39, 54, 27, 60, 60,  4, 40,  5])
            >>> a.bundle(b)
            CGRTensor([32, 26, 39, 22, 27, 60,  2,  4, 40,  5])

        """
        # Ensure hypervectors are in the same shape, i.e., [..., 1, DIM]
        t1 = self
        if t1.dim() == 1:
            t1 = t1.unsqueeze(0)
        t2 = other
        if t2.dim() == 1:
            t2 = t2.unsqueeze(0)

        t = torch.stack((t1, t2), dim=-2)
        val = t.multibundle()

        # Convert shape back to [DIM] if inputs are plain hypervectors
        need_squeeze = self.dim() == 1 and other.dim() == 1
        if need_squeeze:
            return val.squeeze(0)

        return val

    def multibundle(self) -> "CGRTensor":
        """Bundle multiple hypervectors"""
        # The use of torch.mode() makes untying deterministic as it always
        # returns the lowest index among the ties. For example, if there is an
        # equal amount of 0s and 1s in a bundle, 0 is returned.
        val, _ = torch.mode(self, dim=-2)
        return val

    def bind(self, other: "CGRTensor") -> "CGRTensor":
        return super().bind(other)

    def multibind(self) -> "CGRTensor":
        """Bind multiple hypervectors"""
        return super().multibind()

    def inverse(self) -> "CGRTensor":
        return super().inverse()

    def permute(self, shifts: int = 1) -> "CGRTensor":
        return super().permute(shifts=shifts)

    def normalize(self) -> "CGRTensor":
        return super().normalize()

    def dot_similarity(self, others: "CGRTensor", *, dtype=None) -> Tensor:
        return super().dot_similarity(others, dtype=dtype)

    def cosine_similarity(self, others: "CGRTensor", *, dtype=None) -> Tensor:
        return super().cosine_similarity(others, dtype=dtype)

