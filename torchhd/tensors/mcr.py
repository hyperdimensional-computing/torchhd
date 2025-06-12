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

from torchhd.tensors.base import VSATensor
from torchhd.tensors.basemcr import BaseMCRTensor


class MCRTensor(BaseMCRTensor):
    r"""Modular Composite Representation (MCR)

    Proposed in `Modular Composite Representation <https://link.springer.com/article/10.1007/s12559-013-9243-y>`_, this model works with modular integer vectors.
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
    ) -> "MCRTensor":
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
    ) -> "MCRTensor":
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
    ) -> "MCRTensor":
        return super().random(
            num_vectors,
            dimensions,
            block_size=block_size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def bundle(self, other: "MCRTensor") -> "MCRTensor":
        r"""Bundle the hypervector with normalized complex vector addition.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (MCR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MCRTensor.random(2, 10, block_size=64)
            >>> a
            MCRTensor([32, 26, 22, 22, 34, 30,  2,  2, 40, 43])
            >>> b
            MCRTensor([33, 27, 39, 54, 27, 60, 60,  4, 24,  5])
            >>> a.bundle(b)
            MCRTensor([32, 26, 39, 54, 27, 60,  2,  4, 40,  5])

        """
        self_phasor = self.to_complex_unit()
        other_phasor = other.to_complex_unit()

        # Adding the vectors of each element
        sum_of_phasors = self_phasor + other_phasor

        # To define the ultimate number that the summation will land on
        # we first find the theta of summation then quantize it to block_size
        angels = torch.angle(sum_of_phasors)
        result = self.block_size * (angels / (2 * torch.pi))

        # In cases where the two elements are inverse of each other
        # the sum will be 0 + 0j and it makes the final result to be nan.
        # We return the average of two operands in such a case.
        is_zero = torch.isclose(sum_of_phasors, torch.zeros_like(sum_of_phasors))
        result = torch.where(is_zero, (self + other) / 2, result).round()

        return torch.remainder(result, self.block_size).type(self.dtype)

    def multibundle(self) -> "MCRTensor":
        """Bundle multiple hypervectors"""

        self_phasor = self.to_complex_unit()
        sum_of_phasors = torch.sum(self_phasor, dim=-2)

        # To define the ultimate number that the summation will land on
        # we first find the theta of summation then quantize it to block_size
        angels = torch.angle(sum_of_phasors)
        result = self.block_size * (angels / (2 * torch.pi))

        # In cases where the two elements are inverse of each other
        # the sum will be 0 + 0j and it makes the final result to be nan.
        # We return the average of two operands in such a case.
        is_zero = torch.isclose(sum_of_phasors, torch.zeros_like(sum_of_phasors))
        result = torch.where(is_zero, torch.mean(self, dim=-2, dtype=torch.float), result).round()

        return torch.remainder(result, self.block_size).type(self.dtype)

    def bind(self, other: "MCRTensor") -> "MCRTensor":
        return super().bind(other)

    def multibind(self) -> "MCRTensor":
        """Bind multiple hypervectors"""
        return super().multibind()

    def inverse(self) -> "MCRTensor":
        return super().inverse()

    def permute(self, shifts: int = 1) -> "MCRTensor":
        return super().permute(shifts=shifts)

    def normalize(self) -> "MCRTensor":
        return super().normalize()

    def dot_similarity(self, others: "MCRTensor", *, dtype=None) -> Tensor:
        return super().dot_similarity(others, dtype=dtype)

    def cosine_similarity(self, others: "MCRTensor", *, dtype=None) -> Tensor:
        return super().cosine_similarity(others, dtype=dtype)

