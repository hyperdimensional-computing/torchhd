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
from typing import List, Set
import torch
from torch import Tensor


class VSATensor(Tensor):
    """Base class

    Each model must implement the methods specified on this base class.
    """

    supported_dtypes: Set[torch.dtype]

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSATensor":
        """Creates hypervectors representing empty sets"""
        raise NotImplementedError

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSATensor":
        """Creates identity hypervectors for binding"""
        raise NotImplementedError

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        generator=None,
    ) -> "VSATensor":
        """Creates random or uncorrelated hypervectors"""
        raise NotImplementedError

    def bundle(self, other: "VSATensor") -> "VSATensor":
        """Bundle the hypervector with other"""
        raise NotImplementedError

    def multibundle(self) -> "VSATensor":
        """Bundle multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output

    def bind(self, other: "VSATensor") -> "VSATensor":
        """Bind the hypervector with other"""
        raise NotImplementedError

    def multibind(self) -> "VSATensor":
        """Bind multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bind(tensors[1])
        for i in range(2, n):
            output = output.bind(tensors[i])

        return output

    def inverse(self) -> "VSATensor":
        """Inverse the hypervector for binding"""
        raise NotImplementedError

    def negative(self) -> "VSATensor":
        """Negate the hypervector for the bundling inverse"""
        raise NotImplementedError

    def permute(self, shifts: int = 1) -> "VSATensor":
        """Permute the hypervector"""
        raise NotImplementedError

    def normalize(self) -> "VSATensor":
        """Normalize the hypervector"""
        raise NotImplementedError

    def dot_similarity(self, others: "VSATensor") -> Tensor:
        """Inner product with other hypervectors"""
        raise NotImplementedError

    def cosine_similarity(self, others: "VSATensor") -> Tensor:
        """Cosine similarity with other hypervectors"""
        raise NotImplementedError
