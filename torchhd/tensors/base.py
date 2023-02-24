from typing import List, Set, Any
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

    def __add__(self, other: Any):
        if isinstance(other, VSATensor):
            return self.bundle(other)

        return super().__add__(other)

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

    def __mul__(self, other: Any):
        if isinstance(other, VSATensor):
            return self.bind(other)

        return super().__mul__(other)

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

    def __truediv__(self, other: Any) -> Tensor:
        if isinstance(other, VSATensor):
            return self.bind(other.inverse())

        return super().__truediv__(other)

    def inverse(self) -> "VSATensor":
        """Inverse the hypervector for binding"""
        raise NotImplementedError

    def __sub__(self, other: Any) -> Tensor:
        if isinstance(other, VSATensor):
            return self.bundle(other.negative())

        return super().__sub__(other)

    def negative(self) -> "VSATensor":
        """Negate the hypervector for the bundling inverse"""
        raise NotImplementedError

    def permute(self, shifts: int = 1) -> "VSATensor":
        """Permute the hypervector"""
        return super().roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "VSATensor") -> Tensor:
        """Inner product with other hypervectors"""
        raise NotImplementedError

    def cosine_similarity(self, others: "VSATensor") -> Tensor:
        """Cosine similarity with other hypervectors"""
        raise NotImplementedError
