import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Set

from torchhd.base import VSA_Model


class MAP(VSA_Model):
    """Multiply Add Permute
    
    Proposed in `Multiplicative Binding, Representation Operators & Analogy <https://www.researchgate.net/publication/215992330_Multiplicative_Binding_Representation_Operators_Analogy>`_, this model works with bipolar hypervectors.
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates hypervectors representing empty sets"""

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def identity_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates identity hypervectors for binding"""

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.ones(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates random or uncorrelated hypervectors"""

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=device)
        select.bernoulli_(generator=generator)

        result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "MAP") -> "MAP":
        """Bundle the hypervector with other"""
        return self.add(other)

    def multibundle(self) -> "MAP":
        """Bundle multiple hypervectors"""
        return self.sum(dim=-2, dtype=self.dtype)

    def bind(self, other: "MAP") -> "MAP":
        """Bind the hypervector with other"""
        return self.mul(other)

    def multibind(self) -> "MAP":
        """Bind multiple hypervectors"""
        return self.prod(dim=-2, dtype=self.dtype)

    def inverse(self) -> "MAP":
        """Inverse the hypervector for binding"""
        return self.clone()

    def negative(self) -> "MAP":
        """Negate the hypervector for the bundling inverse"""
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "MAP":
        """Permute the hypervector"""
        return self.roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "MAP") -> Tensor:
        """Inner product with other hypervectors"""
        dtype = torch.get_default_dtype()
        return F.linear(self.to(dtype), others.to(dtype))

    def cos_similarity(self, others: "MAP", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        dtype = torch.get_default_dtype()

        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = self_dot.sqrt()

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return F.linear(self.to(dtype), others.to(dtype)) / magnitude
