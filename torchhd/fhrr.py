import math
import torch
from torch import Tensor
import torch.nn.functional as F

from torchhd.base import VSA_Model

type_conversion = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


class FHRR(VSA_Model):
    """Fourier Holographic Reduced Representation"""

    supported_dtypes: set[torch.dtype] = {torch.complex64, torch.complex128}

    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
    ) -> "FHRR":
        """Creates hypervectors representing empty sets"""

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
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
    ) -> "FHRR":
        """Creates identity hypervectors for binding"""

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
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
        generator=None,
    ) -> "FHRR":
        """Creates random or uncorrelated hypervectors"""

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        dtype = type_conversion[dtype]

        size = (num_vectors, dimensions)
        angle = torch.empty(size, dtype=dtype, device=device)
        angle.uniform_(-math.pi, math.pi, generator=generator)
        magnitude = torch.ones_like(angle)

        result = torch.polar(magnitude, angle)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "FHRR") -> "FHRR":
        """Bundle the hypervector with other"""
        return self.add(other)

    def multibundle(self) -> "FHRR":
        """Bundle multiple hypervectors"""
        return self.sum(dim=-2, dtype=self.dtype)

    def bind(self, other: "FHRR") -> "FHRR":
        """Bind the hypervector with other"""
        return self.mul(other)

    def multibind(self) -> "FHRR":
        """Bind multiple hypervectors"""
        return self.prod(dim=-2, dtype=self.dtype)

    def inverse(self) -> "FHRR":
        """Inverse the hypervector for binding"""
        return self.conj()

    def negative(self) -> "FHRR":
        """Negate the hypervector for the bundling inverse"""
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "FHRR":
        """Permute the hypervector"""
        return self.roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "FHRR") -> Tensor:
        """Inner product with other hypervectors"""
        return F.linear(self, others.conj()).real

    def cos_similarity(self, others: "FHRR", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        self_dot = torch.real(self * self.conj()).sum(dim=-1)
        self_mag = self_dot.sqrt()

        others_dot = torch.real(others * others.conj()).sum(dim=-1)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return self.dot_similarity(others) / magnitude
