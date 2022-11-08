import torch
from typing import Set
from torch import Tensor
import torch.nn.functional as F

from torchhd.base import VSA_Model


def biggest_power_two(n):
    """Returns the biggest power of two <= n"""
    # if n is a power of two simply return it
    if not (n & (n - 1)):
        return n

    # else set only the most significant bit
    return int("1" + (len(bin(n)) - 3) * "0", 2)


class BSC(VSA_Model):
    """Binary Spatter Codes
    
    Proposed in `Binary spatter-coding of ordered K-tuples <https://link.springer.com/chapter/10.1007/3-540-61510-5_146>`_, this model works with binary valued hypervectors.
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    }

    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.bool,
        device=None,
    ) -> "BSC":
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
        )
        return result.as_subclass(cls)

    @classmethod
    def identity_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.bool,
        device=None,
    ) -> "BSC":
        """Creates identity hypervectors for binding"""

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
        )
        return result.as_subclass(cls)

    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        sparsity=0.5,
        generator=None,
        dtype=torch.bool,
        device=None,
    ) -> "BSC":
        """
        Creates random or uncorrelated hypervectors

            sparsity (float, optional): the expected fraction of elements to be in-active. Has no effect on complex hypervectors. Default: ``0.5``.

        """
        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.bernoulli_(1.0 - sparsity, generator=generator)
        return result.as_subclass(cls)

    def bundle(self, other: "BSC", *, generator: torch.Generator = None) -> "BSC":
        """Bundle the hypervector with other"""
        tiebreaker = torch.empty_like(other)
        tiebreaker.bernoulli_(0.5, generator=generator)

        is_majority = self == other
        return self.where(is_majority, tiebreaker)

    def multibundle(self, *, generator: torch.Generator = None) -> "BSC":
        """Bundle multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)

        count = self.sum(dim=-2, dtype=torch.long)

        # add a tiebreaker when there are an even number of hvs
        if n % 2 == 0:
            tiebreaker = torch.empty_like(count)
            tiebreaker.bernoulli_(0.5, generator=generator)
            count += tiebreaker
            n += 1

        threshold = n // 2
        return torch.greater(count, threshold).to(self.dtype)

    def bind(self, other: "BSC") -> "BSC":
        """Bind the hypervector with other"""
        return self.logical_xor(other).to(other.dtype)

    def multibind(self) -> "BSC":
        """Bind multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        n_ = biggest_power_two(n)
        output = self[..., :n_, :]

        # parallelize many XORs in a hierarchical manner
        # for larger batches this is significantly faster
        while output.size(-2) > 1:
            output = torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])

        output = output.squeeze(-2)

        # TODO: as an optimization we could also perform the hierarchical XOR
        # on the leftovers in a recursive fashion
        leftovers = torch.unbind(self[..., n_:, :], -2)
        for i in range(n - n_):
            output = torch.logical_xor(output, leftovers[i])

        return output.to(self.dtype)

    def inverse(self) -> "BSC":
        """Inverse the hypervector for binding"""
        return self.clone()

    def negative(self) -> "BSC":
        """Negate the hypervector for the bundling inverse"""
        return self.logical_not()

    def permute(self, shifts: int = 1) -> "BSC":
        """Permute the hypervector"""
        return self.roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "BSC") -> Tensor:
        """Inner product with other hypervectors"""
        dtype = torch.get_default_dtype()

        min_one = torch.tensor(-1.0, dtype=dtype)
        plus_one = torch.tensor(1.0, dtype=dtype)

        self_as_bipolar = torch.where(self.bool(), min_one, plus_one)
        others_as_bipolar = torch.where(others.bool(), min_one, plus_one)

        return F.linear(self_as_bipolar, others_as_bipolar)

    def cos_similarity(self, others: "BSC") -> Tensor:
        """Cosine similarity with other hypervectors"""
        d = self.size(-1)
        return self.dot_similarity(others) / d
