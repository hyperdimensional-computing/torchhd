import torch
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
    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.bool,
        device=None,
    ) -> "BSC":

        if dtype in {torch.complex64, torch.complex128, torch.float16, torch.bfloat16}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors cannot be of dtype complex64, complex128, float16, or bfloat16."
            )

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
        if dtype in {torch.complex64, torch.complex128, torch.float16, torch.bfloat16}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors cannot be of dtype complex64, complex128, float16, or bfloat16."
            )

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
        if dtype in {torch.complex64, torch.complex128, torch.float16, torch.bfloat16}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors cannot be of dtype complex64, complex128, float16, or bfloat16."
            )

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.bernoulli_(1.0 - sparsity, generator=generator)
        return result.as_subclass(cls)

    def bundle(self, other: "BSC", *, generator: torch.Generator = None) -> "BSC":
        tiebreaker = torch.empty_like(other)
        tiebreaker.bernoulli_(0.5, generator=generator)

        is_majority = self == other
        return self.where(is_majority, tiebreaker)

    def multibundle(self, *, generator: torch.Generator = None) -> "BSC":
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
        return count > threshold

    def bind(self, other: "BSC") -> "BSC":
        return self.logical_xor(other).to(other.dtype)

    def multibind(self) -> "BSC":
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

        return output

    def inverse(self) -> "BSC":
        return self.clone()

    def negative(self) -> "BSC":
        return self.logical_not()

    def permute(self, n: int = 1) -> "BSC":
        return self.roll(shifts=n, dims=-1)

    def dot_similarity(self, others: "BSC") -> Tensor:
        dtype = torch.get_default_dtype()
        
        min_one = torch.tensor(-1.0, dtype=dtype)
        plus_one = torch.tensor(1.0, dtype=dtype)

        self_as_bipolar = torch.where(self.bool(), min_one, plus_one)
        others_as_bipolar = torch.where(others.bool(), min_one, plus_one)

        return F.linear(self_as_bipolar, others_as_bipolar)

    def cos_similarity(self, others: "BSC") -> Tensor:
        d = self.size(-1)
        return self.dot_similarity(others) / d
