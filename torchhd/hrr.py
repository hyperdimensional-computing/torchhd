import torch
from torch import Tensor
import torch.nn.functional as F
from torch.fft import fft, ifft

from torchhd.base import VSA_Model


class HRR(VSA_Model):
    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "HRR":
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in {torch.float32, torch.float64}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors must be of dtype float16, bfloat16, float32, or float64."
            )

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
    ) -> "HRR":
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in {torch.float32, torch.float64}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors must be of dtype float16, bfloat16, float32, or float64."
            )

        result = torch.ones(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
        )
        result = torch.real(ifft(result))
        result.requires_grad = requires_grad
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
    ) -> "HRR":
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in {torch.float32, torch.float64}:
            name = cls.__name__
            raise ValueError(
                f"{name} vectors must be of dtype float16, bfloat16, float32, or float64."
            )

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.normal_(0, 1.0 / dimensions, generator=generator)

        f = torch.abs(fft(result))
        p = ifft(fft(result) / f).real
        result = torch.nan_to_num(p)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "HRR") -> "HRR":
        return self.add(other)

    def multibundle(self) -> "HRR":
        return self.sum(dim=-2)

    def bind(self, other: "HRR") -> "HRR":
        result = ifft(torch.mul(fft(self), fft(other)))
        return result.real

    def multibind(self) -> "HRR":
        result = ifft(torch.prod(fft(self), dim=-2))
        return result.real

    def inverse(self) -> "HRR":
        return self.flip(dims=(-1,)).roll(1, dims=-1)

    def permute(self, n: int = 1) -> "HRR":
        return self.roll(shifts=n, dim=-1)

    def dot_similarity(self, others: "HRR") -> Tensor:
        return F.linear(self, others)

    def cos_similarity(self, others: "HRR", *, eps=1e-08) -> Tensor:
        self_dot = torch.sum(self * self, dim=-1)
        self_mag = self_dot.sqrt()

        others_dot = torch.sum(others * others, dim=-1)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return self.dot_similarity(others) / magnitude
