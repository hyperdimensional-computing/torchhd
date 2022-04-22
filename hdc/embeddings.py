import math
import torch
import torch.nn as nn
from . import functional


class Identity(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, requires_grad=False, **kwargs):
        super(Identity, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.identity_hv(
            self.num_embeddings,
            self.embedding_dim,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()


class Random(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        low=0.0,
        high=1.0,
        randomness=0.0,
        requires_grad=False,
        **kwargs
    ):
        self.low_value = low
        self.high_value = high
        self.randomness = randomness

        super(Random, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.random_hv(
            self.num_embeddings,
            self.embedding_dim,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()


class Level(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        low=0.0,
        high=1.0,
        randomness=0.0,
        requires_grad=False,
        **kwargs
    ):
        self.low_value = low
        self.high_value = high
        self.randomness = randomness

        super(Level, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.level_hv(
            self.num_embeddings,
            self.embedding_dim,
            randomness=self.randomness,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # tranform the floating point input to an index
        # make first variable a copy of the input, then we can reuse the buffer.
        # normalized between 0 and 1
        normalized = (input - self.low_value) / (self.high_value - self.low_value)

        indices = normalized.mul_(self.num_embeddings).floor_()
        indices = indices.clamp_(0, self.num_embeddings - 1).long()

        return super(Level, self).forward(indices)


class Circular(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        low=0.0,
        high=2 * math.pi,
        randomness=0.0,
        requires_grad=False,
        **kwargs
    ):
        self.low_value = low
        self.high_value = high
        self.randomness = randomness

        super(Circular, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.circular_hv(
            self.num_embeddings,
            self.embedding_dim,
            randomness=self.randomness,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # tranform the floating point input to an index
        # make first variable a copy of the input, then we can reuse the buffer.
        # normalized between 0 and 1
        normalized = (input - self.low_value) / (self.high_value - self.low_value)
        normalized.remainder_(1.0)

        indices = normalized.mul_(self.num_embeddings).floor_()
        indices = indices.clamp_(0, self.num_embeddings - 1).long()

        return super(Circular, self).forward(indices)
