import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchhd.functional as functional

__all__ = [
    "Identity",
    "Random",
    "Level",
    "Circular",
    "Projection",
]


class Identity(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.functional.identity_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Identity(5, 3)
        >>> idx = torch.LongTensor([0, 1, 4])
        >>> emb(idx)
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])

    """

    def __init__(self, num_embeddings, embedding_dim, requires_grad=False, **kwargs):
        super(Identity, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }

        self.weight.data.copy_(
            functional.identity_hv(
                self.num_embeddings, self.embedding_dim, **factory_kwargs
            )
        )

        self._fill_padding_idx_with_zero()


class Random(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.functional.random_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Random(5, 3)
        >>> idx = torch.LongTensor([0, 1, 4])
        >>> emb(idx)
        tensor([[ 1., -1.,  1.],
                [ 1., -1.,  1.],
                [ 1.,  1.,  1.]])

    """

    def __init__(self, num_embeddings, embedding_dim, requires_grad=False, **kwargs):
        super(Random, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }

        self.weight.data.copy_(
            functional.random_hv(
                self.num_embeddings, self.embedding_dim, **factory_kwargs
            )
        )

        self._fill_padding_idx_with_zero()


class Level(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.functional.level_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        randomness (float, optional): r-value to interpolate between level at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Level(5, 10, low=-1, high=2)
        >>> x = torch.FloatTensor([0.3, 1.9, -0.8])
        >>> emb(x)
        tensor([[-1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.],
                [-1.,  1., -1.,  1., -1., -1., -1.,  1.,  1., -1.],
                [ 1., -1.,  1., -1., -1., -1.,  1.,  1., -1.,  1.]])

    """

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

        self.weight.data.copy_(
            functional.level_hv(
                self.num_embeddings,
                self.embedding_dim,
                randomness=self.randomness,
                **factory_kwargs
            )
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        indices = functional.value_to_index(
            input, self.low_value, self.high_value, self.num_embeddings
        ).clamp(0, self.num_embeddings - 1)

        return super(Level, self).forward(indices)


class Circular(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.functional.circular_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        low (float, optional): The lower bound of the real number range that the circular levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the circular levels represent. Default: ``2 * pi``
        randomness (float, optional): r-value to interpolate between circular at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Circular(5, 10)
        >>> x = torch.FloatTensor([0.0, 3.14, 6.28])
        >>> emb(x)
        tensor([[ 1., -1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1.],
                [ 1., -1., -1.,  1.,  1.,  1.,  1., -1., -1., -1.],
                [ 1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.]])

    """

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

        self.weight.data.copy_(
            functional.circular_hv(
                self.num_embeddings,
                self.embedding_dim,
                randomness=self.randomness,
                **factory_kwargs
            )
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        indices = functional.value_to_index(
            input, self.low_value, self.high_value, self.num_embeddings
        ).remainder(self.num_embeddings - 1)

        return super(Circular, self).forward(indices)


class Projection(nn.Module):
    r"""Embedding using a random projection matrix.

    Implemented based on `A Theoretical Perspective on Hyperdimensional Computing <https://arxiv.org/abs/2010.07426>`_.
    :math:`\Phi x` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose rows are uniformly sampled at random from the surface of an :math:`m`-dimensional unit sphere.
    This encoding ensures that similarities in the input space are preserved in the hyperspace.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Examples::

        >>> emb = embeddings.Projection(5, 3)
        >>> x = torch.rand(2, 5)
        >>> emb(x)
        tensor([[ 0.2747, -0.8804, -0.6810],
                [ 0.5610, -0.9227,  0.1671]])

    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features, out_features, requires_grad=False, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Projection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1, 1)
        self.weight.data[:] = F.normalize(self.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight)
