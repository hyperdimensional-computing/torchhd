import math
from typing import Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchhd.functional as functional
from torchhd.base import VSA_Model
from torchhd.map import MAP

__all__ = [
    "Identity",
    "Random",
    "Level",
    "Thermometer",
    "Circular",
    "Projection",
    "Sinusoid",
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

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(MAP)


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

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(MAP)


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

        return super(Level, self).forward(indices).as_subclass(MAP)


class Thermometer(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.functional.thermometer_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Thermometer(11, 10, low=-1, high=2)
        >>> x = torch.FloatTensor([0.3, 1.9, -0.8])
        >>> emb(x)
        tensor([[ 1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                [ 1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])

    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        low=0.0,
        high=1.0,
        requires_grad=False,
        **kwargs
    ):
        self.low_value = low
        self.high_value = high

        super(Thermometer, self).__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.requires_grad = requires_grad

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }

        self.weight.data.copy_(
            functional.thermometer_hv(
                self.num_embeddings, self.embedding_dim, **factory_kwargs
            )
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        indices = functional.value_to_index(
            input, self.low_value, self.high_value, self.num_embeddings
        ).clamp(0, self.num_embeddings - 1)

        return super(Thermometer, self).forward(indices).as_subclass(MAP)


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

        return super(Circular, self).forward(indices).as_subclass(MAP)


class Projection(nn.Module):
    r"""Embedding using a random projection matrix.

    Implemented based on `A Theoretical Perspective on Hyperdimensional Computing <https://arxiv.org/abs/2010.07426>`_.
    It computes :math:`x \Phi^{\mathsf{T}}` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose rows are uniformly sampled at random from the surface of an :math:`d`-dimensional unit sphere.
    This encoding ensures that similarities in the input space are preserved in the hyperspace.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Examples::

        >>> embed = embeddings.Projection(6, 5)
        >>> x = torch.randn(3, 6)
        >>> x
        tensor([[ 0.4119, -0.4284,  1.8022,  0.3715, -1.4563, -0.2842],
                [-0.3772, -1.2664, -1.5173,  1.3317,  0.4707, -1.3362],
                [-1.8142,  0.0274, -1.0989,  0.8193,  0.7619,  0.9181]])
        >>> embed(x).sign()
        tensor([[-1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.],
                [ 1., -1., -1., -1., -1.]])

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
        nn.init.normal_(self.weight, 0, 1)
        self.weight.data.copy_(F.normalize(self.weight.data))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight).as_subclass(MAP)


class Sinusoid(nn.Module):
    r"""Embedding using a nonlinear random projection

    Implemented based on `Scalable Edge-Based Hyperdimensional Learning System with Brain-Like Neural Adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.
    It computes :math:`\cos(x \Phi^{\mathsf{T}} + b) \odot \sin(x \Phi^{\mathsf{T}})` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose elements are sampled at random from a standard normal distribution and :math:`b \in \mathbb{R}^{d}` is a vectors whose elements are sampled uniformly at random between 0 and :math:`2\pi`.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Examples::

        >>> embed = embeddings.Sinusoid(6, 5)
        >>> x = torch.randn(3, 6)
        >>> x
        tensor([[ 0.5043,  0.3161, -0.0938,  0.6134, -0.1280,  0.3647],
                [-0.1907,  1.6468, -0.3242,  0.8614,  0.3332, -0.2055],
                [-0.8662, -1.3861, -0.1577,  0.1321, -0.1157, -2.8928]])
        >>> embed(x)
        tensor([[-0.0555,  0.2292, -0.1833,  0.0301, -0.2416],
                [-0.0725,  0.7042, -0.5644,  0.2235,  0.3603],
                [-0.9021,  0.8899, -0.9802,  0.3565,  0.2367]])

    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self, in_features, out_features, requires_grad=False, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Sinusoid, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=requires_grad,
        )

        self.bias = nn.parameter.Parameter(
            torch.empty((1, out_features), **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 1)
        nn.init.uniform_(self.bias, 0, 2 * math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = F.linear(input, self.weight)
        output = torch.cos(projected + self.bias) * torch.sin(projected)
        return output.as_subclass(MAP)
