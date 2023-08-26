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
import math
from typing import Type, Union, Optional, Literal, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import torchhd.functional as functional
from torchhd.tensors.base import VSATensor
from torchhd.tensors.map import MAPTensor
from torchhd.tensors.fhrr import FHRRTensor, type_conversion as fhrr_type_conversion
from torchhd.tensors.hrr import HRRTensor
from torchhd.types import VSAOptions


__all__ = [
    "Empty",
    "Identity",
    "Random",
    "Level",
    "Thermometer",
    "Circular",
    "Projection",
    "Sinusoid",
    "Density",
    "FractionalPower",
]


class Empty(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.empty`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Empty(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAPTensor([[0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.]])

        >>> emb = embeddings.Empty(4, 6, "FHRR")
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        FHRRTensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs

        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.empty(
            num_embeddings, embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        # we don't need to set the padding to empty because it is already empty.

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.empty(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(input).as_subclass(vsa_tensor)


class Identity(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.identity`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Identity(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAPTensor([[1., 1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 1.]])

        >>> emb = embeddings.Identity(4, 6, "HRR")
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        HRRTensor([[1., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0., 0.]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs

        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.identity(
            num_embeddings, embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        self._fill_padding_idx_with_empty()

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.identity(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

        self._fill_padding_idx_with_empty()

    def _fill_padding_idx_with_empty(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        if self.padding_idx is not None:
            with torch.no_grad():
                empty = functional.empty(
                    1, self.embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
                )
                self.weight[self.padding_idx].copy_(empty.squeeze(0))

    def forward(self, input: Tensor) -> Tensor:
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(input).as_subclass(vsa_tensor)


class Random(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.random`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Random(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAPTensor([[-1.,  1., -1.,  1., -1., -1.],
                   [ 1., -1., -1., -1.,  1., -1.],
                   [ 1., -1.,  1.,  1.,  1.,  1.]])

        >>> emb = embeddings.Random(4, 6, "BSC")
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        BSCTensor([[ True, False, False, False, False,  True],
                   [False,  True,  True,  True, False,  True],
                   [False, False,  True, False, False, False]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs

        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.random(
            num_embeddings, embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        self._fill_padding_idx_with_empty()

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.random(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

        self._fill_padding_idx_with_empty()

    def _fill_padding_idx_with_empty(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        if self.padding_idx is not None:
            with torch.no_grad():
                empty = functional.empty(
                    1, self.embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
                )
                self.weight[self.padding_idx].copy_(empty.squeeze(0))

    def forward(self, input: Tensor) -> Tensor:
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(input).as_subclass(vsa_tensor)


class Level(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.level`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        randomness (float, optional): r-value to interpolate between level-hypervectors at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the closed bound.

    Examples::

        >>> emb = embeddings.Level(4, 6)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.6444, 0.9286, 0.9225, 0.3675])
        >>> emb(x)
        MAPTensor([[ 1.,  1.,  1., -1.,  1.,  1.],
                   [ 1.,  1., -1.,  1.,  1.,  1.],
                   [ 1.,  1., -1.,  1.,  1.,  1.],
                   [ 1.,  1.,  1., -1., -1.,  1.]])

        >>> emb = embeddings.Level(4, 6, "BSC")
        >>> x = torch.rand(4)
        >>> x
        tensor([0.1825, 0.1541, 0.4435, 0.1512])
        >>> emb(x)
        BSCTensor([[False,  True, False, False, False, False],
                   [False,  True, False, False, False, False],
                   [False,  True, False, False, False, False],
                   [False,  True, False, False, False, False]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "low",
        "high",
        "randomness",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    low: float
    high: float
    randomness: float
    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        low: float = 0.0,
        high: float = 1.0,
        randomness: float = 0.0,
        requires_grad: bool = False,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs
        self.low = low
        self.high = high
        self.randomness = randomness

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.level(
            num_embeddings,
            embedding_dim,
            self.vsa,
            randomness=randomness,
            **factory_kwargs,
            **self.vsa_kwargs,
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.level(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                randomness=self.randomness,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)


class Thermometer(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.thermometer`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the closed bound.

    Examples::

        >>> emb = embeddings.Thermometer(4, 6)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.5295, 0.0618, 0.0675, 0.1750])
        >>> emb(x)
        MAPTensor([[ 1.,  1.,  1.,  1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1.],
                   [ 1.,  1., -1., -1., -1., -1.]])

        >>> emb = embeddings.Thermometer(4, 6, "FHRR")
        >>> x = torch.rand(4)
        >>> x
        tensor([0.2668, 0.7668, 0.8083, 0.6247])
        >>> emb(x)
        FHRRTensor([[ 1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "low",
        "high",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    low: float
    high: float
    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        low: float = 0.0,
        high: float = 1.0,
        requires_grad: bool = False,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs
        self.low = low
        self.high = high

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.thermometer(
            num_embeddings, embedding_dim, self.vsa, **factory_kwargs, **self.vsa_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.thermometer(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)


class Flocet(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.thermometer`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the closed bound.

    Examples::

        >>> emb = embeddings.Thermometer(4, 6)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.5295, 0.0618, 0.0675, 0.1750])
        >>> emb(x)
        MAPTensor([[ 1.,  1.,  1.,  1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1.],
                   [-1., -1., -1., -1., -1., -1.],
                   [ 1.,  1., -1., -1., -1., -1.]])

        >>> emb = embeddings.Thermometer(4, 6, "FHRR")
        >>> x = torch.rand(4)
        >>> x
        tensor([0.2668, 0.7668, 0.8083, 0.6247])
        >>> emb(x)
        FHRRTensor([[ 1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "low",
        "high",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    low: float
    high: float
    vsa: VSAOptions

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        low: float = 0.0,
        high: float = 1.0,
        requires_grad: bool = False,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.low = low
        self.high = high

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.flocet(
            num_embeddings, embedding_dim, self.vsa, **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.flocet(
                self.num_embeddings, self.embedding_dim, self.vsa, **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)


class Circular(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.circular`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        phase (float, optional): The zero offset of the real number periodic interval that the circular levels represent. Default: ``0.0``
        period (float, optional): The period of the real number periodic interval that the circular levels represent. Default: ``2 * pi``
        randomness (float, optional): r-value to interpolate between circular-hypervectors at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Circular(4, 6)
        >>> angle = torch.tensor([0.0, 3.141, 6.282, 9.423])
        >>> emb(angle)
        MAPTensor([[-1., -1., -1., -1., -1.,  1.],
                   [-1., -1.,  1.,  1., -1., -1.],
                   [-1., -1., -1., -1., -1.,  1.],
                   [-1., -1.,  1.,  1., -1., -1.]])

        >>> emb = embeddings.Circular(4, 6, "BSC")
        >>> angle = torch.tensor([0.0, 3.141, 6.282, 9.423])
        >>> emb(angle)
        BSCTensor([[False,  True, False, False,  True,  True],
                   [False, False, False, False, False,  True],
                   [False,  True, False, False,  True,  True],
                   [False, False, False, False, False,  True]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa",
        "phase",
        "period",
        "randomness",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    phase: float
    period: float
    randomness: float
    vsa_tensor: Type[VSATensor]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa: VSAOptions = "MAP",
        phase: float = 0.0,
        period: float = 2 * math.pi,
        randomness: float = 0.0,
        requires_grad: bool = False,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa = vsa
        self.vsa_kwargs = kwargs
        self.phase = phase
        self.period = period
        self.randomness = randomness

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.circular(
            num_embeddings,
            embedding_dim,
            self.vsa,
            randomness=randomness,
            **factory_kwargs,
            **self.vsa_kwargs,
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.circular(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa,
                randomness=self.randomness,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        mapped = functional.map_range(
            input, self.phase, self.period, 0, self.num_embeddings
        )
        index = mapped.round().long() % self.num_embeddings
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)


class Projection(nn.Module):
    r"""Embedding using a random projection matrix.

    Implemented based on `A Theoretical Perspective on Hyperdimensional Computing <https://arxiv.org/abs/2010.07426>`_.
    It computes :math:`x \Phi^{\mathsf{T}}` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose rows are uniformly sampled at random from the surface of an :math:`d`-dimensional unit sphere.
    This encoding ensures that similarities in the input space are preserved in the hyperspace.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
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
        MAPTensor([[-1.,  1.,  1.,  1.,  1.],
                   [ 1.,  1.,  1.,  1.,  1.],
                   [ 1., -1., -1., -1., -1.]])

    """

    __constants__ = ["in_features", "out_features", "vsa"]
    in_features: int
    out_features: int
    vsa: VSAOptions
    weight: torch.Tensor

    def __init__(
        self,
        in_features,
        out_features,
        vsa: VSAOptions = "MAP",
        requires_grad=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Projection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vsa = vsa

        if vsa not in {"MAP", "HRR", "VTB"}:
            raise ValueError(
                f"Projection embedding supports MAP, HRR, VTB but provided: {vsa}"
            )

        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 1)
        self.weight.data.copy_(F.normalize(self.weight.data))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return F.linear(input, self.weight).as_subclass(vsa_tensor)


class Sinusoid(nn.Module):
    r"""Embedding using a nonlinear random projection

    Implemented based on `Scalable Edge-Based Hyperdimensional Learning System with Brain-Like Neural Adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.
    It computes :math:`\cos(x \Phi^{\mathsf{T}} + b) \odot \sin(x \Phi^{\mathsf{T}})` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose rows are uniformly sampled at random from the surface of an :math:`d`-dimensional unit sphere and :math:`b \in \mathbb{R}^{d}` is a vectors whose elements are sampled uniformly at random between 0 and :math:`2\pi`.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
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
        MAPTensor([[-0.0555,  0.2292, -0.1833,  0.0301, -0.2416],
                   [-0.0725,  0.7042, -0.5644,  0.2235,  0.3603],
                   [-0.9021,  0.8899, -0.9802,  0.3565,  0.2367]])

    """

    __constants__ = ["in_features", "out_features", "vsa"]
    in_features: int
    out_features: int
    vsa: VSAOptions
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        in_features,
        out_features,
        vsa: VSAOptions = "MAP",
        requires_grad=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Sinusoid, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vsa = vsa

        if vsa not in {"MAP", "HRR", "VTB"}:
            raise ValueError(
                f"Sinusoid embedding supports MAP, HRR, VTB but provided: {vsa}"
            )

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
        self.weight.data.copy_(F.normalize(self.weight.data))
        nn.init.uniform_(self.bias, 0, 2 * math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = F.linear(input, self.weight)
        output = torch.cos(projected + self.bias) * torch.sin(projected)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return output.as_subclass(vsa_tensor)


class Density(nn.Module):
    """Performs the transformation of input data into hypervectors according to the intRVFL model.

    See details in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels of the thermometer encoding represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels of the thermometer encoding represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.


    Examples::

        >>> embed = embeddings.Density(6, 5)
        >>> x = torch.randn(3, 6)
        >>> x
        tensor([[ 0.5430,  1.0740,  0.7250, -0.3410, -0.1318,  1.3188],
                [ 0.4373,  1.2400, -0.2264,  1.2448, -0.2040, -0.7831],
                [ 1.7460, -0.7359, -1.3271,  0.4338, -0.2401,  1.6553]])
        >>> embed(x)
        MAPTensor([[ 2.,  2., -2., -2.,  0.],
                   [ 4.,  0.,  6.,  4.,  0.],
                   [ 4., -4., -2., -4., -4.]])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        vsa: VSAOptions = "MAP",
        low: float = 0.0,
        high: float = 1.0,
        device=None,
        dtype=None,
        requires_grad: bool = False,
        **kwargs,
    ):
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": requires_grad,
        }
        super(Density, self).__init__()
        # A set of random vectors used as unique IDs for features of the dataset.
        self.key = Random(in_features, out_features, vsa, **factory_kwargs, **kwargs)
        # Thermometer encoding used for transforming input data.
        self.density_encoding = Thermometer(
            out_features + 1,
            out_features,
            vsa,
            low=low,
            high=high,
            **factory_kwargs,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        self.key.reset_parameters()
        self.density_encoding.reset_parameters()

    # Specify the steps needed to perform the encoding
    def forward(self, input: Tensor) -> Tensor:
        # Perform binding of key and value vectors
        output = functional.bind(self.key.weight, self.density_encoding(input))
        # Perform the superposition operation on the bound key-value pairs
        return functional.multibundle(output)


class FractionalPower(nn.Module):
    """Class for fractional power encoding (FPE) method that forms hypervectors for given values, kernel shape, bandwidth, and dimensionality. Implements similarity-preserving hypervectors approximating desired kernel shape as described in `Computing on Functions Using Randomized Vector Representations <https://arxiv.org/abs/2109.03429>`_.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        distribution (str, optional): hyperparameter defining the shape of the kernel by specifying a particular probability distribution that is used to sample the base hypervector(s).  Default: ``"sinc"``.
        bandwidth (float, optional): positive hyperparameter defining the width of the similarity kernel. Lower values lead to broader kernels while larger values lead to more narrow kernels. Default: ``1.0``.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"FHRR"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> embed = embeddings.FractionalPower(1, 6, "sinc", 1.0, "FHRR")
        >>> embed(torch.arange(1, 4, 1.).view(-1, 1))
        FHRRTensor([[-0.7181-0.6959j, -0.5269+0.8499j, -0.0848+0.9964j,  0.9720-0.2348j,
              0.6358+0.7718j,  0.4352+0.9003j],
            [ 0.0314+0.9995j, -0.4447-0.8957j, -0.9856-0.1689j,  0.8897-0.4565j,
             -0.1915+0.9815j, -0.6212+0.7836j],
            [ 0.6730-0.7396j,  0.9956+0.0940j,  0.2519-0.9678j,  0.7576-0.6527j,
             -0.8793+0.4762j, -0.9759-0.2183j]])

    """

    # The collection of distributions for basic predefined kernels
    predefined_kernels = {
        "sinc": torch.distributions.Uniform(-math.pi, math.pi),
        "gaussian": torch.distributions.Normal(0.0, 1.0),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        distribution: Union[
            torch.distributions.Distribution, Literal["sinc", "gaussian"]
        ] = "sinc",
        bandwidth: float = 1.0,
        vsa: Literal["HRR", "FHRR"] = "FHRR",
        device=None,
        dtype=None,
        requires_grad: bool = False,
    ) -> None:
        super(FractionalPower, self).__init__()

        self.in_features = in_features  # data dimensions
        self.out_features = out_features  # hypervector dimensions
        self.bandwidth = bandwidth
        self.requires_grad = requires_grad

        if vsa not in {"HRR", "FHRR"}:
            raise ValueError(
                f"FractionalPower embedding only supports HRR and FHRR but provided: {vsa}"
            )

        self.vsa_tensor = functional.get_vsa_tensor_class(vsa)

        # If a specific dtype is specified make sure it is supported by the VSA model
        if dtype != None and dtype not in self.vsa_tensor.supported_dtypes:
            raise ValueError(f"dtype {dtype} not supported by {vsa}")

        # The internal weights/phases are stored as floats even if the output is a complex tensor
        if dtype != None and vsa == "FHRR":
            dtype = fhrr_type_conversion[dtype]

        factory_kwargs = {"device": device, "dtype": dtype}

        # If the distribution is a string use the presets in predefined_kernels
        if isinstance(distribution, str):
            try:
                self.distribution = self.predefined_kernels[distribution]
            except KeyError:
                available_names = ", ".join(list(self.predefined_kernels.keys()))
                raise ValueError(
                    f"{distribution} kernel is not supported, use one of: {available_names}, or provide a custom torch distribution."
                )
        else:
            self.distribution = distribution

        # Initialize encoding's parameters
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, **factory_kwargs),
            requires_grad,
        )
        self.reset_parameters()

    # Sample the angles using the provided distribution
    def reset_parameters(self) -> None:
        """Generate the angles for basis hypervector(s) to be used for encoding the data."""

        sample_shape = self.distribution.sample().shape

        # Check HD/VSA model type
        if self.vsa_tensor == FHRRTensor:
            # Generate the angles for base hypervector(s) that determines the shape of the FPE kernel
            # If the distribution is one-dimensional this implies that base hypervectors are independent so it is safe to generate self.in_features * self.out_features independent samples
            if sample_shape == ():
                # Draw angles from a uniform  distribution for base hypervector(s). Note that data dimensions here are independent but this does not have to be always the case
                phases = self.distribution.sample((self.out_features, self.in_features))
                phases = phases.to(self.weight)
                self.weight.data.copy_(phases)

            # If base hypervectors are correlated then the dimensionality of the distribution should match that of the data
            elif sample_shape == (self.in_features,):
                phases = self.distribution.sample((self.out_features,))
                phases = phases.to(self.weight)
                self.weight.data.copy_(phases)

            # Raise error due to the ambiguity of the situation
            else:
                raise ValueError(
                    f"The provided distribution has shape {sample_shape} while the input data expects shape () or ({self.in_features},) so there is a mismatch."
                )

        elif self.vsa_tensor == HRRTensor:
            # Fewer angles are needed
            dimensions_real = int((self.out_features - 1) / 2)

            # Generate the angles for base hypervector(s) that determines the shape of the FPE kernel
            # If the distribution is one-dimensional this implies that base hypervectors are independent so it is safe to generate self.in_features * self.out_features independent samples
            if sample_shape == ():
                # Draw angles from a uniform  distribution for base hypervector(s). Note that data dimensions here are independent but this does not have to be always the case
                phases = self.distribution.sample((dimensions_real, self.in_features))

            # If base hypervectors are correlated then the dimensionality of the distribution should match that of the data
            elif sample_shape == (self.in_features,):
                phases = self.distribution.sample((dimensions_real,))

            # Raise error due to the ambiguity of the situation
            else:
                raise ValueError(
                    f"The provided distribution has shape {sample_shape} while the input data expects shape () or ({self.in_features},) so there is a mismatch."
                )

            # Make the generated angles negatively symmetric so they look as a spectrum
            phases = torch.cat(
                (
                    phases,
                    torch.zeros(1, self.in_features),
                    -torch.flip(phases, dims=[0]),
                ),
                dim=0,
            )
            if self.out_features % 2 == 0:
                phases = torch.cat((torch.zeros(1, self.in_features), phases), dim=0)

            phases = phases.to(self.weight)
            # Set the generated angles to the object's parameters
            self.weight.data.copy_(phases)

    def basis(self):
        """Return the values of the base hypervector(s)"""

        # Use the angles in self.weight to obtain the values of the base hypervector(s)
        if self.vsa_tensor == FHRRTensor:
            hvs = torch.complex(self.weight.cos(), self.weight.sin()).T
            hvs = hvs.as_subclass(FHRRTensor)

        elif self.vsa_tensor == HRRTensor:
            complex_hv = torch.complex(self.weight.cos(), self.weight.sin()).T
            hvs = torch.real(
                torch.fft.ifft(torch.fft.ifftshift(complex_hv, dim=1), dim=1)
            )
            hvs = hvs.as_subclass(HRRTensor)

        return hvs

    def forward(self, input: Tensor) -> Tensor:
        """Creates a fractional power encoding (FPE) for given values.

        Args:
            input (Tensor): values for which FPE hypervectors should be generated. Either a vector or a batch of vectors.

        Shapes:
            - Input: :math:`(*, f)` where f is the in_features and * is an optional batch dimension.
            - Output: :math:`(*, d)` where d is the out_features and * is an optional batch dimension.

        """

        # Perform FPE of the desired values using the base hypervector(s)
        # Simultaneously computes angles for given values and their sum that is equivalent to the binding
        if self.vsa_tensor == FHRRTensor:
            phases = F.linear(self.bandwidth * input, self.weight)
            hv = torch.complex(phases.cos(), phases.sin())
            hv = hv.as_subclass(FHRRTensor)

        elif self.vsa_tensor == HRRTensor:
            phases = F.linear(self.bandwidth * input, self.weight)
            hv = torch.complex(phases.cos(), phases.sin())
            hv = torch.real(torch.fft.ifft(torch.fft.ifftshift(hv, dim=1), dim=1))
            hv = hv.as_subclass(HRRTensor)

        return hv


class DensityFlocet(nn.Module):
    """Performs the transformation of input data into hypervectors according to the intRVFL model.

    See details in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels of the thermometer encoding represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels of the thermometer encoding represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.


    Examples::

        >>> embed = embeddings.Density(6, 5)
        >>> x = torch.randn(3, 6)
        >>> x
        tensor([[ 0.5430,  1.0740,  0.7250, -0.3410, -0.1318,  1.3188],
                [ 0.4373,  1.2400, -0.2264,  1.2448, -0.2040, -0.7831],
                [ 1.7460, -0.7359, -1.3271,  0.4338, -0.2401,  1.6553]])
        >>> embed(x)
        MAPTensor([[ 2.,  2., -2., -2.,  0.],
                   [ 4.,  0.,  6.,  4.,  0.],
                   [ 4., -4., -2., -4., -4.]])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        vsa: VSAOptions = "MAP",
        low: float = 0.0,
        high: float = 1.0,
        device=None,
        dtype=None,
        requires_grad: bool = False,
    ):
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": requires_grad,
        }
        super(DensityFlocet, self).__init__()
        # A set of random vectors used as unique IDs for features of the dataset.
        self.key = Random(in_features, out_features, vsa=vsa, **factory_kwargs)

        self.flocet_encoding = Flocet(
            math.floor(out_features / 2) + 1,
            out_features,
            vsa,
            low=low,
            high=high,
            **factory_kwargs,
        )

    def reset_parameters(self) -> None:
        self.key.reset_parameters()
        self.flocet_encoding.reset_parameters()

    # Specify the steps needed to perform the encoding
    def forward(self, input: Tensor) -> Tensor:
        # Perform binding of key and value vectors
        output = functional.bind(self.key.weight, self.flocet_encoding(input))
        # Perform the superposition operation on the bound key-value pairs
        return functional.multibundle(output)
