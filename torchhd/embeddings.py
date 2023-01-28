import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import torchhd.functional as functional
from torchhd.base import VSA_Model
from torchhd.map import MAP

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
]


class Empty(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.empty_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSA_Model.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Empty(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAP([[0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0.]])

        >>> emb = embeddings.Empty(4, 6, torchhd.FHRR)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        FHRR([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
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
        self.vsa_model = vsa_model

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

        embeddings = functional.empty_hv(
            num_embeddings, embedding_dim, vsa_model, **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        # we don't need to set the padding to empty because it is already empty.

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.empty_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Identity(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.identity_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSA_Model.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Identity(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAP([[1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.]])

        >>> emb = embeddings.Identity(4, 6, torchhd.HRR)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        HRR([[1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
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
        self.vsa_model = vsa_model

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

        embeddings = functional.identity_hv(
            num_embeddings, embedding_dim, vsa_model, **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        self._fill_padding_idx_with_empty()

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.identity_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

        self._fill_padding_idx_with_empty()

    def _fill_padding_idx_with_empty(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        if self.padding_idx is not None:
            with torch.no_grad():
                empty = functional.empty_hv(
                    1, self.embedding_dim, self.vsa_model, **factory_kwargs
                )
                self.weight[self.padding_idx].copy_(empty.squeeze(0))

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Random(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.random_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``vsa_model``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Random(4, 6)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        MAP([[-1.,  1., -1.,  1., -1., -1.],
            [ 1., -1., -1., -1.,  1., -1.],
            [ 1., -1.,  1.,  1.,  1.,  1.]])

        >>> emb = embeddings.Random(4, 6, torchhd.BSC)
        >>> idx = torch.LongTensor([0, 1, 3])
        >>> emb(idx)
        BSC([[ True, False, False, False, False,  True],
            [False,  True,  True,  True, False,  True],
            [False, False,  True, False, False, False]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
        requires_grad: bool = False,
        padding_idx: Optional[int] = None,
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
        self.vsa_model = vsa_model

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

        embeddings = functional.random_hv(
            num_embeddings, embedding_dim, vsa_model, **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        self._fill_padding_idx_with_empty()

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.random_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

        self._fill_padding_idx_with_empty()

    def _fill_padding_idx_with_empty(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        if self.padding_idx is not None:
            with torch.no_grad():
                empty = functional.empty_hv(
                    1, self.embedding_dim, self.vsa_model, **factory_kwargs
                )
                self.weight[self.padding_idx].copy_(empty.squeeze(0))

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Level(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.level_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        randomness (float, optional): r-value to interpolate between level-hypervectors at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``vsa_model``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the closed bound.

    Examples::

        >>> emb = embeddings.Level(4, 6)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.6444, 0.9286, 0.9225, 0.3675])
        >>> emb(x)
        MAP([[ 1.,  1.,  1., -1.,  1.,  1.],
             [ 1.,  1., -1.,  1.,  1.,  1.],
             [ 1.,  1., -1.,  1.,  1.,  1.],
             [ 1.,  1.,  1., -1., -1.,  1.]])

        >>> emb = embeddings.Level(4, 6, torchhd.BSC)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.1825, 0.1541, 0.4435, 0.1512])
        >>> emb(x)
        BSC([[False,  True, False, False, False, False],
             [False,  True, False, False, False, False],
             [False,  True, False, False, False, False],
             [False,  True, False, False, False, False]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
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
    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
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
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa_model = vsa_model
        self.low = low
        self.high = high
        self.randomness = randomness

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.level_hv(
            num_embeddings,
            embedding_dim,
            vsa_model,
            randomness=randomness,
            **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.level_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                randomness=self.randomness,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        return super().forward(index).as_subclass(self.vsa_model)


class Thermometer(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.thermometer_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of VSA_Model.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the closed bound.

    Examples::

        >>> emb = embeddings.Thermometer(4, 6)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.5295, 0.0618, 0.0675, 0.1750])
        >>> emb(x)
        MAP([[ 1.,  1.,  1.,  1., -1., -1.],
             [-1., -1., -1., -1., -1., -1.],
             [-1., -1., -1., -1., -1., -1.],
             [ 1.,  1., -1., -1., -1., -1.]])

        >>> emb = embeddings.Thermometer(4, 6, torchhd.FHRR)
        >>> x = torch.rand(4)
        >>> x
        tensor([0.2668, 0.7668, 0.8083, 0.6247])
        >>> emb(x)
        FHRR([[ 1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
              [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
              [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
              [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
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
    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
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
        self.vsa_model = vsa_model
        self.low = low
        self.high = high

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.thermometer_hv(
            num_embeddings, embedding_dim, vsa_model, **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.thermometer_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        return super().forward(index).as_subclass(self.vsa_model)


class Circular(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.circular_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        phase (float, optional): The zero offset of the real number periodic interval that the circular levels represent. Default: ``0.0``
        period (float, optional): The period of the real number periodic interval that the circular levels represent. Default: ``2 * pi``
        randomness (float, optional): r-value to interpolate between circular-hypervectors at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``vsa_model``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> emb = embeddings.Circular(4, 6)
        >>> angle = torch.tensor([0.0, 3.141, 6.282, 9.423])
        >>> emb(angle)
        MAP([[-1., -1., -1., -1., -1.,  1.],
             [-1., -1.,  1.,  1., -1., -1.],
             [-1., -1., -1., -1., -1.,  1.],
             [-1., -1.,  1.,  1., -1., -1.]])

        >>> emb = embeddings.Circular(4, 6, torchhd.BSC)
        >>> angle = torch.tensor([0.0, 3.141, 6.282, 9.423])
        >>> emb(angle)
        BSC([[False,  True, False, False,  True,  True],
             [False, False, False, False, False,  True],
             [False,  True, False, False,  True,  True],
             [False, False, False, False, False,  True]])

    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "vsa_model",
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
    vsa_model: Type[VSA_Model]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vsa_model: Type[VSA_Model] = MAP,
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
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vsa_model = vsa_model
        self.phase = phase
        self.period = period
        self.randomness = randomness

        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        embeddings = functional.circular_hv(
            num_embeddings,
            embedding_dim,
            vsa_model,
            randomness=randomness,
            **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = functional.circular_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                randomness=self.randomness,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        mapped = functional.map_range(
            input, self.phase, self.period, 0, self.num_embeddings
        )
        index = mapped.round().long() % self.num_embeddings
        return super().forward(index).as_subclass(self.vsa_model)


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
    It computes :math:`\cos(x \Phi^{\mathsf{T}} + b) \odot \sin(x \Phi^{\mathsf{T}})` where :math:`\Phi \in \mathbb{R}^{d \times m}` is a matrix whose rows are uniformly sampled at random from the surface of an :math:`d`-dimensional unit sphere and :math:`b \in \mathbb{R}^{d}` is a vectors whose elements are sampled uniformly at random between 0 and :math:`2\pi`.

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
        self.weight.data.copy_(F.normalize(self.weight.data))
        nn.init.uniform_(self.bias, 0, 2 * math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = F.linear(input, self.weight)
        output = torch.cos(projected + self.bias) * torch.sin(projected)
        return output.as_subclass(MAP)


class Density(nn.Module):
    """Performs the transformation of input data into hypervectors according to the intRVFL model. 
    
    See details in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        in_features (int): the dimensionality of the input feature vector.
        out_features (int): the dimensionality of the hypervectors.
        vsa_model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        low (float, optional): The lower bound of the real number range that the levels of the thermometer encoding represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels of the thermometer encoding represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``vsa_model``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        vsa_model: Type[VSA_Model] = MAP,
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
        super(Density, self).__init__()

        # A set of random vectors used as unique IDs for features of the dataset.
        self.key = Random(in_features, out_features, vsa_model, **factory_kwargs)
        # Thermometer encoding used for transforming input data.
        self.density_encoding = Thermometer(
            out_features + 1,
            out_features,
            vsa_model,
            low=low,
            high=high,
            **factory_kwargs
        )

    # Specify the steps needed to perform the encoding
    def forward(self, input: Tensor) -> Tensor:
        # Perform binding of key and value vectors
        output = functional.bind(self.key.weight, self.density_encoding(input))
        # Perform the superposition operation on the bound key-value pairs
        return functional.multibundle(output)
