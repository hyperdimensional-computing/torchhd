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
]


class Empty(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.empty_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
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

        self._fill_padding_idx_with_zero()

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

        self._fill_padding_idx_with_zero()

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Identity(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.identity_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
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

        self._fill_padding_idx_with_zero()

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

        self._fill_padding_idx_with_zero()

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Random(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.random_hv`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        model: (``Type[VSA_Model]``, optional): specifies the hypervector type to be instantiated. Default: ``torchhd.MAP``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
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

        self._fill_padding_idx_with_zero()

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

        self._fill_padding_idx_with_zero()

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).as_subclass(self.vsa_model)


class Level(nn.Module):
    """Embedding wrapper around :func:`~torchhd.level_hv`.

    Args:
        embedding_dim (int): the dimensionality of the hypervectors.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        num_ortho (int): the number of quasi-orthogonal hypervectors to generate over the span from ``low`` to ``high``. Default: ``2.0``
        randomness (float, optional): r-value to interpolate between level at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``

    For example, ``num_ortho=2.0`` means that the ``low`` and ``high`` values have exactly one random hypervector each and in between is an interpolation of those two.
    And with ``num_ortho=3.0`` there is one additional random hypervector that represents halfway between ``low`` and ``high``.

    Examples::

        >>> emb = embeddings.Level(5, 10, low=-1, high=2)
        >>> x = torch.FloatTensor([0.3, 1.9, -0.8])
        >>> emb(x)
        tensor([[-1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.],
                [-1.,  1., -1.,  1., -1., -1., -1.,  1.,  1., -1.],
                [ 1., -1.,  1., -1., -1., -1.,  1.,  1., -1.,  1.]])

    """

    __constants__ = [
        "num_ortho",
        "num_embeddings",
        "embedding_dim",
        "low",
        "high",
        "vsa_model",
    ]

    num_ortho: float
    num_embeddings: int
    embedding_dim: int
    low: float
    high: float
    vsa_model: Type[VSA_Model]
    threshold: Tensor
    weight: Tensor

    def __init__(
        self,
        embedding_dim: int,
        low: float = 0.0,
        high: float = 1.0,
        num_ortho: float = 2.0,
        vsa_model: Type[VSA_Model] = MAP,
        requires_grad: bool = False,
        device=None,
        dtype=None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super(Level, self).__init__()

        assert num_ortho > 1.0, "num_ortho must be more than 1.0"

        self.num_ortho = num_ortho
        self.num_embeddings = int(math.ceil(num_ortho))
        self.embedding_dim = embedding_dim
        self.vsa_model = vsa_model
        self.low = low
        self.high = high

        embeddings = functional.random_hv(
            self.num_embeddings, self.embedding_dim, vsa_model, **factory_kwargs
        )
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

        threshold = torch.rand(
            self.num_embeddings - 1,
            embedding_dim,
            dtype=torch.float,
            device=device,
        )
        self.register_buffer("threshold", threshold)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            self.threshold.uniform_()
            embeddings = functional.random_hv(
                self.num_embeddings,
                self.embedding_dim,
                self.vsa_model,
                **factory_kwargs
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape

        span = functional.map_range(
            input, self.low, self.high, 0, self.num_embeddings - 1
        )
        span = span.clamp(min=0, max=self.num_embeddings - 1)
        span_idx = span.floor().clamp_max(self.num_embeddings - 2)

        tau = (span - span_idx).ravel().unsqueeze(-1)
        span_idx = span_idx.long().ravel()

        span_start = self.weight.index_select(0, span_idx)
        span_end = self.weight.index_select(0, span_idx + 1)
        threshold = self.threshold.index_select(0, span_idx)

        hv = torch.where(threshold < tau, span_start, span_end)
        hv = hv.view(*shape, -1)
        return hv.as_subclass(self.vsa_model)


class Thermometer(nn.Embedding):
    """Embedding wrapper around :func:`~torchhd.thermometer_hv`.

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
    """Embedding wrapper around :func:`~torchhd.circular_hv`.

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
