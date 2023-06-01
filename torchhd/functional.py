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
from typing import Type, Callable, Optional
import math
import torch
from torch import LongTensor, FloatTensor, Tensor
from collections import deque

import torchhd
from torchhd.tensors.base import VSATensor
from torchhd.tensors.bsc import BSCTensor
from torchhd.tensors.map import MAPTensor
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.fhrr import FHRRTensor
from torchhd.types import VSAOptions


__all__ = [
    "ensure_vsa_tensor",
    "empty",
    "identity",
    "random",
    "level",
    "thermometer",
    "flocet",
    "circular",
    "bind",
    "bundle",
    "permute",
    "inverse",
    "negative",
    "cleanup",
    "create_random_permute",
    "hard_quantize",
    "soft_quantize",
    "hamming_similarity",
    "cosine_similarity",
    "cos",
    "dot_similarity",
    "dot",
    "multiset",
    "multibind",
    "cross_product",
    "bundle_sequence",
    "bind_sequence",
    "ngrams",
    "hash_table",
    "graph",
    "resonator",
    "ridge_regression",
    "map_range",
    "value_to_index",
    "index_to_value",
    "generic",
    "FractionalPowerEncoding",
]


def get_vsa_tensor_class(vsa: VSAOptions) -> Type[VSATensor]:
    if vsa == "BSC":
        return BSCTensor
    elif vsa == "MAP":
        return MAPTensor
    elif vsa == "HRR":
        return HRRTensor
    elif vsa == "FHRR":
        return FHRRTensor

    raise ValueError(f"Provided VSA model is not supported, specified: {vsa}")


def ensure_vsa_tensor(
    data,
    vsa: VSAOptions = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> VSATensor:
    """Converts data into a VSA model tensor.

    If data is already a VSA model of the correct model, dtype and device then data itself is returned.
    A copy of the data is created when dtype or device don't match using ``torch.as_tensor(data, dtype=dtype, device=device)``.

    When no model is specified boolean tensors are converted to Binary Spatter Codes, complex valued tensors to Fourier Holographic Reduced Representations and otherwise to the Multiply Add Permute VSA model.

    Args:
        data (array_like): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor.
        device (``torch.device``, optional): the desired device of returned tensor.

    Examples::

        >>> x = [True, False, False, True, False, False]
        >>> x = torchhd.ensure_vsa_tensor(x)
        >>> x
        tensor([ True, False, False,  True, False, False])
        >>> type(x)
        <class 'torchhd.tensors.bsc.BSCTensor'>

        >>> x = torch.rand(6)
        >>> x
        tensor([0.2083, 0.0665, 0.6302, 0.8650, 0.6618, 0.0886])
        >>> x = torchhd.ensure_vsa_tensor(x)
        >>> x
        tensor([0.2083, 0.0665, 0.6302, 0.8650, 0.6618, 0.0886])
        >>> type(x)
        <class 'torchhd.tensors.map.MAPTensor'>

    """
    input = torch.as_tensor(data, dtype=dtype, device=device)

    if vsa is not None:
        vsa_tensor = get_vsa_tensor_class(vsa)
        if input.dtype not in vsa_tensor.supported_dtypes:
            name = vsa_tensor.__name__
            options = ", ".join([str(x) for x in vsa_tensor.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        return input.as_subclass(vsa_tensor)

    if isinstance(input, VSATensor):
        return input

    if input.dtype == torch.bool:
        return input.as_subclass(BSCTensor)

    elif torch.is_complex(input):
        return input.as_subclass(FHRRTensor)

    else:
        return input.as_subclass(MAPTensor)


def empty(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    **kwargs,
) -> VSATensor:
    """Creates a set of hypervectors representing empty sets.

    When bundled with a random-hypervector :math:`x`, the result is :math:`x`.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.empty(3, 6, "BSC")
        tensor([[False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]])

        >>> torchhd.empty(3, 6, "MAP")
        tensor([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]])

        >>> torchhd.empty(3, 6, "FHRR")
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)
    return vsa_tensor.empty(num_vectors, dimensions, **kwargs)


def identity(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    **kwargs,
) -> VSATensor:
    """Creates a set of identity hypervectors.

    When bound with a random-hypervector :math:`x`, the result is :math:`x`.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.identity(3, 6, "BSC")
        tensor([[False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]])

        >>> torchhd.identity(3, 6, "MAP")
        tensor([[1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.]])

        >>> torchhd.identity(3, 6, "FHRR")
        tensor([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)
    return vsa_tensor.identity(num_vectors, dimensions, **kwargs)


def random(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    **kwargs,
) -> VSATensor:
    """Creates a set of random independent hypervectors.

    The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.random(3, 6, "BSC")
        tensor([[ True,  True,  True,  True,  True,  True],
                [False,  True, False, False,  True,  True],
                [ True,  True, False, False,  True,  True]])

        >>> torchhd.random(3, 6, "MAP")
        tensor([[ 1.,  1., -1.,  1., -1.,  1.],
                [ 1., -1.,  1., -1., -1., -1.],
                [ 1., -1.,  1.,  1.,  1., -1.]])

        >>> torchhd.random(3, 6, "FHRR")
        tensor([[-0.830-0.557j, -0.411+0.911j,  0.980-0.197j, -0.202+0.979j, -0.792+0.609j, -0.932-0.360j],
                [-0.977-0.212j,  0.191-0.981j,  0.340-0.940j,  0.902-0.431j,  0.141+0.990j, -0.661+0.749j],
                [-0.690+0.723j,  0.981-0.190j,  0.971+0.236j, -0.356-0.934j,  0.788-0.615j,  0.360-0.932j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)
    return vsa_tensor.random(num_vectors, dimensions, **kwargs)


def level(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    *,
    randomness: float = 0.0,
    requires_grad=False,
    **kwargs,
) -> VSATensor:
    """Creates a set of level correlated hypervectors.

    Implements level-hypervectors as an interpolation between random-hypervectors as described in `An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing <https://arxiv.org/abs/2205.07920>`_.
    The first and last hypervector in the generated set are quasi-orthogonal.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        randomness (float, optional): r-value to interpolate between level at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.level(5, 6, "BSC")
        tensor([[ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [False,  True,  True,  True,  True, False],
                [False,  True,  True,  True,  True, False],
                [False,  True,  True,  True,  True, False]])

        >>> torchhd.level(5, 6, "MAP")
        tensor([[ 1.,  1., -1.,  1., -1.,  1.],
                [ 1.,  1.,  1.,  1., -1.,  1.],
                [ 1.,  1.,  1.,  1., -1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.],
                [ 1., -1.,  1.,  1.,  1., -1.]])

        >>> torchhd.level(5, 6, "FHRR")
        tensor([[-0.996+0.079j,  0.447+0.894j, -0.840-0.541j, -0.999+0.020j, -0.742+0.669j, -0.999+0.042j],
                [-0.886-0.462j,  0.447+0.894j, -0.840-0.541j, -0.999+0.020j, -0.742+0.669j, -0.886+0.462j],
                [-0.886-0.462j,  0.447+0.894j, -0.146-0.989j, -0.999+0.020j, -0.350-0.936j, -0.886+0.462j],
                [-0.886-0.462j,  0.507+0.861j, -0.146-0.989j, -0.999+0.020j, -0.350-0.936j, -0.886+0.462j],
                [-0.886-0.462j,  0.507+0.861j, -0.146-0.989j, -0.611-0.791j, -0.350-0.936j, -0.886+0.462j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)

    # convert from normalized "randomness" variable r to number of orthogonal vectors sets "span"
    levels_per_span = (1 - randomness) * (num_vectors - 1) + randomness * 1
    # must be at least one to deal with the case that num_vectors is less than 2
    levels_per_span = max(levels_per_span, 1)
    span = (num_vectors - 1) / levels_per_span

    # generate the set of orthogonal vectors within the level vector set
    span_hv = vsa_tensor.random(
        int(math.ceil(span + 1)),
        dimensions,
        **kwargs,
    )

    # for each span within the set create a threshold vector
    # the threshold vector is used to interpolate between the
    # two random vector bounds of each span.
    threshold_v = torch.rand(
        int(math.ceil(span)),
        dimensions,
        dtype=torch.float,
        device=kwargs.get("device", None),
        generator=kwargs.get("generator", None),
    )

    hv = torch.empty(
        num_vectors,
        dimensions,
        dtype=span_hv.dtype,
        device=span_hv.device,
    )

    for i in range(num_vectors):
        span_idx = int(i // levels_per_span)

        # special case: if we are on a span border (e.g. on the first or last levels)
        # then set the orthogonal vector directly.
        # This also prevents an index out of bounds error for the last level
        # when threshold_v[span_idx], and span_hv[span_idx + 1] are not available.
        if abs(i % levels_per_span) < 1e-12:
            hv[i] = span_hv[span_idx]
        else:
            level_within_span = i % levels_per_span
            # the threshold value from the start hv's perspective
            t = 1 - (level_within_span / levels_per_span)

            span_start_hv = span_hv[span_idx]
            span_end_hv = span_hv[span_idx + 1]
            hv[i] = torch.where(threshold_v[span_idx] < t, span_start_hv, span_end_hv)

    hv.requires_grad = requires_grad
    return hv.as_subclass(vsa_tensor)


def thermometer(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    *,
    requires_grad=False,
    **kwargs,
) -> VSATensor:
    """Creates a thermometer code for given dimensionality.

    Implements similarity-preserving hypervectors as described in `Sparse Binary Distributed Encoding of Scalars <https://doi.org/10.1615/J%20Automat%20Inf%20Scien.v37.i6.20>`_.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.thermometer(7, 6, "BSC")
        tensor([[False, False, False, False, False, False],
                [ True, False, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True,  True]])

        >>> torchhd.thermometer(4, 6, "MAP")
        tensor([[-1., -1., -1., -1., -1., -1.],
                [ 1.,  1., -1., -1., -1., -1.],
                [ 1.,  1.,  1.,  1., -1., -1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.]])

        >>> torchhd.thermometer(6, 6, "FHRR")
        tensor([[-1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+-0.j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)

    # Check if the requested number of vectors can be accommodated
    if num_vectors > dimensions + 1:
        raise ValueError(
            f"For the given dimensionality: {dimensions}, the thermometer code cannot create more than {dimensions+1} hypervectors."
        )
    else:
        # Based on num_vectors and dimensions compute step between neighboring hypervectors
        step = 0
        if num_vectors > 1:
            step = (dimensions) // (num_vectors - 1)

    # generate a random vector as a placeholder to get dtype and device
    rand_hv = vsa_tensor.random(
        1,
        dimensions,
        **kwargs,
    )

    if vsa_tensor == BSCTensor:
        # Use binary vectors
        hv = torch.zeros(
            num_vectors,
            dimensions,
            dtype=rand_hv.dtype,
            device=rand_hv.device,
        )
    elif (vsa_tensor == MAPTensor) | (vsa_tensor == FHRRTensor):
        # Use bipolar vectors
        hv = torch.full(
            (
                num_vectors,
                dimensions,
            ),
            -1,
            dtype=rand_hv.dtype,
            device=rand_hv.device,
        )
    else:
        raise ValueError(f"{vsa_tensor} HD/VSA model is not defined.")

    # Create hypervectors using the obtained step
    for i in range(1, num_vectors):
        hv[i, 0 : i * step] = 1

    hv.requires_grad = requires_grad
    return hv.as_subclass(vsa_tensor)


def flocet(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    *,
    requires_grad=False,
    **kwargs,
) -> VSATensor:
    """Creates a thermometer code for given dimensionality.

    Implements similarity-preserving hypervectors as described in `Sparse Binary Distributed Encoding of Scalars <https://doi.org/10.1615/J%20Automat%20Inf%20Scien.v37.i6.20>`_.

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.flocet(7, 6, "BSC")
        tensor([[False, False, False, False, False, False],
                [ True, False, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True,  True]])

        >>> torchhd.flocet(4, 6, "MAP")
        tensor([[-1., -1., -1., -1., -1., -1.],
                [ 1.,  1., -1., -1., -1., -1.],
                [ 1.,  1.,  1.,  1., -1., -1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.]])

        >>> torchhd.flocet(6, 6, "FHRR")
        tensor([[-1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j, -1.+0.j],
                [ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, -1.+-0.j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)

    # generate a random vector as a placeholder to get dtype and device
    rand_hv = vsa_tensor.random(
        1,
        dimensions,
        **kwargs,
    )

    if vsa_tensor == BSCTensor:
        # Use binary vectors
        hv = torch.zeros(
            num_vectors,
            dimensions,
            dtype=rand_hv.dtype,
            device=rand_hv.device,
        )
    elif (vsa_tensor == MAPTensor) | (vsa_tensor == FHRRTensor):
        # Use bipolar vectors
        hv = torch.full(
            (
                num_vectors,
                dimensions,
            ),
            -1,
            dtype=rand_hv.dtype,
            device=rand_hv.device,
        )
    else:
        raise ValueError(f"{vsa_tensor} HD/VSA model is not defined.")

    # Create hypervectors using the obtained step
    for i in range(0, num_vectors):
        hv[i, i : i + int(dimensions / 2)] = 1

    hv.requires_grad = requires_grad
    return hv.as_subclass(vsa_tensor)


def circular(
    num_vectors: int,
    dimensions: int,
    vsa: VSAOptions = "MAP",
    *,
    randomness: float = 0.0,
    requires_grad=False,
    **kwargs,
) -> VSATensor:
    """Creates a set of circularly correlated hypervectors.

    Implements circular-hypervectors based on level-hypervectors as described in `An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing <https://arxiv.org/abs/2205.07920>`_.
    Any hypervector is quasi-orthogonal to the hypervector opposite site of the circle.

    .. note::
        Circular hypervectors cannot be created directly with Holographic Reduced Representations (HRR) because of imprecisions inherent to HRR.
        One way around this is to use FHRR for the creation of circular hypervectors and then transform them to HRR vectors. Example:

        >>> hv = torchhd.circular(10, 6, "FHRR")
        >>> hv = torch.real(torch.fft.ifft(hv)).as_subclass("HRR")

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        randomness (float, optional): r-value to interpolate between circular at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.circular(10, 6, "BSC")
        tensor([[False, False,  True, False,  True,  True],
                [False, False,  True, False,  True,  True],
                [False, False,  True, False,  True,  True],
                [False, False,  True,  True,  True,  True],
                [ True, False,  True,  True,  True,  True],
                [ True, False,  True,  True,  True,  True],
                [ True, False,  True,  True,  True,  True],
                [ True, False,  True,  True,  True,  True],
                [ True, False,  True, False,  True,  True],
                [False, False,  True, False,  True,  True]])

        >>> torchhd.circular(10, 6, "MAP")
        tensor([[-1., -1., -1., -1., -1.,  1.],
                [-1., -1., -1., -1., -1.,  1.],
                [-1., -1., -1.,  1., -1.,  1.],
                [-1., -1., -1.,  1., -1.,  1.],
                [-1., -1., -1.,  1., -1.,  1.],
                [-1., -1.,  1.,  1., -1.,  1.],
                [-1., -1.,  1.,  1., -1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.]])

        >>> torchhd.circular(10, 6, "FHRR")
        tensor([[-0.887-0.460j, -0.906+0.421j, -0.727-0.686j, -0.271+0.962j, -0.387+0.921j, -0.895-0.445j],
                [-0.887-0.460j, -0.906+0.421j, -0.727-0.686j, -0.947+0.319j, -0.387+0.921j, -0.895-0.445j],
                [-0.887-0.460j, -0.906+0.421j, -0.828+0.560j, -0.947+0.319j, -0.387+0.921j, -0.895-0.445j],
                [-0.887-0.460j, -0.906+0.421j, -0.828+0.560j, -0.947+0.319j, -0.387+0.921j, -0.895-0.445j],
                [ 0.983-0.183j,  0.732+0.680j, -0.828+0.560j, -0.947+0.319j, -0.387+0.921j, -0.895-0.445j],
                [ 0.983-0.183j,  0.732+0.680j, -0.828+0.560j, -0.947+0.319j, -0.705-0.709j,  0.562-0.827j],
                [ 0.983-0.183j,  0.732+0.680j, -0.828+0.560j, -0.271+0.962j, -0.705-0.709j,  0.562-0.827j],
                [ 0.983-0.183j,  0.732+0.680j, -0.727-0.686j, -0.271+0.962j, -0.705-0.709j,  0.562-0.827j],
                [ 0.983-0.183j,  0.732+0.680j, -0.727-0.686j, -0.271+0.962j, -0.705-0.709j,  0.562-0.827j],
                [-0.887-0.460j, -0.906+0.421j, -0.727-0.686j, -0.271+0.962j, -0.705-0.709j,  0.562-0.827j]])

    """
    vsa_tensor = get_vsa_tensor_class(vsa)

    if vsa_tensor == HRRTensor:
        raise ValueError(
            "The circular hypervectors don't currently work with the HRR model. We are not sure why, if you have any insight that could help please share it at: https://github.com/hyperdimensional-computing/torchhd/issues/108."
        )

    # convert from normalized "randomness" variable r to
    # number of levels between orthogonal pairs or "span"
    levels_per_span = ((1 - randomness) * (num_vectors / 2) + randomness * 1) * 2
    span = num_vectors / levels_per_span

    # generate the set of orthogonal vectors within the level vector set
    span_hv = vsa_tensor.random(
        int(math.ceil(span + 1)),
        dimensions,
        **kwargs,
    )
    # for each span within the set create a threshold vector
    # the threshold vector is used to interpolate between the
    # two random vector bounds of each span.
    threshold_v = torch.rand(
        int(math.ceil(span)),
        dimensions,
        dtype=torch.float,
        device=kwargs.get("device", None),
        generator=kwargs.get("generator", None),
    )

    hv = torch.empty(
        num_vectors,
        dimensions,
        dtype=span_hv.dtype,
        device=span_hv.device,
    )

    mutation_history = deque()

    # first vector is always a random vector
    hv[0] = span_hv[0]
    # mutation hypervector is the last generated vector while walking through the circle
    mutation_hv = span_hv[0]

    for i in range(1, num_vectors + 1):
        span_idx = int(i // levels_per_span)

        # special case: if we are on a span border (e.g. on the first or last levels)
        # then set the orthogonal vector directly.
        # This also prevents an index out of bounds error for the last level
        # when threshold_v[span_idx], and span_hv[span_idx + 1] are not available.
        if abs(i % levels_per_span) < 1e-12:
            temp_hv = span_hv[span_idx]

        else:
            span_start_hv = span_hv[span_idx]
            span_end_hv = span_hv[span_idx + 1]

            level_within_span = i % levels_per_span
            # the threshold value from the start hv's perspective
            t = 1 - (level_within_span / levels_per_span)

            temp_hv = torch.where(threshold_v[span_idx] < t, span_start_hv, span_end_hv)

        mutation_history.append(bind(temp_hv, inverse(mutation_hv)))
        mutation_hv = temp_hv

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    for i in range(num_vectors + 1, num_vectors * 2 - 1):
        mut = mutation_history.popleft()
        mutation_hv = bind(mutation_hv, inverse(mut))

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    hv.requires_grad = requires_grad
    return hv.as_subclass(vsa_tensor)


def bind(input: VSATensor, other: VSATensor) -> VSATensor:
    r"""Binds two hypervectors which produces a hypervector dissimilar to both.

    Binding is used to associate information, for instance, to assign values to variables.

    .. math::

        \otimes: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Args:
        input (VSATensor): input hypervector
        other (VSATensor): other input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a, b = torchhd.random(2, 10)
        >>> a
        tensor([ 1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.])
        >>> b
        tensor([-1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1., -1.])
        >>> torchhd.bind(a, b)
        tensor([-1., -1., -1., -1., -1.,  1., -1., -1.,  1.,  1.])

    """
    input = ensure_vsa_tensor(input)
    other = ensure_vsa_tensor(other)
    return input.bind(other)


def bundle(input: VSATensor, other: VSATensor) -> VSATensor:
    r"""Bundles two hypervectors which produces a hypervector maximally similar to both.

    The bundling operation is used to aggregate information into a single hypervector.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Args:
        input (VSATensor): input hypervector
        other (VSATensor): other input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a, b = torchhd.random(2, 10)
        >>> a
        tensor([-1., -1., -1., -1.,  1.,  1., -1., -1.,  1.,  1.])
        >>> b
        tensor([-1.,  1., -1., -1.,  1.,  1., -1.,  1., -1.,  1.])
        >>> torchhd.bundle(a, b)
        tensor([-2.,  0., -2., -2.,  2.,  2., -2.,  0.,  0.,  2.])

    """
    input = ensure_vsa_tensor(input)
    other = ensure_vsa_tensor(other)
    return input.bundle(other)


def permute(input: VSATensor, *, shifts=1) -> VSATensor:
    r"""Permutes hypervector by specified number of shifts.

    The permutation operator is used to assign an order to hypervectors.

    .. math::

        \Pi: \mathcal{H} \to \mathcal{H}

    Args:
        input (VSATensor): input hypervector
        shifts (int, optional): The number of places by which the elements of the tensor are shifted.

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a = torchhd.random(1, 10)
        >>> a
        tensor([[-1., -1., -1.,  1., -1., -1.,  1., -1., -1., -1.]])
        >>> torchhd.permute(a)
        tensor([[-1., -1., -1., -1.,  1., -1., -1.,  1., -1., -1.]])

    """
    input = ensure_vsa_tensor(input)
    return input.permute(shifts)


class create_random_permute(torch.nn.Module):
    r"""Creates random permutation functions.

    Args:
        dim (int): dimension of the hypervectors

    Examples::

        >>> a = torchhd.random(3, 10)
        >>> a
        tensor([[-1.,  1.,  1.,  1., -1., -1., -1., -1.,  1., -1.],
                [-1., -1., -1.,  1., -1.,  1., -1., -1.,  1., -1.],
                [ 1.,  1.,  1., -1., -1.,  1., -1.,  1.,  1.,  1.]])
        >>> p = torchhd.create_random_permute(10)
        >>> p(a, 2)
        tensor([[ 1.,  1., -1., -1., -1.,  1., -1., -1.,  1., -1.],
                [ 1., -1., -1., -1.,  1.,  1., -1., -1., -1., -1.],
                [ 1.,  1.,  1., -1.,  1., -1., -1.,  1.,  1.,  1.]])
        >>> p(a, -2)
        tensor([[-1.,  1.,  1.,  1., -1., -1., -1., -1.,  1., -1.],
                [-1., -1., -1.,  1., -1.,  1., -1., -1.,  1., -1.],
                [ 1.,  1.,  1., -1., -1.,  1., -1.,  1.,  1.,  1.]])

    """

    forward_indices: LongTensor
    backward_indices: LongTensor

    def __init__(self, dim: int) -> None:
        super().__init__()

        forward = torch.randperm(dim)
        backward = torch.empty_like(forward)
        backward[forward] = torch.arange(dim)

        self.register_buffer("forward_indices", forward)
        self.register_buffer("backward_indices", backward)

    def __call__(self, input: VSATensor, shifts: int = 1) -> VSATensor:
        y = input

        if shifts > 0:
            for _ in range(abs(shifts)):
                y = y[..., self.forward_indices]

        elif shifts < 0:
            for _ in range(abs(shifts)):
                y = y[..., self.backward_indices]

        return y.clone()


def inverse(input: VSATensor) -> VSATensor:
    r"""Inverse for the binding operation.

    See :func:`~torchhd.bind`.

    Args:
        input (VSATensor): input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a = torchhd.random(1, 6, "FHRR")
        >>> a
        tensor([[ 0.879-0.476j,  0.995-0.090j, -0.279+0.960j, -0.752-0.658j, -0.874+0.485j, -0.527-0.849j]])
        >>> torchhd.inverse(a)
        tensor([[ 0.879+0.476j,  0.995+0.090j, -0.279-0.960j, -0.752+0.658j, -0.874-0.485j, -0.527+0.849j]])

    """
    input = ensure_vsa_tensor(input)
    return input.inverse()


def negative(input: VSATensor) -> VSATensor:
    r"""Inverse for the bundling operation.

    See :func:`~torchhd.bundle`.

    Args:
        input (VSATensor): input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a = torchhd.random(1, 10)
        >>> a
        tensor([[ 1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1.,  1.]])
        >>> torchhd.negative(a)
        tensor([[-1., -1.,  1., -1., -1., -1., -1.,  1., -1., -1.]])

    """
    input = ensure_vsa_tensor(input)
    return input.negative()


def soft_quantize(input: Tensor):
    """Applies the hyperbolic tanh function to all elements of the input tensor.

    .. warning::
        This function does not take the VSA model class into account.

    Args:
        input (Tensor): input tensor.

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = torchhd.random(2, 6)
        >>> x
        tensor([[ 1.,  1., -1.,  1.,  1.,  1.],
            [ 1., -1., -1., -1.,  1., -1.]])
        >>> y = torchhd.bundle(x[0], x[1])
        >>> y
        tensor([ 2.,  0., -2.,  0.,  2.,  0.])
        >>> torchhd.soft_quantize(y)
        tensor([ 0.9640,  0.0000, -0.9640,  0.0000,  0.9640,  0.0000])

    """
    return torch.tanh(input)


def hard_quantize(input: Tensor):
    """Applies binary quantization to all elements of the input tensor.

    .. warning::
        This function does not take the VSA model class into account.

    Args:
        input (Tensor): input tensor

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = torchhd.random(2, 6)
        >>> x
        tensor([[ 1.,  1., -1.,  1.,  1.,  1.],
            [ 1., -1., -1., -1.,  1., -1.]])
        >>> y = torchhd.bundle(x[0], x[1])
        >>> y
        tensor([ 2.,  0., -2.,  0.,  2.,  0.])
        >>> torchhd.hard_quantize(y)
        tensor([ 1., -1., -1., -1.,  1., -1.])

    """
    # Make sure that the output tensor has the same dtype and device
    # as the input tensor.
    positive = torch.tensor(1.0, dtype=input.dtype, device=input.device)
    negative = torch.tensor(-1.0, dtype=input.dtype, device=input.device)

    return torch.where(input > 0, positive, negative)


def dot_similarity(input: VSATensor, others: VSATensor) -> VSATensor:
    """Dot product between the input vector and each vector in others.

    Aliased as ``torchhd.dot``.

    Args:
        input (VSATensor): hypervectors to compare against others
        others (VSATensor): hypervectors to compare with

    Shapes:
        - Input: :math:`(*, d)`
        - Others: :math:`(n, d)` or :math:`(d)`
        - Output: :math:`(*, n)` or :math:`(*)`, depends on shape of others

    .. note::

        Output ``dtype`` for ``torch.bool`` is ``torch.long``,
        for ``torch.complex64`` is ``torch.float``,
        for ``torch.complex128`` is ``torch.double``, otherwise same as input ``dtype``.

    Examples::

        >>> x = torchhd.random(3, 6)
        >>> x
        tensor([[ 1., -1.,  1.,  1.,  1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.],
                [-1.,  1.,  1., -1.,  1.,  1.]])
        >>> torchhd.dot_similarity(x, x)
        tensor([[6., 0., 0.],
                [0., 6., 2.],
                [0., 2., 6.]])

        >>> x = torchhd.random(3, 6, "FHRR")
        >>> x
        tensor([[-0.123-0.992j,  0.342-0.939j, -0.840-0.542j, -0.999+0.041j, -0.861-0.508j,  0.658-0.752j],
                [-0.754+0.656j,  0.574-0.818j, -0.449+0.893j, -0.705-0.708j,  0.652-0.757j,  0.444-0.895j],
                [ 0.805+0.593j, -0.647-0.762j, -0.192-0.981j, -0.796-0.605j, -0.380-0.924j, -0.556+0.830j]])
        >>> torchhd.dot_similarity(x, x)
        tensor([[ 6.0000,  1.7658,  1.0767],
                [ 1.7658,  6.0000, -0.3047],
                [ 1.0767, -0.3047,  6.0000]])

    """
    input = ensure_vsa_tensor(input)
    others = ensure_vsa_tensor(others)
    return input.dot_similarity(others)


dot = dot_similarity


def cosine_similarity(input: VSATensor, others: VSATensor) -> VSATensor:
    """Cosine similarity between the input vector and each vector in others.

    Aliased as ``torchhd.cos``.

    Args:
        input (Tensor): hypervectors to compare against others
        others (Tensor): hypervectors to compare with

    Shapes:
        - Input: :math:`(*, d)`
        - Others: :math:`(n, d)` or :math:`(d)`
        - Output: :math:`(*, n)` or :math:`(*)`, depends on shape of others

    .. note::

        Output ``dtype`` is ``torch.get_default_dtype()``.

    Examples::

        >>> x = torchhd.random(3, 6)
        >>> x
        tensor([[ 1., -1., -1., -1.,  1., -1.],
                [-1., -1.,  1., -1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1., -1.]])
        >>> torchhd.cosine_similarity(x, x)
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000, -0.3333],
                [ 0.0000, -0.3333,  1.0000]])

        >>> x = torchhd.random(3, 6, "FHRR")
        >>> x
        tensor([[ 0.986+0.166j,  0.886+0.463j,  0.205+0.978j,  0.952+0.304j,  0.923+0.384j, -0.529+0.848j],
                [-0.293+0.956j,  0.965+0.259j,  0.999-0.023j, -0.665-0.746j,  0.451-0.892j, -0.082+0.996j],
                [-0.991-0.127j, -0.326-0.945j,  0.785+0.618j,  0.518-0.855j,  0.149+0.988j,  0.020-0.999j]])
        >>> torchhd.cosine_similarity(x, x)
        tensor([[ 1.0000,  0.1884, -0.1779],
                [ 0.1884,  1.0000, -0.1900],
                [-0.1779, -0.1900,  1.0000]])

    """
    input = ensure_vsa_tensor(input)
    others = ensure_vsa_tensor(others)
    return input.cosine_similarity(others)


cos = cosine_similarity


def hamming_similarity(input: VSATensor, others: VSATensor) -> LongTensor:
    """Hamming similarity is the number of equal elements between the input vectors and each vector in others.

    Args:
        input (VSATensor): hypervectors to compare against others
        others (VSATensor): hypervectors to compare with

    Shapes:
        - Input: :math:`(*, d)`
        - Others: :math:`(n, d)` or :math:`(d)`
        - Output: :math:`(*, n)` or :math:`(*)`, depends on shape of others

    Examples::

        >>> x = torchhd.random(3, 6)
        >>> x
        tensor([[ 1.,  1., -1., -1.,  1.,  1.],
                [ 1.,  1.,  1.,  1., -1., -1.],
                [ 1.,  1., -1., -1., -1.,  1.]])
        >>> torchhd.hamming_similarity(x, x)
        tensor([[6, 2, 5],
                [2, 6, 3],
                [5, 3, 6]])

    """
    if input.dim() > 1 and others.dim() > 1:
        equals = input.unsqueeze(-2) == others.unsqueeze(-3)
        return torch.sum(equals, dim=-1, dtype=torch.long)

    return torch.sum(input == others, dim=-1, dtype=torch.long)


def multiset(input: VSATensor) -> VSATensor:
    r"""Multiset of input hypervectors.

    Bundles all the input hypervectors together.

    .. math::

        \bigoplus_{i=0}^{n-1} V_i

    Args:
        input (VSATensor): input hypervector tensor

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(3, 6)
        >>> x
        tensor([[-1., -1.,  1., -1.,  1., -1.],
                [-1.,  1., -1.,  1., -1.,  1.],
                [-1., -1.,  1., -1.,  1., -1.]])
        >>> torchhd.multiset(x)
        tensor([-3., -1.,  1., -1.,  1., -1.])

    """
    input = ensure_vsa_tensor(input)
    return input.multibundle()


multibundle = multiset


def randsel(
    input: VSATensor,
    other: VSATensor,
    *,
    p: float = 0.5,
    generator: torch.Generator = None,
) -> VSATensor:
    r"""Bundles two hypervectors by selecting random elements.

    A bundling operation is used to aggregate information into a single hypervector.
    The resulting hypervector has elements selected at random from input or other.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Args:
        input (VSATensor): input hypervector
        other (VSATensor): other input hypervector
        p (float, optional): probability of selecting elements from the input hypervector. Default: 0.5.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> a, b = torchhd.random(2, 6, "FHRR")
        >>> a
        tensor([-0.7404-0.6721j,  0.8280-0.5608j, -0.5059+0.8626j, -0.9965-0.0841j, -0.7337+0.6795j, -0.9925-0.1223j])
        >>> b
        tensor([-0.5593+0.8290j,  0.8097-0.5869j,  0.8306+0.5569j, -0.4970+0.8678j,  0.9962+0.0875j, -0.6631+0.7485j])
        >>> torchhd.randsel(a, b)
        tensor([-0.7404-0.6721j,  0.8280-0.5608j, -0.5059+0.8626j, -0.9965-0.0841j, -0.7337+0.6795j, -0.9925-0.1223j])

    """
    input = ensure_vsa_tensor(input)
    other = ensure_vsa_tensor(other)

    select = torch.empty_like(input, dtype=torch.bool)
    select.bernoulli_(1 - p, generator=generator)
    return input.where(select, other)


def multirandsel(
    input: VSATensor, *, p: FloatTensor = None, generator: torch.Generator = None
) -> VSATensor:
    r"""Bundling multiple hypervectors by sampling random elements.

    Bundles all the input hypervectors together.
    The resulting hypervector has elements selected at random from the input tensor of hypervectors.

    .. math::

        \bigoplus_{i=0}^{n-1} V_i

    Args:
        input (VSATensor): input hypervector tensor
        p (FloatTensor, optional): probability of selecting elements from the input hypervector. Default: uniform.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Probability (p): :math:`(*, n)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(4, 6, "FHRR")
        >>> x
        tensor([[-0.6344+0.7730j, -0.5673+0.8235j,  0.9051-0.4253j,  0.1355-0.9908j, -0.6559-0.7549j,  0.7526-0.6585j],
                [ 0.9136+0.4067j,  0.7351+0.6780j,  0.9999-0.0108j, -0.5853+0.8108j, -0.8442-0.5361j,  0.9487-0.3162j],
                [ 0.6320-0.7750j, -0.9836+0.1806j, -0.6542-0.7563j, -0.8747+0.4846j,  0.4030+0.9152j,  0.1324+0.9912j],
                [ 0.3632+0.9317j, -0.9414+0.3373j,  0.4078-0.9131j,  0.9815-0.1914j,  0.2741+0.9617j,  0.5697+0.8219j]])
        >>> torchhd.multirandsel(x)
        tensor([ 0.3632+0.9317j, -0.9836+0.1806j, -0.6542-0.7563j,  0.9815-0.1914j, -0.6559-0.7549j,  0.7526-0.6585j])

    """
    input = ensure_vsa_tensor(input)

    d = input.size(-1)
    device = input.device

    if p is None:
        p = torch.ones(input.shape[:-1], dtype=torch.float, device=device)

    select = torch.multinomial(p, d, replacement=True, generator=generator)
    select.unsqueeze_(-2)

    return input.gather(-2, select).squeeze(-2)


def multibind(input: VSATensor) -> VSATensor:
    r"""Binding of multiple hypervectors.

    Binds all the input hypervectors together.

    .. math::

        \bigotimes_{i=0}^{n-1} V_i

    Args:
        input (VSATensor): input hypervector tensor.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(3, 6)
        >>> x
        tensor([[ 1., -1.,  1., -1., -1., -1.],
                [-1., -1.,  1., -1., -1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.]])
        >>> torchhd.multibind(x)
        tensor([ 1., -1.,  1., -1., -1., -1.])

    """
    input = ensure_vsa_tensor(input)
    return input.multibind()


def cross_product(input: VSATensor, other: VSATensor) -> VSATensor:
    r"""Cross product between two sets of hypervectors.

    First creates a multiset from both tensors ``input`` (:math:`A`) and ``other`` (:math:`B`).
    Then binds those together to generate all cross products, i.e., :math:`A_1 * B_1 + A_1 * B_2 + \dots + A_1 * B_m + \dots + A_n * B_m`.

    .. math::

        \big( \bigoplus_{i=0}^{n-1} A_i \big) \otimes \big( \bigoplus_{i=0}^{m-1} B_i \big)

    Args:
        input (VSATensor): first set of input hypervectors
        other (VSATensor): second set of input hypervectors

    Shapes:
        - Input: :math:`(*, n, d)`
        - Other: :math:`(*, m, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> a = torchhd.random(2, 6)
        >>> a
        tensor([[ 1.,  1.,  1., -1.,  1.,  1.],
                [-1., -1.,  1., -1., -1.,  1.]])
        >>> b = torchhd.random(5, 6)
        >>> b
        tensor([[ 1., -1.,  1.,  1., -1., -1.],
                [-1.,  1.,  1., -1., -1.,  1.],
                [-1.,  1.,  1., -1., -1., -1.],
                [ 1., -1.,  1., -1., -1.,  1.],
                [ 1., -1.,  1.,  1., -1., -1.]])
        >>> torchhd.cross_product(a, b)
        tensor([ 0., -0., 10.,  2., -0., -2.])

    """
    input = ensure_vsa_tensor(input)
    other = ensure_vsa_tensor(other)
    return bind(multiset(input), multiset(other))


def ngrams(input: VSATensor, n: int = 3) -> VSATensor:
    r"""Creates a hypervector with the :math:`n`-gram statistics of the input.

    .. math::

        \bigoplus_{i=0}^{m - n} \bigotimes_{j = 0}^{n - 1} \Pi^{n - j - 1}(V_{i + j})

    .. note::
        For :math:`n=1` use :func:`~torchhd.multiset` instead and for :math:`n=m` use :func:`~torchhd.bind_sequence` instead.

    Args:
        input (VSATensor): The value hypervectors.
        n (int, optional): The size of each :math:`n`-gram, :math:`1 \leq n \leq m`. Default: ``3``.

    Shapes:
        - Input: :math:`(*, m, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(5, 6)
        >>> x
        tensor([[-1., -1., -1.,  1.,  1.,  1.],
                [ 1., -1.,  1.,  1.,  1.,  1.],
                [-1., -1.,  1.,  1., -1., -1.],
                [-1., -1.,  1.,  1., -1.,  1.],
                [ 1., -1.,  1.,  1., -1.,  1.]])
        >>> torchhd.ngrams(x)
        tensor([-1., -1.,  1., -3., -1., -3.])

    """
    input = ensure_vsa_tensor(input)

    n_gram = permute(input[..., : -(n - 1), :], shifts=n - 1)
    for i in range(1, n):
        stop = None if i == (n - 1) else -(n - i - 1)
        sample = permute(input[..., i:stop, :], shifts=n - i - 1)
        n_gram = bind(n_gram, sample)

    return multiset(n_gram)


def hash_table(keys: VSATensor, values: VSATensor) -> VSATensor:
    r"""Hash table from keys-values hypervector pairs.

    .. math::

        \bigoplus_{i = 0}^{n - 1} K_i \otimes V_i

    Args:
        keys (VSATensor): The keys hypervectors, must be the same shape as values.
        values (VSATensor): The values hypervectors, must be the same shape as keys.

    Shapes:
        - Keys: :math:`(*, n, d)`
        - Values: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> k = torchhd.random(2, 6)
        >>> k
        tensor([[-1., -1., -1.,  1.,  1.,  1.],
                [-1.,  1.,  1., -1., -1.,  1.]])
        >>> v = torchhd.random(2, 6)
        >>> v
        tensor([[-1.,  1.,  1.,  1., -1., -1.],
                [-1., -1.,  1., -1., -1., -1.]])
        >>> torchhd.hash_table(k, v)
        tensor([ 2., -2.,  0.,  2.,  0., -2.])

    """
    keys = ensure_vsa_tensor(keys)
    values = ensure_vsa_tensor(values)
    return multiset(bind(keys, values))


def bundle_sequence(input: VSATensor) -> VSATensor:
    r"""Bundling-based sequence.

    The first value is permuted :math:`n-1` times, the last value is not permuted.

    .. math::

        \bigoplus_{i=0}^{n-1} \Pi^{n - i - 1}(V_i)

    Args:
        input (VSATensor): The hypervector values.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(4, 6)
        >>> x
        tensor([[ 1., -1.,  1.,  1.,  1.,  1.],
                [-1.,  1., -1., -1.,  1., -1.],
                [ 1.,  1., -1., -1., -1.,  1.],
                [-1., -1.,  1., -1.,  1.,  1.]])
        >>> torchhd.bundle_sequence(x)
        tensor([ 2.,  0.,  2.,  0., -2.,  0.])

    """
    input = ensure_vsa_tensor(input)

    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multiset(permuted)


def bind_sequence(input: VSATensor) -> VSATensor:
    r"""Binding-based sequence.

    The first value is permuted :math:`n-1` times, the last value is not permuted.

    .. math::

        \bigotimes_{i=0}^{n-1} \Pi^{n - i - 1}(V_i)

    Args:
        input (VSATensor): The hypervector values.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(4, 6)
        >>> x
        tensor([[ 1.,  1.,  1., -1., -1.,  1.],
                [ 1.,  1., -1., -1., -1., -1.],
                [ 1., -1., -1.,  1.,  1.,  1.],
                [-1., -1.,  1.,  1., -1.,  1.]])
        >>> torchhd.bind_sequence(x)
        tensor([-1., -1., -1., -1.,  1., -1.])

    """
    input = ensure_vsa_tensor(input)

    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multibind(permuted)


def graph(input: VSATensor, *, directed=False) -> VSATensor:
    r"""Graph from node hypervector pairs.

    If ``directed=False`` this computes:

    .. math::

        \bigoplus_{i = 0}^{n - 1} V_{0,i} \otimes V_{1,i}

    If ``directed=True`` this computes:

    .. math::

        \bigoplus_{i = 0}^{n - 1} V_{0,i} \otimes \Pi(V_{1,i})

    Args:
        input (VSATensor): tensor containing pairs of node hypervectors that share an edge.
        directed (bool, optional): specify if the graph is directed or not. Default: ``False``.

    Shapes:
        - Input: :math:`(*, 2, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = torchhd.random(4, 6)
        >>> x
        tensor([[-1., -1.,  1.,  1.,  1., -1.],
                [-1., -1., -1.,  1.,  1.,  1.],
                [-1., -1.,  1., -1.,  1., -1.],
                [ 1., -1., -1., -1.,  1., -1.]])
        >>> edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
        >>> edges_hv = torch.index_select(x, 0, edges.ravel()).view(2, 4, 6)
        >>> edges_hv
        tensor([[[-1., -1.,  1.,  1.,  1., -1.],
                [-1., -1.,  1.,  1.,  1., -1.],
                [-1., -1., -1.,  1.,  1.,  1.],
                [-1., -1.,  1., -1.,  1., -1.]],

                [[-1., -1., -1.,  1.,  1.,  1.],
                [-1., -1.,  1., -1.,  1., -1.],
                [-1., -1.,  1., -1.,  1., -1.],
                [ 1., -1., -1., -1.,  1., -1.]]])
        >>> torchhd.graph(edges_hv)
        tensor([ 2.,  4., -2.,  0.,  4.,  0.])

    """
    input = ensure_vsa_tensor(input)

    to_nodes = input[..., 0, :, :]
    from_nodes = input[..., 1, :, :]

    if directed:
        from_nodes = permute(from_nodes)

    return multiset(bind(to_nodes, from_nodes))


def cleanup(input: VSATensor, memory: VSATensor, threshold=0.0) -> VSATensor:
    """Gets the most similar hypervector in memory.

    If the cosine similarity is less than threshold, raises a KeyError.

    Args:
        input (VSATensor): The hypervector to cleanup.
        memory (VSATensor): The hypervectors in memory.
        threshold (float, optional): minimal similarity between input and any hypervector in memory. Default: ``0.0``.

    Shapes:
        - Input: :math:`(d)`
        - Memory: :math:`(n, d)`
        - Output: :math:`(d)`

    Examples::

        >>> x = torchhd.random(4, 6)
        >>> x
        tensor([[-1.,  1.,  1., -1., -1., -1.],
                [ 1.,  1., -1.,  1., -1.,  1.],
                [-1.,  1., -1., -1.,  1.,  1.],
                [ 1., -1.,  1.,  1.,  1., -1.]])
        >>> torchhd.cleanup(x[0], x)
        tensor([[-1.,  1.,  1., -1., -1., -1.]])

    """
    input = ensure_vsa_tensor(input)

    scores = cosine_similarity(input, memory)
    value, index = torch.max(scores, dim=-1)

    if value.item() < threshold:
        raise KeyError(
            "Hypervector with the highest similarity is less similar than the provided threshold"
        )

    return torch.index_select(memory, -2, index)


def resonator(input: VSATensor, estimates: VSATensor, domains: VSATensor) -> VSATensor:
    """A step of the resonator network that factorizes the input.

    Given current estimates for each factor, it returns the next estimates for those factors.

    Args:
        input (VSATensor): The hypervector to be factorized.
        estimates (VSATensor): The current estimates of the factors, typically starts as a multiset of the domain.
        domains (VSATensor): The domains of each factor containing all possible factors.

    Shapes:
        - Input: :math:`(*, d)`
        - Estimates: :math:`(*, n, d)`
        - Domains: :math:`(*, n, m, d)`
        - Output: :math:`(*, n, d)`

    Examples::

        >>> X = torchhd.random(5, 100)
        >>> Y = torchhd.random(5, 100)
        >>> Z = torchhd.random(5, 100)
        >>> domains = torch.stack((X, Y, Z), dim=0)
        >>> domains.shape
        torch.Size([3, 5, 100])
        >>> x_hat = torchhd.multiset(X)
        >>> y_hat = torchhd.multiset(Y)
        >>> z_hat = torchhd.multiset(Z)
        >>> estimates = torch.stack((x_hat, y_hat, z_hat), dim=0)
        >>> estimates.shape
        torch.Size([3, 100])
        >>> # look at similarity of estimates with the domain
        >>> torchhd.dot_similarity(estimates.unsqueeze(-2), domains).squeeze(-2)
        MAPTensor([[112.,  80., 136., 106., 106.],
                   [ 98., 102., 100., 110.,  74.],
                   [116.,  94., 104., 112.,  82.]])
        >>> # Create the combined symbol
        >>> s = X[0].bind(Y[1]).bind(Z[3])
        >>> # resonator step
        >>> estimates = torchhd.resonator(s, estimates, domains)
        >>> # look at similarity of new estimates with the domain
        >>> torchhd.dot_similarity(estimates.unsqueeze(-2), domains).squeeze(-2)
        MAPTensor([[100.,   8.,   8.,  -2.,  -2.],
                   [-18.,  70.,  44.,  -6.,  14.],
                   [  8.,   2.,   4., 100.,  -2.]])
        >>> # resonator step
        >>> estimates = torchhd.resonator(s, estimates, domains)
        >>> # look at similarity of new estimates with the domain
        >>> torchhd.dot_similarity(estimates.unsqueeze(-2), domains).squeeze(-2)
        MAPTensor([[100.,   8.,   8.,  -2.,  -2.],
                   [  4., 100.,  14.,   0., -16.],
                   [  8.,   2.,   4., 100.,  -2.]])

    """
    input = ensure_vsa_tensor(input)
    estimates = ensure_vsa_tensor(estimates)
    domains = ensure_vsa_tensor(domains)

    if not isinstance(input, MAPTensor):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {input.__class__.__name__}"
        )

    if not isinstance(estimates, MAPTensor):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {estimates.__class__.__name__}"
        )

    if not isinstance(domains, MAPTensor):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {domains.__class__.__name__}"
        )

    n = estimates.size(-2)

    # Get binding inverse of the estimates
    inv_estimates = estimates.inverse()

    # Roll over the number of estimates to align each row with the other symbols
    # Example: for factorizing x, y, z the stacked matrix has the following estimates:
    # [[z, y],
    #  [x, z],
    #  [y, x]]
    rolled = []
    for i in range(1, n):
        rolled.append(inv_estimates.roll(i, -2))

    inv_estimates = torch.stack(rolled, dim=-2)

    # First bind all the other estimates together: z * y, x * z, y * z
    inv_others = multibind(inv_estimates)
    # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
    new_estimates = bind(input.unsqueeze(-2), inv_others)

    similarity = dot_similarity(new_estimates.unsqueeze(-2), domains)
    output = dot_similarity(similarity, domains.transpose(-2, -1)).squeeze(-2)

    # normalize the output vector with a non-linearity
    return output.sign()


def ridge_regression(
    samples: Tensor,
    labels: Tensor,
    alpha: Optional[float] = 1,
):
    """Compute weights (readout matrix) with ridge regression.

    It is a common way to form classifiers within randomized neural networks see, e.g., `Randomness in Neural Networks: An Overview  <https://doi.org/10.1002/widm.1200>`_.

    Args:
        samples (Tensor): The feature vectors.
        labels (Tensor): The target vectors, typically one-hot vectors for classification problems.
        alpha (float, optional): Scalar for the variance of the samples. Default is 1.

    Shapes:
       - Samples: :math:`(n, d)`
       - Labels: :math:`(n, c)`
       - Output: :math:`(c, d)`

    """

    variance = alpha * torch.diag(torch.var(samples, -2))

    return (
        labels.transpose(-2, -1)
        @ samples
        @ torch.linalg.pinv(samples.transpose(-2, -1) @ samples + variance)
    )


def map_range(
    input: Tensor,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> Tensor:
    """Maps the input real value range to an output real value range.

    .. note::

        Input values outside the min-max range are not clamped.

    Args:
        input (Tensor): The values to map
        in_min (float): the minimum value of the input range
        in_max (float): the maximum value of the input range
        out_min (float): the minimum value of the output range
        out_max (float): the maximum value of the output range

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = torch.rand(2, 3)
        >>> x
        tensor([[0.2211, 0.1291, 0.3081],
                [0.7654, 0.2155, 0.4381]])
        >>> functional.map_range(x, 0, 1, -10, 10)
        tensor([[-5.5781, -7.4176, -3.8374],
                [ 5.3082, -5.6906, -1.2383]])

    """
    if not torch.is_floating_point(input):
        raise ValueError("map_range only supports floating point tensors.")

    return out_min + (out_max - out_min) * (input - in_min) / (in_max - in_min)


def value_to_index(
    input: Tensor, in_min: float, in_max: float, index_length: int
) -> torch.LongTensor:
    """Maps the input real value range to an index range.

    .. note::

        Input values outside the min-max range are not clamped.

    Args:
        input (torch.LongTensor): The values to map
        in_min (float): the minimum value of the input range
        in_max (float): the maximum value of the input range
        index_length (int): The length of the output index, i.e., one more than the maximum output

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = torch.rand(2, 3)
        >>> x
        tensor([[0.2211, 0.1291, 0.3081],
                [0.7654, 0.2155, 0.4381]])
        >>> functional.value_to_index(x, 0, 1, 10)
        tensor([[2, 1, 3],
                [7, 2, 4]])

    """
    if torch.is_complex(input):
        raise ValueError("value_to_index does not support complex numbers")

    mapped = map_range(input.float(), in_min, in_max, 0, index_length - 1)
    return mapped.round().long()


def index_to_value(
    input: torch.LongTensor, index_length: int, out_min: float, out_max: float
) -> torch.FloatTensor:
    """Maps the input index range to a real value range.

    .. note::

        Input values greater or equal to ``index_length`` are not clamped.

    Args:
        input (torch.LongTensor): The values to map
        index_length (int): The length of the input index, i.e., one more than the maximum index
        out_min (float): the minimum value of the output range
        out_max (float): the maximum value of the output range

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = torch.randint(0, 10, (2, 3))
        >>> x
        tensor([[3, 0, 3],
                [2, 5, 5]])
        >>> functional.index_to_value(x, 10, 0, 1)
        tensor([[0.3333, 0.0000, 0.3333],
                [0.2222, 0.5556, 0.5556]])

    """
    return map_range(input.float(), 0, index_length - 1, out_min, out_max)


def generic(id, level, ngram):
    levels = [
        torchhd.ngrams(level[0][i : i + ngram], ngram)
        for i in range(1, id.shape[0] - ngram + 1)
    ]
    if len(levels) > 0:
        levels = torch.stack(levels)
        return torchhd.multiset(torchhd.bind(id[:-ngram], levels)).unsqueeze(0)
    return torchhd.multiset(torchhd.bind(id, level))


class FractionalPowerEncoding:
    """Class for fractional power encoding (FPE) method that forms hypervectors for given values, kernel shape, bandwidth, and dimensionality. Implements similarity-preserving hypervectors approximating desired kernel shape as described in `Computing on Functions Using Randomized Vector Representations <https://arxiv.org/abs/2109.03429>`_.

    Args:
        dimensions (int): the dimensionality of the hypervectors.
        data_dimensions (int): the dimensionality of input data
        kernel_shape (str, optional): hyperparameter defining the shape of the kernel by specifying a particular probability distribution that is used to sample the base hypervector(s).  Default: ``"sinc"``.
        bandwidth (float, optional): positive hyperparameter defining the width of the similarity kernel. Lower values lead to broader kernels while larger values lead to more narrow kernels. Default: ``1.0``.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"FHRR"``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> torchhd.FractionalPowerEncoding(6, 1, "sinc", 1.0, "FHRR").encoding(torch.arange(1, 4, 1.))
        FHRRTensor([[-0.7181-0.6959j, -0.5269+0.8499j, -0.0848+0.9964j,  0.9720-0.2348j,
              0.6358+0.7718j,  0.4352+0.9003j],
            [ 0.0314+0.9995j, -0.4447-0.8957j, -0.9856-0.1689j,  0.8897-0.4565j,
             -0.1915+0.9815j, -0.6212+0.7836j],
            [ 0.6730-0.7396j,  0.9956+0.0940j,  0.2519-0.9678j,  0.7576-0.6527j,
             -0.8793+0.4762j, -0.9759-0.2183j]])

    """

    def __init__(
        self,
        dimensions: int,
        data_dimensions: int,
        kernel_shape: Optional[Callable] = "sinc",
        bandwidth: float = 1.0,
        vsa: VSAOptions = "FHRR",
        requires_grad: bool = False,
        device=None,
    ):
        self.dimensions = dimensions
        self.data_dimensions = data_dimensions
        self.kernel_shape = kernel_shape
        self.bandwidth = bandwidth
        self.requires_grad = requires_grad
        self.vsa_tensor = get_vsa_tensor_class(vsa)

        super().__init__()

        self.base_hv, self.angle = self.generate_base()

    def generate_base(self):
        """Generate the basis hypervector(s) to be used for encoding the data."""

        # Set the values of the base hypervector(s).
        base_hv = self.vsa_tensor.empty(
            self.data_dimensions,
            self.dimensions,
        )

        # Check HD/VSA model type
        if self.vsa_tensor == FHRRTensor:
            # Generate the base vector that determines the shape of the FPE kernel
            if self.kernel_shape == "sinc":
                # Define the corresppnding distribution
                self.kernel_dist = torch.distributions.Uniform(
                    torch.tensor([-math.pi]), torch.tensor([math.pi])
                )

                # Draw angles from a uniform  distribution for base hypervector(s). Note that data dimensions here are independent but this does not have to be always the case
                angle = torch.reshape(
                    self.kernel_dist.sample(
                        sample_shape=torch.Size(
                            [self.data_dimensions * self.dimensions]
                        )
                    ),
                    (self.data_dimensions, self.dimensions),
                )

            elif self.kernel_shape == "Gaussian":
                # Define the corresppnding distribution
                self.kernel_dist = torch.distributions.Normal(
                    torch.tensor([0.0]), torch.tensor([1.0])
                )

                # Draw angles from a uniform  distribution for base hypervector(s). Note that data dimensions here are independent but this does not have to be always the case
                angle = torch.reshape(
                    self.kernel_dist.sample(
                        sample_shape=torch.Size(
                            [self.data_dimensions * self.dimensions]
                        )
                    ),
                    (self.data_dimensions, self.dimensions),
                )

            else:
                raise ValueError(
                    f"{self.kernel_shape} kernel is not supported at the moment."
                )

            # Set the values of the base hypervector(s).
            base_hv[:, :] = torch.complex(angle.cos(), angle.sin())

        else:
            raise ValueError(
                f"{self.vsa_tensor} Fractioncal Power Encoding for this HD/VSA model is not implemented or defined."
            )

        return base_hv, angle

    def encoding(self, values, device=None):
        """Creates a fractional power encoding (FPE) for given values.

        Args:
            values (Tensor): values for which FPE hypervectors should be generated.

        """

        hv = self.vsa_tensor.empty(
            values.size()[0],
            self.dimensions,
        )

        # Check if input data is uni or multi-dimensional to smoothly support both
        try:
            torch.any(values, dim=1)
        except:
            # In case the input is one-dimensional, add an extra singleton dimension
            values = torch.unsqueeze(values, 1)

        if self.vsa_tensor == FHRRTensor:
            # Perform FPE of the desired values using the base hypervector(s)
            # Simultaneously computes angles for given values and their sum that is equivalent to the binding
            hv_angles = torch.matmul(
                self.bandwidth * values.to(device), self.angle.to(device)
            )
            hv[:, :] = torch.complex(hv_angles.cos(), hv_angles.sin())

        else:
            raise ValueError(
                f"{self.vsa_tensor} Fractioncal Power Encoding for this HD/VSA model is not implemented or defined."
            )

        hv.requires_grad = self.requires_grad
        return hv
