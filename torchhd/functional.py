import math
import torch
from torch import BoolTensor, LongTensor, Tensor
import torch.nn.functional as F
from collections import deque


__all__ = [
    "identity_hv",
    "random_hv",
    "level_hv",
    "circular_hv",
    "bind",
    "unbind",
    "bundle",
    "permute",
    "cleanup",
    "hard_quantize",
    "soft_quantize",
    "hamming_similarity",
    "cosine_similarity",
    "dot_similarity",
    "multiset",
    "multibind",
    "cross_product",
    "bundle_sequence",
    "bind_sequence",
    "ngrams",
    "hash_table",
    "map_range",
    "value_to_index",
    "index_to_value",
]


def identity_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates a set of identity hypervector.

    When bound with a random-hypervector :math:`x`, the result is :math:`x`.

    Aliased as ``torchhd.identity_hv``.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> functional.identity_hv(3, 6)
        tensor([[1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.]])

        >>> functional.identity_hv(3, 6, dtype=torch.bool)
        tensor([[False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]])

        >>> functional.identity_hv(3, 6, dtype=torch.long)
        tensor([[1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]])

        >>> functional.identity_hv(3, 6, dtype=torch.complex64)
        tensor([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]])

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype in {torch.complex64, torch.complex128}:
        return torch.full(
            (num_embeddings, embedding_dim),
            1 + 0j,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    if dtype == torch.bool:
        return torch.zeros(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    return torch.ones(
        num_embeddings,
        embedding_dim,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def random_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    sparsity=0.5,
    generator=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates a set of random-hypervectors.

    The resulting hypervectors are sampled uniformly at random from the ``embedding_dim``-dimensional hyperspace.

    Aliased as ``torchhd.random_hv``.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        sparsity (float, optional): the expected fraction of elements to be in-active. Has no effect on complex hypervectors. Default: ``0.5``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> functional.random_hv(2, 5)
        tensor([[ 1., -1., -1.,  1., -1.,  1.],
                [-1.,  1.,  1.,  1.,  1.,  1.],
                [-1.,  1.,  1.,  1.,  1., -1.]])

        >>> functional.random_hv(2, 5, sparsity=0.9)
        tensor([[ 1.,  1.,  1., -1., -1.,  1.],
                [-1.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1., -1.,  1.,  1.]])

        >>> functional.random_hv(3, 6, dtype=torch.long)
        tensor([[ 1,  1,  1,  1,  1, -1],
                [ 1, -1,  1,  1, -1,  1],
                [ 1,  1, -1,  1,  1, -1]])

        >>> functional.random_hv(3, 6, dtype=torch.bool)
        tensor([[ True, False, False, False, False,  True],
                [ True,  True, False,  True,  True, False],
                [False, False, False,  True, False,  True]])

        >>> functional.random_hv(3, 6, dtype=torch.bool)
        tensor([[ True, False, False, False, False,  True],
                [ True,  True, False,  True,  True, False],
                [False, False, False,  True, False,  True]])

        >>> functional.random_hv(3, 6, dtype=torch.complex64)
        tensor([[-0.9849-0.1734j,  0.1267+0.9919j, -0.9160+0.4012j,  0.5063-0.8624j, 0.9898-0.1424j, -0.4378+0.8991j],
                [-0.4516+0.8922j,  0.7086-0.7056j,  0.8579+0.5138j,  0.9629-0.2699j, -0.2023+0.9793j, -0.9787-0.2052j],
                [-0.2974+0.9548j, -0.9936+0.1127j, -0.9740+0.2264j, -0.9999+0.0113j, 0.4434-0.8963j,  0.3888+0.9213j]])

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype in {torch.complex64, torch.complex128}:
        dtype = torch.float if dtype == torch.complex64 else torch.double

        angle = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        angle.uniform_(-math.pi, math.pi)
        magnitude = torch.ones(
            num_embeddings, embedding_dim, dtype=dtype, device=device
        )

        result = torch.polar(magnitude, angle)
        result.requires_grad = requires_grad
        return result

    select = torch.empty(
        (
            num_embeddings,
            embedding_dim,
        ),
        dtype=torch.bool,
    ).bernoulli_(1.0 - sparsity, generator=generator)

    if dtype == torch.bool:
        select.requires_grad = requires_grad
        return select

    result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
    result.requires_grad = requires_grad
    return result


def level_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    sparsity=0.5,
    randomness=0.0,
    generator=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates a set of level-hypervectors.

    Implements level-hypervectors as an interpolation between random-hypervectors as described in `An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing <https://arxiv.org/abs/2205.07920>`_.
    The first and last hypervector in the generated set are quasi-orthogonal.

    Aliased as ``torchhd.level_hv``.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        sparsity (float, optional): the expected fraction of elements to be in-active. Has no effect on complex hypervectors. Default: ``0.5``.
        randomness (float, optional): r-value to interpolate between level at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> functional.level_hv(5, 10)
        tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.],
                [ 1.,  1.,  1.,  1.,  1., -1., -1., -1.,  1.,  1.],
                [ 1.,  1.,  1., -1.,  1., -1.,  1., -1., -1.,  1.],
                [ 1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.],
                [ 1., -1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1.]])

        >>> functional.level_hv(5, 8, dtype=torch.bool)
        tensor([[ True, False, False,  True, False, False, False,  True],
                [ True,  True, False,  True,  True, False, False,  True],
                [ True,  True, False,  True,  True, False, False, False],
                [ True,  True, False,  True,  True, False,  True, False],
                [ True,  True, False,  True,  True, False,  True, False]])

        >>> functional.level_hv(5, 6, dtype=torch.complex64)
        tensor([[ 9.4413e-01+0.3296j, -9.5562e-01-0.2946j,  7.9306e-04+1.0000j, -8.8154e-01-0.4721j, -6.6328e-01+0.7484j, -8.6131e-01-0.5081j],
                [ 9.4413e-01+0.3296j, -9.5562e-01-0.2946j,  7.9306e-04+1.0000j, -8.8154e-01-0.4721j, -6.6328e-01+0.7484j, -8.6131e-01-0.5081j],
                [ 9.4413e-01+0.3296j, -9.5562e-01-0.2946j,  7.9306e-04+1.0000j, -8.8154e-01-0.4721j, -6.6328e-01+0.7484j, -8.6131e-01-0.5081j],
                [-9.9803e-01+0.0627j, -9.5562e-01-0.2946j,  7.9306e-04+1.0000j,  9.9992e-01+0.0126j, -6.6328e-01+0.7484j, -8.6131e-01-0.5081j],
                [-9.9803e-01+0.0627j, -8.5366e-01+0.5208j,  6.5232e-01-0.7579j,  9.9992e-01+0.0126j,  3.6519e-01+0.9309j,  9.7333e-01-0.2294j]])
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    # convert from normalized "randomness" variable r to number of orthogonal vectors sets "span"
    levels_per_span = (1 - randomness) * (num_embeddings - 1) + randomness * 1
    # must be at least one to deal with the case that num_embeddings is less than 2
    levels_per_span = max(levels_per_span, 1)
    span = (num_embeddings - 1) / levels_per_span

    hv = torch.empty(
        num_embeddings,
        embedding_dim,
        dtype=dtype,
        device=device,
    )

    # generate the set of orthogonal vectors within the level vector set
    span_hv = random_hv(
        int(math.ceil(span + 1)),
        embedding_dim,
        generator=generator,
        sparsity=sparsity,
        dtype=dtype,
        device=device,
    )
    # for each span within the set create a threshold vector
    # the threshold vector is used to interpolate between the
    # two random vector bounds of each span.
    threshold_v = torch.rand(
        int(math.ceil(span)),
        embedding_dim,
        generator=generator,
        dtype=torch.float,
        device=device,
    )

    for i in range(num_embeddings):
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
    return hv


def circular_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    sparsity=0.5,
    randomness=0.0,
    generator=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates a set of circular-hypervectors.

    Implements circular-hypervectors based on level-hypervectors as described in `An Extension to Basis-Hypervectors for Learning from Circular Data in Hyperdimensional Computing <https://arxiv.org/abs/2205.07920>`_.
    Any hypervector is quasi-orthogonal to the hypervector opposite site of the circle.

    Aliased as ``torchhd.circular_hv``.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        sparsity (float, optional): the expected fraction of elements to be in-active. Has no effect on complex hypervectors. Default: ``0.5``.
        randomness (float, optional): r-value to interpolate between circular at ``0.0`` and random-hypervectors at ``1.0``. Default: ``0.0``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> functional.circular_hv(8, 10)
        tensor([[-1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1., -1.],
                [-1.,  1., -1., -1., -1.,  1.,  1., -1., -1., -1.],
                [-1.,  1., -1., -1., -1.,  1.,  1., -1., -1., -1.],
                [-1.,  1.,  1., -1., -1.,  1.,  1., -1., -1., -1.],
                [ 1.,  1.,  1., -1., -1.,  1., -1., -1., -1., -1.],
                [ 1.,  1.,  1., -1.,  1.,  1., -1.,  1., -1., -1.],
                [ 1.,  1.,  1., -1.,  1.,  1., -1.,  1., -1., -1.],
                [ 1.,  1., -1., -1.,  1.,  1., -1.,  1., -1., -1.]])

        >>> functional.circular_hv(10, 8, dtype=torch.bool)
        tensor([[False,  True, False, False,  True, False,  True,  True],
                [False,  True, False, False,  True, False,  True,  True],
                [False,  True, False, False,  True, False,  True,  True],
                [False,  True, False, False,  True, False,  True, False],
                [False,  True, False, False, False, False, False, False],
                [ True,  True, False,  True, False, False, False, False],
                [ True,  True, False,  True, False, False, False, False],
                [ True,  True, False,  True, False, False, False, False],
                [ True,  True, False,  True, False, False, False,  True],
                [ True,  True, False,  True,  True, False,  True,  True]])

        >>> functional.circular_hv(10, 6, dtype=torch.complex64)
        tensor([[ 0.0691+0.9976j, -0.1847+0.9828j, -0.4434-0.8963j, -0.8287+0.5596j, -0.8357-0.5493j, -0.5358+0.8443j],
                [ 0.0691+0.9976j, -0.1847+0.9828j, -0.4434-0.8963j, -0.8287+0.5596j,  0.9427-0.3336j, -0.5358+0.8443j],
                [ 0.0691+0.9976j, -0.1847+0.9828j, -0.4434-0.8963j, -0.0339-0.9994j,  0.9427-0.3336j, -0.6510-0.7591j],
                [ 0.0691+0.9976j, -0.3881+0.9216j, -0.4434-0.8963j, -0.0339-0.9994j,  0.9427-0.3336j, -0.6510-0.7591j],
                [-0.6866-0.7271j, -0.3881+0.9216j, -0.4434-0.8963j, -0.0339-0.9994j,  0.9427-0.3336j, -0.6510-0.7591j],
                [-0.6866-0.7271j, -0.3881+0.9216j, -0.7324+0.6809j, -0.0339-0.9994j,  0.9427-0.3336j, -0.6510-0.7591j],
                [-0.6866-0.7271j, -0.3881+0.9216j, -0.7324+0.6809j, -0.0339-0.9994j, -0.8357-0.5493j, -0.6510-0.7591j],
                [-0.6866-0.7271j, -0.3881+0.9216j, -0.7324+0.6809j, -0.8287+0.5596j, -0.8357-0.5493j, -0.5358+0.8443j],
                [-0.6866-0.7271j, -0.1847+0.9828j, -0.7324+0.6809j, -0.8287+0.5596j, -0.8357-0.5493j, -0.5358+0.8443j],
                [ 0.0691+0.9976j, -0.1847+0.9828j, -0.7324+0.6809j, -0.8287+0.5596j, -0.8357-0.5493j, -0.5358+0.8443j]])

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    hv = torch.empty(
        num_embeddings,
        embedding_dim,
        dtype=dtype,
        device=device,
    )

    # convert from normalized "randomness" variable r to
    # number of levels between orthogonal pairs or "span"
    levels_per_span = ((1 - randomness) * (num_embeddings / 2) + randomness * 1) * 2
    span = num_embeddings / levels_per_span

    # generate the set of orthogonal vectors within the level vector set
    span_hv = random_hv(
        int(math.ceil(span + 1)),
        embedding_dim,
        generator=generator,
        sparsity=sparsity,
        dtype=dtype,
        device=device,
    )
    # for each span within the set create a threshold vector
    # the threshold vector is used to interpolate between the
    # two random vector bounds of each span.
    threshold_v = torch.rand(
        int(math.ceil(span)),
        embedding_dim,
        generator=generator,
        dtype=torch.float,
        device=device,
    )

    mutation_history = deque()

    # first vector is always a random vector
    hv[0] = span_hv[0]
    # mutation hypervector is the last generated vector while walking through the circle
    mutation_hv = span_hv[0]

    for i in range(1, num_embeddings + 1):
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

        mutation_history.append(unbind(temp_hv, mutation_hv))
        mutation_hv = temp_hv

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    for i in range(num_embeddings + 1, num_embeddings * 2 - 1):
        mut = mutation_history.popleft()
        mutation_hv = unbind(mutation_hv, mut)

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    hv.requires_grad = requires_grad
    return hv


def bind(input: Tensor, other: Tensor) -> Tensor:
    r"""Binds two hypervectors which produces a hypervector dissimilar to both.

    Binding is used to associate information, for instance, to assign values to variables.

    .. math::

        \otimes: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bind``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1., -1., -1.],
                [ 1.,  1.,  1.]])
        >>> functional.bind(x[0], x[1])
        tensor([ 1., -1., -1.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        return torch.logical_xor(input, other)

    return torch.mul(input, other)


def unbind(input: Tensor, other: Tensor) -> Tensor:
    r"""Inverse of the binding operation.

    See :func:`~torchhd.functional.bind`.

    Aliased as ``torchhd.unbind``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 6)
        >>> x
        tensor([[-1.,  1.,  1., -1., -1.,  1.],
                [-1., -1.,  1.,  1.,  1., -1.]])
        >>> functional.unbind(functional.bind(x[0], x[1]), x[1])
        tensor([-1.,  1.,  1., -1., -1.,  1.])

        >>> x = functional.random_hv(2, 6, dtype=torch.complex64)
        >>> x
        tensor([[-0.6510+0.7591j, -0.9675+0.2528j,  0.7358-0.6772j, -0.1791-0.9838j, -0.9874-0.1585j, -0.3726+0.9280j],
                [ 0.1429+0.9897j, -0.9173+0.3983j, -0.4906+0.8714j,  0.4710-0.8821j, 0.6478+0.7618j,  0.8753+0.4836j]])
        >>> functional.unbind(functional.bind(x[0], x[1]), x[1])
        tensor([-0.6510+0.7591j, -0.9675+0.2528j,  0.7358-0.6772j, -0.1791-0.9838j, -0.9874-0.1585j, -0.3726+0.9280j])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        return torch.logical_xor(input, other)

    if torch.is_complex(input):
        return torch.mul(input, other.conj())

    return torch.mul(input, other)


def bundle(input: Tensor, other: Tensor, *, tie: BoolTensor = None) -> Tensor:
    r"""Bundles two hypervectors which produces a hypervector maximally similar to both.

    The bundling operation is used to aggregate information into a single hypervector.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bundle``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector
        tie (BoolTensor, optional): specifies how to break a tie while bundling boolean hypervectors. Default: only set bit if both ``input`` and ``other`` are ``True``.

    Shapes:
        - Input: :math:`(*)`
        - Other: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.bundle(x[0], x[1])
        tensor([0., 2., 0.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        if tie is not None:
            return torch.where(input == other, input, tie)
        else:
            return torch.logical_and(input, other)

    return torch.add(input, other)


def permute(input: Tensor, *, shifts=1, dims=-1) -> Tensor:
    r"""Permutes hypervector by specified number of shifts.

    The permutation operator is used to assign an order to hypervectors.

    .. math::

        \Pi: \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.permute``.

    Args:
        input (Tensor): input hypervector
        shifts (int or tuple of ints, optional): The number of places by which the elements of the tensor are shifted. If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value.
        dims (int or tuple of ints, optional): axis along which to permute the hypervector. Default: ``-1``.

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(1, 3)
        >>> x
        tensor([ 1.,  -1.,  -1.])
        >>> functional.permute(x)
        tensor([ -1.,  1.,  -1.])

    """
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.roll(input, shifts=shifts, dims=dims)


def soft_quantize(input: Tensor):
    """Applies the hyperbolic tanh function to all elements of the input tensor.

    Args:
        input (Tensor): input tensor.

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> y = functional.bundle(x[0], x[1])
        >>> y
        tensor([0., 2., 0.])
        >>> functional.soft_quantize(y)
        tensor([0.0000, 0.9640, 0.0000])

    """
    return torch.tanh(input)


def hard_quantize(input: Tensor):
    """Applies binary quantization to all elements of the input tensor.

    Args:
        input (Tensor): input tensor

    Shapes:
        - Input: :math:`(*)`
        - Output: :math:`(*)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> y = functional.bundle(x[0], x[1])
        >>> y
        tensor([ 0., -2., -2.])
        >>> functional.hard_quantize(y)
        tensor([ 1., -1., -1.])

    """
    # Make sure that the output tensor has the same dtype and device
    # as the input tensor.
    positive = torch.tensor(1.0, dtype=input.dtype, device=input.device)
    negative = torch.tensor(-1.0, dtype=input.dtype, device=input.device)

    return torch.where(input > 0, positive, negative)


def dot_similarity(input: Tensor, others: Tensor) -> Tensor:
    """Dot product between the input vector and each vector in others.

    Aliased as ``torchhd.dot_similarity``.

    Args:
        input (Tensor): hypervectors to compare against others
        others (Tensor): hypervectors to compare with

    Shapes:
        - Input: :math:`(*, d)`
        - Others: :math:`(n, d)` or :math:`(d)`
        - Output: :math:`(*, n)` or :math:`(*)`, depends on shape of others

    .. note::

        Output ``dtype`` for ``torch.bool`` is ``torch.long``,
        for ``torch.complex64`` is ``torch.float``,
        for ``torch.complex128`` is ``torch.double``, otherwise same as input ``dtype``.

    Examples::

        >>> x = functional.random_hv(3, 6)
        >>> x
        tensor([[ 1., -1., -1.,  1., -1., -1.],
                [ 1., -1., -1., -1.,  1., -1.],
                [-1.,  1.,  1., -1.,  1., -1.]])
        >>> functional.dot_similarity(x, x)
        tensor([[ 6.,  2., -4.],
                [ 2.,  6.,  0.],
                [-4.,  0.,  6.]])

        >>> x = functional.random_hv(3, 6, dtype=torch.complex64)
        >>> x
        tensor([[ 0.5931+0.8051j, -0.7391+0.6736j, -0.9725+0.2328j, -0.9290+0.3701j, -0.8220+0.5696j,  0.9757-0.2190j],
                [-0.1053+0.9944j,  0.6918-0.7221j, -0.6242+0.7813j, -0.9580-0.2869j, 0.4799+0.8773j, -0.4127+0.9109j],
                [ 0.4230-0.9061j, -0.9658+0.2592j,  0.9961-0.0883j, -0.3829+0.9238j, -0.2551-0.9669j,  0.7827-0.6224j]])
        >>> functional.dot_similarity(x, x)
        tensor([[ 6.0000,  0.8164,  0.6771],
                [ 0.8164,  6.0000, -4.2506],
                [ 0.6771, -4.2506,  6.0000]])

    """
    if input.dtype == torch.bool:
        input_as_bipolar = torch.where(input, -1, 1)
        others_as_bipolar = torch.where(others, -1, 1)

        return F.linear(input_as_bipolar, others_as_bipolar)

    if torch.is_complex(input):
        return F.linear(input, others.conj()).real

    return F.linear(input, others)


def cosine_similarity(input: Tensor, others: Tensor, *, eps=1e-08) -> Tensor:
    """Cosine similarity between the input vector and each vector in others.

    Aliased as ``torchhd.cosine_similarity``.

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

        >>> x = functional.random_hv(3, 6)
        >>> x
        tensor([[-1.,  1.,  1., -1.,  1., -1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1., -1.,  1., -1.]])
        >>> functional.cosine_similarity(x, x)
        tensor([[1.0000, 0.0000, 0.6667],
                [0.0000, 1.0000, 0.3333],
                [0.6667, 0.3333, 1.0000]])

        >>> x = functional.random_hv(3, 6, dtype=torch.complex64)
        >>> x
        tensor([[-0.5578-0.8299j, -0.0043-1.0000j, -0.0181+0.9998j,  0.1107+0.9939j, -0.8215-0.5702j, -0.4585+0.8887j],
                [-0.7400-0.6726j,  0.6895-0.7243j, -0.8760+0.4823j, -0.4582-0.8889j, -0.6128+0.7903j, -0.4839-0.8751j],
                [-0.7839+0.6209j, -0.9239-0.3827j, -0.9961-0.0884j,  0.4614+0.8872j, -0.8546+0.5193j, -0.5468-0.8372j]])
        >>> functional.cosine_similarity(x, x)
        tensor([[1.0000, 0.1255, 0.1806],
                [0.1255, 1.0000, 0.2607],
                [0.1806, 0.2607, 1.0000]])

    """
    out_dtype = torch.get_default_dtype()

    # calculate vector magnitude
    if input.dtype == torch.bool:
        input_mag = torch.full(
            input.shape[:-1],
            math.sqrt(input.size(-1)),
            dtype=out_dtype,
            device=input.device,
        )
        others_mag = torch.full(
            others.shape[:-1],
            math.sqrt(others.size(-1)),
            dtype=out_dtype,
            device=others.device,
        )

    elif torch.is_complex(input):
        input_dot = torch.real(input * input.conj()).sum(dim=-1, dtype=out_dtype)
        input_mag = input_dot.sqrt()

        others_dot = torch.real(others * others.conj()).sum(dim=-1, dtype=out_dtype)
        others_mag = others_dot.sqrt()

    else:
        input_dot = torch.sum(input * input, dim=-1, dtype=out_dtype)
        input_mag = input_dot.sqrt()

        others_dot = torch.sum(others * others, dim=-1, dtype=out_dtype)
        others_mag = others_dot.sqrt()

    if input.dim() > 1:
        magnitude = input_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
    else:
        magnitude = input_mag * others_mag

    return dot_similarity(input, others).to(out_dtype) / (magnitude + eps)


def hamming_similarity(input: Tensor, others: Tensor) -> LongTensor:
    """Number of equal elements between the input vectors and each vector in others.

    Args:
        input (Tensor): hypervectors to compare against others
        others (Tensor): hypervectors to compare with

    Shapes:
        - Input: :math:`(*, d)`
        - Others: :math:`(n, d)` or :math:`(d)`
        - Output: :math:`(*, n)` or :math:`(*)`, depends on shape of others

    Examples::

        >>> x = functional.random_hv(3, 6)
        >>> x
        tensor([[ 1.,  1., -1., -1.,  1.,  1.],
                [ 1.,  1.,  1.,  1., -1., -1.],
                [ 1.,  1., -1., -1., -1.,  1.]])
        >>> functional.hamming_similarity(x, x)
        tensor([[6, 2, 5],
                [2, 6, 3],
                [5, 3, 6]])

    """
    if input.dim() > 1 and others.dim() > 1:
        return torch.sum(
            input.unsqueeze(-2) == others.unsqueeze(-3), dim=-1, dtype=torch.long
        )

    return torch.sum(input == others, dim=-1, dtype=torch.long)


def multiset(input: Tensor) -> Tensor:
    r"""Multiset of input hypervectors.

    Bundles all the input hypervectors together.

    Aliased as ``torchhd.functional.multibundle``.

    .. math::

        \bigoplus_{i=0}^{n-1} V_i

    Args:
        input (Tensor): input hypervector tensor

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(3, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.multiset(x)
        tensor([-1.,  3.,  1.])

    """
    dim = -2
    dtype = input.dtype

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        count = torch.sum(input, dim=dim, dtype=torch.long)
        threshold = input.size(dim) // 2
        return torch.greater(count, threshold)

    return torch.sum(input, dim=dim, dtype=dtype)


multibundle = multiset


def multibind(input: Tensor) -> Tensor:
    r"""Binding of multiple hypervectors.

    Binds all the input hypervectors together.

    .. math::

        \bigotimes_{i=0}^{n-1} V_i

    Args:
        input (Tensor): input hypervector tensor.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    .. note::

        This method is not supported for ``torch.float16`` and ``torch.bfloat16`` input data types on a CPU device.

    Examples::

        >>> x = functional.random_hv(3, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.multibind(x)
        tensor([ 1.,  1., -1.])

    """
    dtype = input.dtype
    dim = -2

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    if dtype == torch.bool:
        hvs = torch.unbind(input, dim)
        result = hvs[0]

        for i in range(1, len(hvs)):
            result = torch.logical_xor(result, hvs[i])

        return result

    return torch.prod(input, dim=dim, dtype=dtype)


def cross_product(input: Tensor, other: Tensor) -> Tensor:
    r"""Cross product between two sets of hypervectors.

    First creates a multiset from both tensors ``input`` (:math:`A`) and ``other`` (:math:`B`).
    Then binds those together to generate all cross products, i.e., :math:`A_1 * B_1 + A_1 * B_2 + \dots + A_1 * B_m + \dots + A_n * B_m`.

    .. math::

        \big( \bigoplus_{i=0}^{n-1} A_i \big) \otimes \big( \bigoplus_{i=0}^{m-1} B_i \big)

    Args:
        input (Tensor): first set of input hypervectors
        other (Tensor): second set of input hypervectors

    Shapes:
        - Input: :math:`(*, n, d)`
        - Other: :math:`(*, m, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> a = functional.random_hv(2, 3)
        >>> a
        tensor([[ 1.,  1., -1.],
                [ 1., -1.,  1.]])
        >>> b = functional.random_hv(5, 3)
        >>> b
        tensor([[ 1., -1.,  1.],
                [-1., -1., -1.],
                [-1., -1., -1.],
                [ 1.,  1., -1.],
                [ 1., -1., -1.]])
        >>> functional.cross_product(a, b)
        tensor([2., -0., -0.])

    """
    return bind(multiset(input), multiset(other))


def ngrams(input: Tensor, n: int = 3) -> Tensor:
    r"""Creates a hypervector with the :math:`n`-gram statistics of the input.

    .. math::

        \bigoplus_{i=0}^{m - n} \bigotimes_{j = 0}^{n - 1} \Pi^{n - j - 1}(V_{i + j})

    .. note::
        For :math:`n=1` use :func:`~torchhd.functional.multiset` instead and for :math:`n=m` use :func:`~torchhd.functional.bind_sequence` instead.

    Args:
        input (Tensor): The value hypervectors.
        n (int, optional): The size of each :math:`n`-gram, :math:`1 \leq n \leq m`. Default: ``3``.

    Shapes:
        - Input: :math:`(*, m, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(5, 3)
        >>> x
        tensor([[ 1., -1.,  1.],
                [-1., -1.,  1.],
                [ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1., -1., -1.]])
        >>> functional.ngrams(x)
        tensor([-1.,  1., -3.])

    """
    n_gram = permute(input[..., : -(n - 1), :], shifts=n - 1)
    for i in range(1, n):
        stop = None if i == (n - 1) else -(n - i - 1)
        sample = permute(input[..., i:stop, :], shifts=n - i - 1)
        n_gram = bind(n_gram, sample)

    return multiset(n_gram)


def hash_table(keys: Tensor, values: Tensor) -> Tensor:
    r"""Hash table from keys-values hypervector pairs.

    .. math::

        \bigoplus_{i = 0}^{m - 1} K_i \otimes V_i

    Args:
        keys (Tensor): The keys hypervectors, must be the same shape as values.
        values (Tensor): The values hypervectors, must be the same shape as keys.

    Shapes:
        - Keys: :math:`(*, n, d)`
        - Values: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> keys = functional.random_hv(2, 3)
        >>> keys
        tensor([[ 1., -1.,  1.],
                [ 1., -1.,  1.]])
        >>> values = functional.random_hv(2, 3)
        >>> values
        tensor([[-1., -1.,  1.],
                [ 1., -1., -1.]])
        >>> functional.hash_table(keys, values)
        tensor([0., 2., 0.])

    """
    return multiset(bind(keys, values))


def bundle_sequence(input: Tensor) -> Tensor:
    r"""Bundling-based sequence.

    The first value is permuted :math:`n-1` times, the last value is not permuted.

    .. math::

        \bigoplus_{i=0}^{m-1} \Pi^{m - i - 1}(V_i)

    Args:
        input (Tensor): The hypervector values.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(5, 3)
        >>> x
        tensor([[ 1., -1., -1.],
                [-1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [-1., -1., -1.],
                [ 1.,  1.,  1.]])
        >>> functional.bundle_sequence(x)
        tensor([-1.,  3.,  1.])

    """
    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multiset(permuted)


def bind_sequence(input: Tensor) -> Tensor:
    r"""Binding-based sequence.

    The first value is permuted :math:`n-1` times, the last value is not permuted.

    .. math::

        \bigotimes_{i=0}^{m-1} \Pi^{m - i - 1}(V_i)

    Args:
        input (Tensor): The hypervector values.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    .. note::

        This method is not supported for ``torch.float16`` and ``torch.bfloat16`` input data types on a CPU device.

    Examples::

        >>> x = functional.random_hv(5, 3)
        >>> x
        tensor([[-1.,  1., -1.],
                [-1., -1.,  1.],
                [ 1., -1., -1.],
                [ 1., -1., -1.],
                [-1., -1., -1.]])
        >>> functional.bind_sequence(x)
        tensor([-1.,  1.,  1.])

    """
    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multibind(permuted)


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


def cleanup(input: Tensor, memory: Tensor, threshold=0.0) -> Tensor:
    """Gets the most similar hypervector in memory.

    If the cosine similarity is less than threshold, raises a KeyError.

    Args:
        input (Tensor): The hypervector to cleanup.
        memory (Tensor): The hypervectors in memory.
        threshold (float, optional): minimal similarity between input and any hypervector in memory. Default: ``0.0``.

    Shapes:
        - Input: :math:`(d)`
        - Memory: :math:`(n, d)`
        - Output: :math:`(d)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1., -1., -1.],
                [ 1.,  1.,  1.]])
        >>> functional.cleanup(x[0], x)
        tensor([[ 1., -1., -1.]])

    """
    if input.dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if input.dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    scores = cosine_similarity(input.float(), memory.float())
    value, index = torch.max(scores, dim=-1)

    if value.item() < threshold:
        raise KeyError(
            "Hypervector with the highest similarity is less similar than the provided threshold"
        )

    return torch.index_select(memory, -2, index)
