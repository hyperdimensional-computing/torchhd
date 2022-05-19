import math
import torch
from torch import LongTensor, Tensor
import torch.nn.functional as F

from collections import deque


__all__ = [
    "identity_hv",
    "random_hv",
    "level_hv",
    "circular_hv",
    "bind",
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
    "sequence",
    "distinct_sequence",
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

        >>> functional.identity_hv(2, 3)
        tensor([[ 1.,  1.,  1.],
                [ 1.,  1.,  1.]])

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

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
        sparsity (float, optional): the expected fraction of elements to be +1. Default: ``0.5``.
        generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples::

        >>> functional.random_hv(2, 5)
        tensor([[-1.,  1., -1., -1.,  1.],
                [ 1., -1., -1., -1., -1.]])
        >>> functional.random_hv(2, 5, sparsity=0.9)
        tensor([[ 1.,  1.,  1., -1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.]])
        >>> functional.random_hv(2, 5, dtype=torch.long)
        tensor([[ 1, -1,  1,  1,  1],
                [ 1,  1, -1, -1,  1]])

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    select = torch.empty(
        (
            num_embeddings,
            embedding_dim,
        ),
        dtype=torch.bool,
    ).bernoulli_(1.0 - sparsity, generator=generator)
    result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
    result.requires_grad = requires_grad
    return result


def level_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
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

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    hv = torch.empty(
        num_embeddings,
        embedding_dim,
        dtype=dtype,
        device=device,
    )

    # convert from normalized "randomness" variable r to number of orthogonal vectors sets "span"
    levels_per_span = (1 - randomness) * (num_embeddings - 1) + randomness * 1
    span = (num_embeddings - 1) / levels_per_span
    # generate the set of orthogonal vectors within the level vector set
    span_hv = random_hv(
        int(math.ceil(span + 1)),
        embedding_dim,
        generator=generator,
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

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

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

        mutation_history.append(temp_hv * mutation_hv)
        mutation_hv = temp_hv

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    for i in range(num_embeddings + 1, num_embeddings * 2 - 1):
        mut = mutation_history.popleft()
        mutation_hv *= mut

        if i % 2 == 0:
            hv[i // 2] = mutation_hv

    hv.requires_grad = requires_grad
    return hv


def bind(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Binds two hypervectors which produces a hypervector dissimilar to both.

    Binding is used to associate information, for instance, to assign values to variables.

    .. math::

        \otimes: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bind``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector
        out (Tensor, optional): the output tensor.

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
    if input.dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if input.dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.mul(input, other, out=out)


def bundle(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Bundles two hypervectors which produces a hypervector maximally similar to both.

    The bundling operation is used to aggregate information into a single hypervector.

    .. math::

        \oplus: \mathcal{H} \times \mathcal{H} \to \mathcal{H}

    Aliased as ``torchhd.bundle``.

    Args:
        input (Tensor): input hypervector
        other (Tensor): other input hypervector
        out (Tensor, optional): the output tensor.

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
    if input.dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if input.dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.add(input, other, out=out)


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
    return torch.roll(input, shifts=shifts, dims=dims)


def soft_quantize(input: Tensor, *, out=None):
    """Applies the hyperbolic tanh function to all elements of the input tensor.

    Args:
        input (Tensor): input tensor.
        out (Tensor, optional): output tensor. Defaults to None.

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
    return torch.tanh(input, out=out)


def hard_quantize(input: Tensor, *, out=None):
    """Applies binary quantization to all elements of the input tensor.

    Args:
        input (Tensor): input tensor
        out (Tensor, optional): output tensor. Defaults to None.

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

    if out != None:
        out[:] = torch.where(input > 0, positive, negative)
        result = out
    else:
        result = torch.where(input > 0, positive, negative)

    return result


def cosine_similarity(input: Tensor, others: Tensor) -> Tensor:
    """Cosine similarity between the input vector and each vector in others.

    Args:
        input (Tensor): one-dimensional tensor
        others (Tensor): two-dimensional tensor

    Shapes:
        - Input: :math:`(d)`
        - Others: :math:`(n, d)`
        - Output: :math:`(n)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[-1., -1.,  1.],
                [ 1.,  1., -1.]])
        >>> functional.cosine_similarity(x[0], x)
        tensor([ 1., -1.])

    """
    return F.cosine_similarity(input, others)


def dot_similarity(input: Tensor, others: Tensor) -> Tensor:
    """Dot product between the input vector and each vector in others.

    Args:
        input (Tensor): one-dimensional tensor
        others (Tensor): two-dimensional tensor

    Shapes:
        - Input: :math:`(d)`
        - Others: :math:`(n, d)`
        - Output: :math:`(n)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1., -1.,  1.],
                [ 1.,  1.,  1.]])
        >>> functional.dot_similarity(x[0], x)
        tensor([3., 1.])

    """
    return F.linear(input, others)


def hamming_similarity(input: Tensor, others: Tensor) -> LongTensor:
    """Number of equal elements between the input vector and each vector in others.

    Args:
        input (Tensor): one-dimensional tensor
        others (Tensor): two-dimensional tensor

    Shapes:
        - Input: :math:`(d)`
        - Others: :math:`(n, d)`
        - Output: :math:`(n)`

    Examples::

        >>> x = functional.random_hv(2, 3)
        >>> x
        tensor([[ 1.,  1., -1.],
                [-1., -1., -1.]])
        >>> functional.hamming_similarity(x[0], x)
        tensor([3., 1.])

    """
    return torch.sum(input == others, dim=-1, dtype=torch.long)


def multiset(
    input: Tensor,
) -> Tensor:
    r"""Multiset of input hypervectors.

    Bundles all the input hypervectors together.

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

    if input.dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if input.dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.sum(input, dim=-2, dtype=input.dtype)


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

    Examples::

        >>> x = functional.random_hv(3, 3)
        >>> x
        tensor([[ 1.,  1.,  1.],
                [-1.,  1.,  1.],
                [-1.,  1., -1.]])
        >>> functional.multibind(x)
        tensor([ 1.,  1., -1.])

    """
    if input.dtype in {torch.bool, torch.complex64, torch.complex128}:
        raise NotImplementedError(
            "Boolean, and Complex hypervectors are not supported yet."
        )

    if input.dtype == torch.uint8:
        raise ValueError("Unsigned integer hypervectors are not supported.")

    return torch.prod(input, dim=-2, dtype=input.dtype)


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
        For :math:`n=1` use :func:`~torchhd.functional.multiset` instead and for :math:`n=m` use :func:`~torchhd.functional.distinct_sequence` instead.

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


def sequence(input: Tensor) -> Tensor:
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
        >>> functional.sequence(x)
        tensor([-1.,  3.,  1.])

    """
    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multiset(permuted)


def distinct_sequence(input: Tensor) -> Tensor:
    r"""Binding-based sequence.

    The first value is permuted :math:`n-1` times, the last value is not permuted.

    .. math::

        \bigotimes_{i=0}^{m-1} \Pi^{m - i - 1}(V_i)

    Args:
        input (Tensor): The hypervector values.

    Shapes:
        - Input: :math:`(*, n, d)`
        - Output: :math:`(*, d)`

    Examples::

        >>> x = functional.random_hv(5, 3)
        >>> x
        tensor([[-1.,  1., -1.],
                [-1., -1.,  1.],
                [ 1., -1., -1.],
                [ 1., -1., -1.],
                [-1., -1., -1.]])
        >>> functional.distinct_sequence(x)
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
