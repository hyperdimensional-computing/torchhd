import math
import torch
from torch import Tensor
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
    out=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates a hypervector of all ones that when bound with x will result in x.
    Uses the bipolar system.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        out (Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        Tensor: Identity hypervector

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    return torch.ones(
        num_embeddings,
        embedding_dim,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def random_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    generator=None,
    out=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates num random hypervectors of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        Tensor: Random Hypervector

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    selection = torch.randint(
        0,
        2,
        size=(num_embeddings * embedding_dim,),
        generator=generator,
        dtype=torch.long,
        device=device,
    )

    if out is not None:
        out = out.view(num_embeddings * embedding_dim)

    options = torch.tensor([1, -1], dtype=dtype, device=device)
    hv = torch.index_select(options, 0, selection, out=out)
    hv.requires_grad = requires_grad
    return hv.view(num_embeddings, embedding_dim)


def level_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    randomness=0.0,
    generator=None,
    out=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates num random level correlated hypervectors of dim-dimensions in the bipolar system.
    Span denotes the number of approximate orthogonalities in the set (only 1 is an exact guarantee)

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        randomness (float, optional): r-value to interpolate between level and random hypervectors. Defaults to 0.0.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        Tensor: Level hypervector

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    hv = torch.zeros(
        num_embeddings,
        embedding_dim,
        out=out,
        dtype=dtype,
        device=device,
    )

    # convert from normilzed "randomness" variable r to number of orthogonal vectors sets "span"
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
    # for each span within the set create a treshold vector
    # the treshold vector is used to interpolate between the
    # two random vector bounds of each span.
    treshold_v = torch.rand(
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
        # when treshold_v[span_idx], and span_hv[span_idx + 1] are not available.
        if abs(i % levels_per_span) < 1e-12:
            hv[i] = span_hv[span_idx]
        else:
            level_within_span = i % levels_per_span
            # the treshold value from the start hv's perspective
            t = 1 - (level_within_span / levels_per_span)

            span_start_hv = span_hv[span_idx]
            span_end_hv = span_hv[span_idx + 1]
            hv[i] = torch.where(treshold_v[span_idx] < t, span_start_hv, span_end_hv)

    hv.requires_grad = requires_grad
    return hv


def circular_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    randomness=0.0,
    generator=None,
    out=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """Creates num random circular level correlated hypervectors
    of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): size of the embedded vector
        randomness (float, optional): r-value. Defaults to 0.0.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        Tensor: circular hypervector

    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    hv = torch.zeros(
        num_embeddings,
        embedding_dim,
        out=out,
        dtype=dtype,
        device=device,
    )

    # convert from normilzed "randomness" variable r to
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
    # for each span within the set create a treshold vector
    # the treshold vector is used to interpolate between the
    # two random vector bounds of each span.
    treshold_v = torch.rand(
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
        # when treshold_v[span_idx], and span_hv[span_idx + 1] are not available.
        if abs(i % levels_per_span) < 1e-12:
            temp_hv = span_hv[span_idx]

        else:
            span_start_hv = span_hv[span_idx]
            span_end_hv = span_hv[span_idx + 1]

            level_within_span = i % levels_per_span
            # the treshold value from the start hv's perspective
            t = 1 - (level_within_span / levels_per_span)

            temp_hv = torch.where(treshold_v[span_idx] < t, span_start_hv, span_end_hv)

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
    """Combines two hypervectors a and b into a new hypervector in the
    same space, represents the vectors a and b as a pair

    Args:
        input (Tensor): input hypervector tensor
        other (Tensor): input hypervector tensor
        out (Tensor, optional): output tensor. Defaults to None.

    Returns:
        Tensor: bound hypervector

    """

    return torch.mul(input, other, out=out)


def bundle(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    """Returns element-wise sum of hypervectors input and other

    Args:
        input (Tensor): input hypervector tensor
        other (Tensor): input hypervector tensor
        out (Tensor, optional): output tensor. Defaults to None.

    Returns:
        Tensor: bundled hypervector

    """

    return torch.add(input, other, out=out)


def permute(input: Tensor, *, shifts=1, dims=-1) -> Tensor:
    """Permutes input hypervector by specified number of shifts

    Args:
        input (Tensor): input tensor.
        shifts (int, optional): Number of places the elements of the hypervector are shifted. Defaults to 1.
        dims (int, optional): axis along which to permute the hypervector. Defaults to -1.

    Returns:
        Tensor: permuted hypervector

    """

    return torch.roll(input, shifts=shifts, dims=dims)


def soft_quantize(input: Tensor, *, out=None):
    """Applies the hyperbolic tanh function to all elements of the input tensor

    Args:
        input (Tensor): input tensor.
        out (Tensor, optional): output tensor. Defaults to None.

    """
    return torch.tanh(input, out=out)


def hard_quantize(input: Tensor, *, out=None):
    """Clamps all elements in the input tensor into the range [-1, 1]

    Args:
        input (Tensor): input tensor
        out (Tensor, optional): output tensor. Defaults to None.

    Returns:
        Tensor: clamped input vector
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
    """Returns the cosine similarity between the input vector and each vector in others

    Args:
        input (Tensor): one-dimensional tensor (dim,)
        others (Tensor): two-dimensional tensor (num_vectors, dim)

    Returns:
        Tensor: output tensor of shape (num_vectors,)

    """
    return F.cosine_similarity(input, others)


def dot_similarity(input: Tensor, others: Tensor) -> Tensor:
    """Returns the dot product between the input vector and each vector in others

    Args:
        input (Tensor): one-dimensional tensor (dim,)
        others (Tensor): two-dimensional tensor (num_vectors, dim)

    Returns:
        Tensor: output tensor of shape (num_vectors,)

    """
    return F.linear(input, others)


def hamming_similarity(input: Tensor, others: Tensor) -> Tensor:
    """Returns the number of equal elements between the input vector and each vector in others

    Args:
        input (Tensor): one-dimensional tensor (dim,)
        others (Tensor): two-dimensional tensor (num_vectors, dim)

    Returns:
        Tensor: output tensor (num_vectors,)

    """
    return torch.sum(input == others, dim=-1, dtype=input.dtype)


def multiset(
    input: Tensor,
    *,
    dim=-2,
    keepdim=False,
    dtype=None,
    out=None,
) -> Tensor:
    """Element-wise sum of input hypervectors

    Args:
        input (Tensor): input hypervector tensor
        dim (int, optional): dimension over which to bundle the hypervectors. Defaults to -2.
        keepdim (bool, optional): whether to keep the bundled dimension. Defaults to False.
        dtype (torch.dtype, optional): if specified determins the type of the returned tensor, otherwise same as input.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: bundled hypervector
    """
    return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


def multibind(input: Tensor, *, dim=-2, keepdim=False, dtype=None, out=None) -> Tensor:
    """Element-wise multiplication of input hypervectors

    Args:
        input (Tensor): input hypervector tensor
        dim (int, optional): dimension over which to bind the hypervectors. Defaults to -2.
        keepdim (bool, optional): whether to keep the bundled dimension. Defaults to False.
        dtype (torch.dtype, optional): if specified determins the type of the returned tensor, otherwise same as input.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: bound hypervector
    """
    return torch.prod(input, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


def ngrams(input: Tensor, n=3) -> Tensor:
    """Creates a hypervector containing the n-gram statistics of input

    Arguments are of shape (*, n, d) where `*` is any dimensions including none, `n` is the
    number of values, and d is the dimensionality of the hypervector.

    Args:
        input (Tensor): The value hypervectors.
        n (int, optional): The size of each n-gram. Defaults to 3.

    Returns:
        Tensor: output hypervector of shape (*, d)
    """
    n_gram = None
    for i in range(0, n):
        if i == (n - 1):
            last_sample = None
        else:
            last_sample = -(n - i - 1)
        sample = permute(input[..., i:last_sample, :], shifts=n - i - 1)
        if n_gram is None:
            n_gram = sample
        else:
            n_gram = bind(n_gram, sample)
    return multiset(n_gram)


def hash_table(keys: Tensor, values: Tensor) -> Tensor:
    """Combines the keys and values hypervectors to create a hash table.

    Arguments are of shape (*, v, d) where `*` is any dimensions including none, `v` is the
    number of key-value pairs, and d is the dimensionality of the hypervector.

    Args:
        keys (Tensor): The keys hypervectors, must be the same shape as values.
        values (Tensor): The values hypervectors, must be the same shape as keys.

    Returns:
        Tensor: output hypervector of shape (*, d)
    """
    return multiset(bind(keys, values))


def sequence(input: Tensor) -> Tensor:
    """Creates a bundling-based sequence

    The first value is permuted n-1 times, the last value is permuted 0 times.

    Args:
        input (Tensor): The n hypervector values of shape (*, n, d).

    Returns:
        Tensor: output hypervector of shape (*, d)
    """
    dim = -2
    n = input.size(dim)

    enum = enumerate(torch.unbind(input, dim))
    permuted = [permute(hv, shifts=n - i - 1) for i, hv in enum]
    permuted = torch.stack(permuted, dim)

    return multiset(permuted)


def distinct_sequence(input: Tensor) -> Tensor:
    """Creates a binding-based sequence

    The first value is permuted n-1 times, the last value is permuted 0 times.

    Args:
        input (Tensor): The n hypervector values of shape (*, n, d).

    Returns:
        Tensor: output hypervector of shape (*, d)
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
    Input values outside the min, max range are not clamped.

    Args:
        input (torch.LongTensor): The values to map
        in_min (float): the minimum value of the input range
        in_max (float): the maximum value of the input range
        out_min (float): the minimum value of the output range
        out_max (float): the maximum value of the output range

    Returns:
        Tensor: output tensor

    """
    return out_min + (out_max - out_min) * (input - in_min) / (in_max - in_min)


def value_to_index(
    input: Tensor, in_min: float, in_max: float, index_length: int
) -> torch.LongTensor:
    """Maps the input real value range to an index range.
    Input values outside the min, max range are clamped.

    Args:
        input (torch.LongTensor): The values to map
        in_min (float): the minimum value of the input range
        in_max (float): the maximum value of the input range
        index_length (int): The length of the output index, i.e., one more than the maximum output

    Returns:
        Tensor: output tensor

    """
    mapped = map_range(input, in_min, in_max, 0, index_length - 1)
    return mapped.round().long().clamp(0, index_length - 1)


def index_to_value(
    input: torch.LongTensor, index_length: int, out_min: float, out_max: float
) -> torch.FloatTensor:
    """Maps the input index range to a real value range.
    Input values greater or equal to index_length are not clamped.

    Args:
        input (torch.LongTensor): The values to map
        index_length (int): The length of the input index, i.e., one more than the maximum index
        out_min (float): the minimum value of the output range
        out_max (float): the maximum value of the output range

    Returns:
        Tensor: output tensor

    """
    return map_range(input.float(), 0, index_length - 1, out_min, out_max)


def cleanup(input: Tensor, memory: Tensor, threshold=0.0) -> Tensor:
    """Returns a copy of the most similar hypervector in memory.

    If the cosine similarity is less than threshold, raises a KeyError.

    Args:
        input (Tensor): The hypervector to cleanup
        memory (Tensor): The `n` hypervectors in memory of shape (n, d)
        threshold (float, optional): minimal similarity between input and any hypervector in memory. Defaults to 0.0.

    Returns:
        Tensor: output tensor
    """
    scores = cosine_similarity(input, memory)
    value, index = torch.max(scores, dim=-1)

    if value.item() < threshold:
        raise KeyError(
            "Hypervector with the highest similarity is less similar than the provided threshold"
        )

    # Copying prevents manipulating the memory tensor
    return torch.clone(memory[index])
