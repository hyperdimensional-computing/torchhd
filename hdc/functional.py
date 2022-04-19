import math
import torch
import torch.nn.functional as F

from collections import deque


def identity_hv(
    num_embeddings: int,
    embedding_dim: int,
    *,
    out=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> torch.Tensor:
    """Creates a hypervector of all ones that when bound with x will result in x.
    Uses the bipolar system.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        out (torch.Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Identity hypervector

    """

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
) -> torch.Tensor:
    """Creates num random hypervectors of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (torch.Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Random Hypervector

    """

    selection = torch.randint(
        0,
        2,
        size=(num_embeddings * embedding_dim,),
        generator=generator,
        dtype=torch.long,
        requires_grad=requires_grad,
        device=device,
    )

    if out is not None:
        out = out.view(num_embeddings * embedding_dim)

    options = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    hv = torch.index_select(options, 0, selection, out=out)
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
) -> torch.Tensor:
    """Creates num random level correlated hypervectors of dim-dimensions in the bipolar system.
    Span denotes the number of approximate orthogonalities in the set (only 1 is an exact guarantee)

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): size of the embedded vector.
        randomness (float, optional): r-value to interpolate between level and random hypervectors. Defaults to 0.0.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (torch.Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Level hypervector

    """

    hv = torch.zeros(
        num_embeddings,
        embedding_dim,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
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
        dtype=dtype,
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
) -> torch.Tensor:
    """Creates num random circular level correlated hypervectors
    of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): size of the embedded vector
        randomness (float, optional): r-value. Defaults to 0.0.
        generator (torch.Generator, optional): specifies random number generator. Defaults to None.
        out (torch.Tensor, optional): specifies the output vector. (Optional) Defaults to None.
        dtype (torch.dtype, optional): specifies data type. Defaults to None.
        device (torch.device, optional): Defaults to None.
        requires_grad (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: circular hypervector

    """

    hv = torch.zeros(
        num_embeddings,
        embedding_dim,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
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
        dtype=dtype,
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

    return hv


def bind(input: torch.Tensor, other: torch.Tensor, *, out=None) -> torch.Tensor:
    """Combines two hypervectors a and b into a new hypervector in the
    same space, represents the vectors a and b as a pair

    Args:
        input (torch.Tensor): input hypervector tensor
        other (torch.Tensor): input hypervector tensor
        out (torch.Tensor, optional): output tensor. Defaults to None.

    Returns:
        torch.Tensor: binded hypervector

    """

    return torch.mul(input, other, out=out)


def bundle(input: torch.Tensor, other: torch.Tensor, *, out=None) -> torch.Tensor:
    """Returns element-wise sum of hypervectors input and other

    Args:
        input (torch.Tensor): input hypervector tensor
        other (torch.Tensor): input hypervector tensor
        out (torch.Tensor, optional): output tensor. Defaults to None.

    Returns:
        torch.Tensor: bundled hypervector

    """

    return torch.add(input, other, out=out)


def batch_bundle(
    input: torch.Tensor,
    *,
    dim=-2,
    keepdim=False,
    dtype=None,
) -> torch.Tensor:
    """Returns element-wise sum of hypervectors hv

    Args:
        input (torch.Tensor): input hypervector tensor
        dim (int, optional): dimension over which to bundle the hypervectors. Defaults to -2.
        keepdim (bool, optional): whether to keep the bundled dimension. Defaults to False.
        dtype (torch.dtype, optional): if specified determins the type of the returned tensor, otherwise same as input.

    Returns:
        torch.Tensor: bundled hypervector

    """

    return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)


def permute(input: torch.Tensor, *, shifts=1, dims=-1) -> torch.Tensor:
    """Permutes input hypervector by specified number of shifts

    Args:
        input (torch.Tensor): input tensor.
        shifts (int, optional): Number of places the elements of the hypervector are shifted. Defaults to 1.
        dims (int, optional): axis along which to permute the hypervector. Defaults to -1.

    Returns:
        torch.Tensor: permuted hypervector

    """

    return torch.roll(input, shifts=shifts, dims=dims)


def soft_quantize(input: torch.Tensor, *, out=None):
    """Applies the hyperbolic tanh function to all elements of the input tensor

    Args:
        input (torch.Tensor): input tensor.
        out (torch.Tensor, optional): output tensor. Defaults to None.

    """
    return torch.tanh(input, out=out)


def hard_quantize(input: torch.Tensor, *, out=None):
    """Clamps all elements in the input tensor into the range [-1, 1]

    Args:
        input (torch.Tensor): input tensor
        out (torch.Tensor, optional): output tensor. Defaults to None.

    Returns:
        torch.Tensor: clamped input vector
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


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Function returns the cosine similarity between two hypervectors

    Args:
        a (torch.Tensor): Input tensor 1.
        b (torch.Tensor): Input tensor 2.

    Returns:
        torch.Tensor: output vector

    """

    is_a_multi = len(a.shape) > 1
    is_b_multi = len(b.shape) > 1
    a = a if is_a_multi else a.unsqueeze(0)
    b = b if is_b_multi else b.unsqueeze(0)
    sim = F.cosine_similarity(a, b)
    sim = sim if is_a_multi or is_b_multi else sim[0]
    return sim


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Returns the dot product between two hypervectors

    Args:
        a (torch.Tensor): input tensor 1.
        b (torch.Tensor): input tensor 2.

    Returns:
        torch.Tensor: output tensor

    """

    is_a_multi = len(a.shape) > 1
    is_b_multi = len(b.shape) > 1
    a = a if is_a_multi else a.unsqueeze(0)
    b = b if is_b_multi else b.unsqueeze(0)
    sim = torch.dot(a, b)
    sim = sim if is_a_multi or is_b_multi else sim[0]
    return sim


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Returns the hamming distance between two hypervectors

    Args:
        a (torch.Tensor): input tensor 1.
        b (torch.Tensor): input tensor 2.

    Returns:
        torch.Tensor: output tensor

    """

    is_a_multi = len(a.shape) > 1
    is_b_multi = len(b.shape) > 1
    a = a if is_a_multi else a.unsqueeze(0)
    b = b if is_b_multi else b.unsqueeze(0)
    sim = (a.shape - torch.sum(a == b)) / 100
    sim = sim if is_a_multi or is_b_multi else sim[0]
    return sim
