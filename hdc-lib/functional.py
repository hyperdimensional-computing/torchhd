import torch
import torch.nn.functional as F

from collections import deque


def random_hv(num: int, dim=None, dtype=None, device=None):
    """
    Creates num random hypervectors of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.
    """
    if dim is None:
        hv = torch.randint(0, 2, size=(num,), dtype=dtype, device=device)
    else:
        hv = torch.randint(0, 2, size=(num, dim), dtype=dtype, device=device)

    hv[hv == 0] = -1
    return hv


def level_hv(num: int, dim: int, dtype=None, device=None):
    """
    Creates num random level correlated hypervectors
    of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.
    """
    hv = torch.zeros((num, dim), dtype=dtype, device=device)

    mut = random_hv(dim, dtype=dtype, device=device)
    for i in range(num):
        hv[i] = mut
        idx = torch.randperm(dim)[: dim // num]
        mut[idx] *= -1

    return hv


def circular_hv(num: int, dim: int, dtype=None, device=None):
    """
    Creates num random circular level correlated hypervectors
    of dim dimensions in the bipolar system.
    When dim is None, creates one hypervector of num dimensions.
    """
    hv = torch.zeros((num, dim), dtype=dtype, device=device)

    mut = random_hv(dim, dtype=dtype, device=device)
    mutation_history = deque()
    # The number of computed mutations is always twice as much
    # as the needed mutations so handling an odd num parameter
    # is straightforward.
    for i in range(num):
        if i % 2 == 0:
            hv[i // 2] = mut
        idx = torch.randperm(dim)[: dim // num]
        mutation_history.append(idx)
        mut[idx] *= -1

    # replay the mutations on the second half of the circle
    for i in range(num, num * 2):
        if i % 2 == 0:
            hv[i // 2] = mut
        idx = mutation_history.popleft()
        mut[idx] *= -1

    return hv


def bind(a: torch.Tensor, b: torch.Tensor, out=None):
    """
    Combines two hypervectors a and b into a new hypervector in the 
    same space, represents the vectors a and b as a pair
    """
    return torch.mul(a, b, out=out)


def bundle(hvs: torch.Tensor):
    """
    Returns majority vote/element-wise sum of hypervectors hv
    """
    majority = torch.ones(hvs.shape[-1], dtype=hvs.dtype, device=hvs.device)
    is_less_than_zero = hvs.sum(dim=0) <= 0
    majority[is_less_than_zero] = -1
    return majority


def similarity(a: torch.Tensor, b: torch.Tensor):
    """
    Returns the similarity between two hypervectors
    """
    is_a_multi = len(a.shape) > 1
    is_b_multi = len(b.shape) > 1
    a = a if is_a_multi else a.unsqueeze(0)
    b = b if is_b_multi else b.unsqueeze(0)
    sim = F.cosine_similarity(a, b)
    sim = sim if is_a_multi or is_b_multi else sim[0]
    return sim
