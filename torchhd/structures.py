from typing import Any, List, Optional, Tuple, overload
import torch
from torch import Tensor

import torchhd.functional as functional


class Memory:
    """Associative memory"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.keys: List[Tensor] = []
        self.values: List[Any] = []

    def __len__(self) -> int:
        """Returns the number of items in memory"""
        return len(self.values)

    def add(self, key: Tensor, value: Any) -> None:
        """Adds one (key, value) pair to memory"""
        self.keys.append(key)
        self.values.append(value)

    def _get_index(self, key: Tensor) -> int:
        key_stack = torch.stack(self.keys, dim=0)
        sim = functional.cosine_similarity(key, key_stack)
        value, index = torch.max(sim, 0)

        if value.item() < self.threshold:
            raise IndexError()

        return index

    def __getitem__(self, key: Tensor) -> Tuple[Tensor, Any]:
        """Get the (key, value) pair with an approximate key"""
        index = self._get_index(key)
        return self.keys[index], self.values[index]

    def __setitem__(self, key: Tensor, value: Any) -> None:
        """Set the value of an (key, value) pair with an approximate key"""
        index = self._get_index(key)
        self.values[index] = value

    def __delitem__(self, key: Tensor) -> None:
        """Delete the (key, value) pair with an approximate key"""
        index = self._get_index(key)
        del self.keys[index]
        del self.values[index]


class Multiset:
    @overload
    def __init__(self, dimensions: int, *, device=None, dtype=None):
        ...

    @overload
    def __init__(self, input: Tensor, *, size=0):
        ...

    def __init__(self, dim_or_input: int, **kwargs):
        self.size = kwargs.get("size", 0)
        if torch.is_tensor(dim_or_input):
            self.value = dim_or_input
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            device = kwargs.get("device", None)
            self.value = torch.zeros(dim_or_input, dtype=dtype, device=device)

    def add(self, input: Tensor) -> None:
        self.value = functional.bundle(self.value, input)
        self.size += 1

    def remove(self, input: Tensor) -> None:
        self.value = functional.bundle(self.value, -input)
        self.size -= 1

    def contains(self, input: Tensor) -> Tensor:
        return functional.cosine_similarity(input, self.value.unsqueeze(0))

    def __len__(self) -> int:
        return self.size

    @classmethod
    def from_ngrams(cls, input: Tensor, n=3):
        value = functional.ngrams(input, n)
        return cls(value, size=input.size(-2) - n + 1)

    @classmethod
    def from_tensor(cls, input: Tensor):
        value = functional.multiset(input, dim=-2)
        return cls(value, size=input.size(-2))


class Sequence:
    @overload
    def __init__(self, dimensions: int, *, device=None, dtype=None):
        ...

    @overload
    def __init__(self, input: Tensor, *, length=0):
        ...

    def __init__(self, dim_or_input: int, **kwargs):
        self.length = kwargs.get("length", 0)
        if torch.is_tensor(dim_or_input):
            self.value = dim_or_input
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            device = kwargs.get("device", None)
            self.value = torch.zeros(dim_or_input, dtype=dtype, device=device)

    def append(self, input: Tensor) -> None:
        rotated_value = functional.permute(self.value, shifts=1)
        self.value = functional.bundle(input, rotated_value)
        self.length += 1

    def appendleft(self, input: Tensor) -> None:
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bundle(self.value, rotated_input)
        self.length += 1

    def pop(self, input: Tensor) -> None:
        self.length -= 1
        self.value = functional.bundle(self.value, -input)
        self.value = functional.permute(self.value, shifts=-1)

    def popleft(self, input: Tensor) -> None:
        self.length -= 1
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bundle(self.value, -rotated_input)

    def replace(self, index: int, old: Tensor, new: Tensor) -> None:
        rotated_old = functional.permute(old, shifts=-self.length + index + 1)
        self.value = functional.bundle(self.value, -rotated_old)

        rotated_new = functional.permute(new, shifts=-self.length + index + 1)
        self.value = functional.bundle(self.value, rotated_new)

    def concat(self, seq: "Sequence") -> "Sequence":
        value = functional.permute(self.value, shifts=len(seq))
        value = functional.bundle(value, seq.value)
        return Sequence(value, length=len(self) + len(seq))

    def __getitem__(self, index: int) -> Tensor:
        return functional.permute(self.value, shifts=-self.length + index + 1)

    def __len__(self) -> int:
        return self.length


class DistinctSequence:
    @overload
    def __init__(self, dimensions: int, *, device=None, dtype=None):
        ...

    @overload
    def __init__(self, input: Tensor, *, length=0):
        ...

    def __init__(self, dim_or_input: int, **kwargs):
        self.length = kwargs.get("length", 0)
        if torch.is_tensor(dim_or_input):
            self.value = dim_or_input
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            device = kwargs.get("device", None)
            self.value = torch.zeros(dim_or_input, dtype=dtype, device=device)

    def append(self, input: Tensor) -> None:
        rotated_value = functional.permute(self.value, shifts=1)
        self.value = functional.bind(input, rotated_value)
        self.length += 1

    def appendleft(self, input: Tensor) -> None:
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bind(self.value, rotated_input)
        self.length += 1

    def pop(self, input: Tensor) -> None:
        self.length -= 1
        self.value = functional.bind(self.value, input)
        self.value = functional.permute(self.value, shifts=-1)

    def popleft(self, input: Tensor) -> None:
        self.length -= 1
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bind(self.value, rotated_input)

    def replace(self, index: int, old: Tensor, new: Tensor) -> None:
        rotated_old = functional.permute(old, shifts=-self.length + index + 1)
        self.value = functional.bind(self.value, rotated_old)

        rotated_new = functional.permute(new, shifts=-self.length + index + 1)
        self.value = functional.bind(self.value, rotated_new)

    def __len__(self) -> int:
        return self.length


class Graph:
    def __init__(self, dimensions, directed=False, device=None, dtype=None):
        self.length = 0
        self.directed = directed
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def add_edge(self, node1: Tensor, node2: Tensor) -> None:
        edge = self.encode_edge(node1, node2)
        self.value = functional.bundle(self.value, edge)

    def encode_edge(self, node1: Tensor, node2: Tensor) -> Tensor:
        if self.directed:
            return functional.bind(node1, node2)
        else:
            return functional.bind(node1, functional.permute(node2))

    def node_neighbors(self, input: Tensor, outgoing=True) -> Tensor:
        if self.directed:
            if outgoing:
                return functional.permute(functional.bind(self.value, input), shifts=-1)
            else:
                return functional.bind(self.value, functional.permute(input, shifts=1))
        else:
            return functional.bind(self.value, input)

    def contains(self, input: Tensor) -> Tensor:
        return functional.cosine_similarity(input, self.value.unsqueeze(0))


class Tree:
    def __init__(self, dimensions, device=None, dtype=None):
        self.dimensions = dimensions
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)
        self.l_r = functional.random_hv(2, dimensions, dtype=dtype, device=device)

    def add_leaf(self, value: Tensor, path: List[str]) -> None:
        for idx, i in enumerate(path):
            if i == "l":
                value = functional.bind(
                    value, functional.permute(self.left, shifts=idx)
                )
            else:
                value = functional.bind(
                    value, functional.permute(self.right, shifts=idx)
                )

        self.value = functional.bundle(self.value, value)

    @property
    def left(self) -> Tensor:
        return self.l_r[0]

    @property
    def right(self) -> Tensor:
        return self.l_r[1]

    def get_leaf(self, path: List[str]) -> Tensor:
        for idx, i in enumerate(path):
            if i == "l":
                if idx == 0:
                    hv_path = self.left
                else:
                    hv_path = functional.bind(
                        hv_path, functional.permute(self.left, shifts=idx)
                    )
            else:
                if idx == 0:
                    hv_path = self.right
                else:
                    hv_path = functional.bind(
                        hv_path, functional.permute(self.right, shifts=idx)
                    )

        return functional.bind(hv_path, self.value)


class FiniteStateAutomata:
    def __init__(self, dimensions, device=None, dtype=None):
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def add_transition(
        self,
        token: Tensor,
        initial_state: Tensor,
        final_state: Tensor,
    ) -> None:
        transition_edge = functional.bind(
            initial_state, functional.permute(final_state)
        )
        transition = functional.bind(token, transition_edge)
        self.value = functional.bundle(self.value, transition)

    def transition(self, state: Tensor, action: Tensor) -> Tensor:
        # Returns the next state + some noise
        next_state = functional.bind(self.value, state)
        next_state = functional.bind(next_state, action)
        return functional.permute(next_state, shifts=-1)
