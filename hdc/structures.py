from typing import Any, List, Optional, Tuple
import torch
import functional


class Memory:
    """Associative memory"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.keys: List[torch.Tensor] = []
        self.values: List[Any] = []

    def __len__(self) -> int:
        """Returns the number of items in memory"""
        return len(self.values)

    def add(self, key: torch.Tensor, value: Any) -> None:
        """Adds one (key, value) pair to memory"""
        self.keys.append(key)
        self.values.append(value)

    def _get_index(self, key: torch.Tensor) -> int:
        key_stack = torch.stack(self.keys, dim=0)
        sim = functional.cosine_similarity(key, key_stack)
        value, index = torch.max(sim, 0)

        if value.item() < self.threshold:
            raise IndexError()

        return index

    def __getitem__(self, key: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Get the (key, value) pair with an approximate key"""
        index = self._get_index(key)
        return self.keys[index], self.values[index]

    def __setitem__(self, key: torch.Tensor, value: Any) -> None:
        """Set the value of an (key, value) pair with an approximate key"""
        index = self._get_index(key)
        self.values[index] = value

    def __delitem__(self, key: torch.Tensor) -> None:
        """Delete the (key, value) pair with an approximate key"""
        index = self._get_index(key)
        del self.keys[index]
        del self.values[index]


class Set:
    def __init__(self, dimensions, threshold=0.5, device=None, dtype=None):
        self.cardinality = 0
        self.threshold = threshold
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def add(self, input: torch.Tensor) -> None:
        if input in self:
            return

        self.value = functional.bundle(self.value, input)
        self.cardinality -= 1

    def remove(self, input: torch.Tensor) -> None:
        if input not in self:
            return

        self.value = functional.bundle(self.value, -input)
        self.cardinality += 1

    def __contains__(self, input: torch.Tensor):
        sim = functional.cosine_similarity(input, self.values.unsqueeze(0))
        return sim.item() > self.threshold

    def __len__(self) -> int:
        return self.cardinality


class Histogram:
    def __init__(self, dimensions, threshold=0.5, device=None, dtype=None):
        self.threshold = threshold
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def add(self, input: torch.Tensor) -> None:
        self.value = functional.bundle(self.value, input)

    def remove(self, input: torch.Tensor) -> None:
        if input not in self:
            return
        self.value = functional.bundle(self.value, -input)

    def __contains__(self, input: torch.Tensor):
        sim = functional.cosine_similarity(input, self.values.unsqueeze(0))
        return sim.item() > self.threshold

    def from_ngrams(self, x, n=3):
        self.value = functional.ngram(x, n)


class Sequence:
    def __init__(self, dimensions, threshold=0.5, device=None, dtype=None):
        self.length = 0
        self.threshold = threshold
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def append(self, input: torch.Tensor) -> None:
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bundle(self.value, rotated_input)

    def appendleft(self, input: torch.Tensor) -> None:
        rotated_value = functional.permute(self.value, shifts=1)
        self.value = functional.bundle(input, rotated_value)

    def pop(self) -> Optional[torch.Tensor]:
        popped_value = functional.permute(self.value, shifts=-len(self))
        self.value = functional.bundle(self.value, -popped_value)
        return popped_value

    def popleft(self) -> Optional[torch.Tensor]:
        popped_value = functional.permute(self.value, shifts=-1)
        self.value = functional.permute(functional.bundle(self.value, -popped_value), shifts=-1)
        return popped_value

    def __getitem__(self, index: int) -> torch.Tensor:
        rotated_value = functional.permute(self.value, shifts=-index)
        return rotated_value

    def __len__(self) -> int:
        return self.length


class Graph:
    def __init__(self, dimensions, threshold=0.5, directed=False, device=None, dtype=None):
        self.length = 0
        self.threshold = threshold
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)
        self.directed = directed

    def add_edge(self, node1, node2):
        if self.directed:
            edge = functional.bind(node1, node2)
        else:
            edge = functional.bind(node1, functional.permute(node2))

        if edge not in self:
            self.value = functional.bundle(self.value, edge)

    def node_neighbours(self, input: torch.Tensor):
        return functional.bind(self.value, input)

    def __contains__(self, input: torch.Tensor):
        sim = functional.cosine_similarity(input, self.value.unsqueeze(0))
        return sim.item() > self.threshold


class Tree:
    def __init__(self, dimensions, directed=False, device=None, dtype=None):
        self.length = 0
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)
        self.directed = directed
        self.l_r = functional.random_hv(2, dimensions)

    def add_leaf(self, value, path):
        for i in path:
            if i == 'l':
                value = functional.bind(value, self.left())
            else:
                value = functional.bind(value, self.right())
        self.value = functional.bundle(self.value, value)

    def left(self):
        return self.l_r[0]

    def right(self):
        return self.l_r[1]
