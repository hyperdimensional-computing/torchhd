from typing import Any, List, Optional, Tuple
import torch
from . import functional


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


class Sequence:
    def __init__(self, dimensions, threshold=0.5, device=None, dtype=None):
        self.length = 0
        self.threshold = threshold
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def append(self, input: torch.Tensor) -> None:
        rotated_input = functional.permute(input, shifts=self.len)
        self.value = functional.bundle(self.value, rotated_input)

    def pop(self, index: Optional[int] = None) -> Optional[torch.Tensor]:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> torch.Tensor:
        rotated_value = functional.permute(self.value, shifts=-index)
        return rotated_value

    def __len__(self) -> int:
        return self.length
