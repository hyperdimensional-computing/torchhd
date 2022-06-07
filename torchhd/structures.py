from typing import Any, List, Optional, Tuple, overload
import torch
from torch import Tensor

import torchhd.functional as functional

__all__ = [
    "Memory",
    "Multiset",
    "HashTable",
    "BundleSequence",
    "BindSequence",
    "Graph",
    "Tree",
    "FiniteStateAutomata",
]


class Memory:
    """Associative memory of hypervector keys and any value.

    Creates a memory object.

    Args:
        threshold (float, optional): minimal similarity between input and any hypervector in memory. Default: ``0.0``.

    Examples::

        >>> memory = structures.Memory()

    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.keys: List[Tensor] = []
        self.values: List[Any] = []

    def __len__(self) -> int:
        """Returns the number of items in memory.

        Examples::

            >>> len(memory)
            0

        """
        return len(self.values)

    def add(self, key: Tensor, value: Any) -> None:
        """Adds one (key, value) pair to memory.

        Args:
            key (Tensor): Hypervector used as key for adding the key-value pair.
            value (Any): Value to be added to the memory.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> memory.add(letters_hv[0], letters[0])

        """
        self.keys.append(key)
        self.values.append(value)

    def index(self, key: Tensor) -> int:
        """Returns the index of the tensor in memory from an approximate key.

        Args:
            key (Tensor): Hypervector key used for index lookup position.

        Examples::

            >>> memory.index(letters_hv[0])
            >>> 0

        """
        if len(self.keys) == 0:
            raise Exception("No elements in memory")
        key_stack = torch.stack(self.keys, dim=0)
        sim = functional.cosine_similarity(key, key_stack)
        value, index = torch.max(sim, 0)

        if value.item() < self.threshold:
            raise IndexError("No elements in memory")

        return index

    def __getitem__(self, key: Tensor) -> Tuple[Tensor, Any]:
        """Get the (key, value) pair from an approximate key.

        Args:
            key (Tensor): Hypervector key used for item lookup.

        Examples::

            >>> memory[letters_hv[0]]
            (tensor([-1.,  1.,  1.,  ...,  1.,  1., -1.]), 'a')

        """
        index = self.index(key)
        return self.keys[index], self.values[index]

    def __setitem__(self, key: Tensor, value: Any) -> None:
        """Set the value of an (key, value) pair from an approximate key.

        Args:
            key (Tensor): Hypervector key used for item lookup.

        Examples::

            >>> memory[letters_hv[0]] = letters[1]
            >>> memory[letters_hv[0]]
            (tensor([-1.,  1.,  1.,  ...,  1.,  1., -1.]), 'b')
        """
        index = self.index(key)
        self.values[index] = value

    def __delitem__(self, key: Tensor) -> None:
        """Delete the (key, value) pair from an approximate key.

        Args:
            key (Tensor): Hypervector key used for item lookup.

        Examples::

            >>> del memory[letters_hv[0]]
            >>> memory[letters_hv[0]]
            Exception: No elements in memory
        """
        index = self.index(key)
        del self.keys[index]
        del self.values[index]


class Multiset:
    """Hypervector multiset data structure.

    Creates an empty multiset of dim dimensions or from an input tensor.

    Args:
        dimensions (int): number of dimensions of the multiset.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Args:
        input (Tensor): tensor representing a multiset.
        size (int, optional): the size of the multiset provided as input. Default: ``0``.

    Examples::

        >>> M = structures.Multiset(10000)

        >>> x = functional.random_hv(1, 10000)
        >>> M = structures.Multiset(x[0], size=1)

    """

    @overload
    def __init__(self, dimensions: int, *, device=None, dtype=None):
        ...

    @overload
    def __init__(self, input: Tensor, *, size=0):
        ...

    def __init__(self, dim_or_input: Any, **kwargs):
        self.size = kwargs.get("size", 0)
        if torch.is_tensor(dim_or_input):
            self.value = dim_or_input
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            device = kwargs.get("device", None)
            self.value = torch.zeros(dim_or_input, dtype=dtype, device=device)

    def add(self, input: Tensor) -> None:
        """Adds a new hypervector (input) to the multiset.

        Args:
            input (Tensor): Hypervector to add to the multiset.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> M.add(letters_hv[0])

        """
        self.value = functional.bundle(self.value, input)
        self.size += 1

    def remove(self, input: Tensor) -> None:
        """Removes a hypervector (input) from the multiset.

        Args:
            input (Tensor): Hypervector to be removed from the multiset.

        Examples::

            >>> M.remove(letters_hv[0])

        """
        self.value = functional.bundle(self.value, -input)
        self.size -= 1

    def contains(self, input: Tensor) -> Tensor:
        """Returns the cosine similarity of the input vector against the multiset.

        Args:
            input (Tensor): Hypervector to compare against the multiset.

        Examples::

            >>> M.contains(letters_hv[0])
            tensor(0.4575)

        """
        return functional.cosine_similarity(input, self.value)

    def __len__(self) -> int:
        """Returns the size of the multiset.

        Examples::

            >>> len(M)
            0

        """
        return self.size

    def clear(self) -> None:
        """Empties the multiset

        Examples::

            >>> M.clear()

        """
        self.size = 0
        self.value.fill_(0.0)

    @classmethod
    def from_ngrams(cls, input: Tensor, n=3):
        r"""Creates a multiset from the ngrams of a set of hypervectors.

        See: :func:`~torchhd.functional.ngrams`.

        Args:
            input (Tensor): Set of hypervectors to convert in a multiset.
            n (int, optional): The size of each :math:`n`-gram, :math:`1 \leq n \leq m`. Default: ``3``.

        Examples::

            >>> x = functional.random_hv(5, 3)
            >>> M = structures.Multiset.from_ngrams(x)

        """
        value = functional.ngrams(input, n)
        return cls(value, size=input.size(-2) - n + 1)

    @classmethod
    def from_tensor(cls, input: Tensor):
        """Creates a multiset from a set of hypervectors.

        See: :func:`~torchhd.functional.multiset`.

        Args:
            input (Tensor): Set of hypervectors to convert in a multiset.

        Examples::

            >>> x = functional.random_hv(3, 3)
            >>> M = structures.Multiset.from_tensor(x)

        """
        value = functional.multiset(input)
        return cls(value, size=input.size(-2))


class HashTable:
    """Hypervector hash table data structure.

    Creates an empty hash table of dim dimensions or a hash table from an input tensor.

    Args:
        dimensions (int): number of dimensions of the hash table.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Args:
        input (Tensor): tensor representing a hash table.
        size (int, optional): the size of the hash table provided as input. Default: ``0``.

    Examples::

        >>> H = structures.HashTable(10000)

        >>> x = functional.random_hv(3, 10000)
        >>> M = structures.HashTable(x)
    """

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

    def add(self, key: Tensor, value: Tensor) -> None:
        """Adds one (key, value) pair to the hash table.

        Args:
            key (Tensor): Hypervector used as key for adding the key-value pair.
            value (Tensor): Tensor to be added as value to the hash table

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> values = functional.random_hv(2, 10000)
            >>> H.add(letters_hv[0], values[0])

        """
        pair = functional.bind(key, value)
        self.value = functional.bundle(self.value, pair)
        self.size += 1

    def remove(self, key: Tensor, value: Tensor) -> None:
        """Removes one (key, value) pair from the hash table.

        Args:
            key (Tensor): Hypervector used as key for removing the key-value pair.
            value (Tensor): Tensor to be removed linked to the key

        Examples::

            >>> H.remove(letters_hv[0], values[0])

        """
        pair = functional.bind(key, value)
        self.value = functional.bundle(self.value, -pair)
        self.size -= 1

    def get(self, key: Tensor) -> Tensor:
        """Gets the approximate value from the key in the hash table.

        Args:
            key (Tensor): Hypervector used as key for looking its value.

        Examples::

            >>> H.get(letters_hv[0])
            tensor([ 1., -1.,  1.,  ..., -1.,  1., -1.])

        """
        return functional.unbind(self.value, key)

    def replace(self, key: Tensor, old: Tensor, new: Tensor) -> None:
        """Replace the value from key-value pair in the hash table.

        Args:
            key (Tensor): Hypervector used as key for looking its value.
            old (Tensor): Old value hypervector.
            new (Tensor): New value hypervector.

        Examples::

            >>> H.replace(letters_hv[0], values[0], values[1])

        """
        self.remove(key, old)
        self.add(key, new)

    def __getitem__(self, key: Tensor) -> Tensor:
        """Gets the approximate value from the key in the hash table.

        Args:
            key (Tensor): Hypervector used as key for looking its value.

        Examples::

            >>> H[letters_hv[0]]
            tensor([ 1., -1.,  1.,  ..., -1.,  1., -1.])

        """
        return self.get(key)

    def __len__(self) -> int:
        """Returns the size of the hash table.

        Examples::

            >>> len(H)
            0

        """
        return self.size

    def clear(self) -> None:
        """Empties the hash table.

        Examples::

            >>> H.clear()

        """
        self.size = 0
        self.value.fill_(0.0)

    @classmethod
    def from_tensors(cls, keys: Tensor, values: Tensor):
        """Creates a hash table from a set of keys and values hypervectors.

        See: :func:`~torchhd.functional.hash_table`.

        Args:
            keys (Tensor): Set of key hypervectors to add in the hash table.
            values (Tensor): Set of value hypervectors to add in the hash table.

        Examples::
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> values = functional.random_hv(len(letters), 10000)
            >>> H = structures.HashTable.from_tensors(letters_hv, values)

        """
        value = functional.hash_table(keys, values)
        return cls(value, size=keys.size(-2))


class BundleSequence:
    """Hypervector bundling-based sequence data structure

    Creates an empty sequence of dim dimensions or from an input tensor.

    Args:
        dimensions (int): number of dimensions of the sequence.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Args:
        input (Tensor): tensor representing a sequence.
        size (int, optional): the length of the sequence provided as input. Default: ``0``.

    Examples::

        >>> S = structures.BundleSequence(10000)

        >>> letters = list(string.ascii_lowercase)
        >>> letters_hv = functional.random_hv(len(letters), 10000)
        >>> S = structures.BundleSequence(letters_hv[0], size=1)

    """

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

    def append(self, input: Tensor) -> None:
        """Appends the input tensor to the right of the sequence.

        Args:
            input (Tensor): Hypervector to append to the sequence.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> S.append(letters_hv[0])

        """
        rotated_value = functional.permute(self.value, shifts=1)
        self.value = functional.bundle(input, rotated_value)
        self.size += 1

    def appendleft(self, input: Tensor) -> None:
        """Appends the input tensor to the left of the sequence.

        Args:
            input (Tensor): Hypervector to append to the right of the sequence.

        Examples::

            >>> S.appendleft(letters_hv[1])

        """
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bundle(self.value, rotated_input)
        self.size += 1

    def pop(self, input: Tensor) -> None:
        """Pops the input tensor from the right of the sequence.

        Args:
            input (Tensor): Hypervector to pop from the sequence.

        Examples::

            >>> S.pop(letters_hv[0])

        """
        self.size -= 1
        self.value = functional.bundle(self.value, -input)
        self.value = functional.permute(self.value, shifts=-1)

    def popleft(self, input: Tensor) -> None:
        """Pops the input tensor from the left of the sequence.

        Args:
            input (Tensor): Hypervector to pop left from the sequence.

        Examples::

            >>> S.popleft(letters_hv[1])

        """
        self.size -= 1
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bundle(self.value, -rotated_input)

    def replace(self, index: int, old: Tensor, new: Tensor) -> None:
        """Replace the old hypervector value from the given index, for the new hypervector value.

        Args:
            index (int): Index from the sequence to replace its value.
            old (Tensor): Old value hypervector.
            new (Tensor): New value hypervector.

        Examples::

            >>> S.replace(0, letters_hv[0], letters_hv[1])

        """
        rotated_old = functional.permute(old, shifts=self.size - index - 1)
        self.value = functional.bundle(self.value, -rotated_old)

        rotated_new = functional.permute(new, shifts=self.size - index - 1)
        self.value = functional.bundle(self.value, rotated_new)

    def concat(self, seq: "BundleSequence") -> "BundleSequence":
        """Concatenates the current sequence with the given one.

        Args:
            seq (Sequence): Sequence to be concatenated with the current one.

        Examples::

            >>> S1 = structures.BundleSequence(dimensions=10000)
            >>> S2 = S.concat(S1)

        """
        value = functional.permute(self.value, shifts=len(seq))
        value = functional.bundle(value, seq.value)
        return BundleSequence(value, size=len(self) + len(seq))

    def __getitem__(self, index: int) -> Tensor:
        """Gets the approximate value from given index.

        Args:
            index (int): Index of the value in the sequence.

        Examples::

            >>> S[0]
            tensor([ 1., -1.,  1.,  ..., -1.,  1., -1.])

        """
        return functional.permute(self.value, shifts=-self.size + index + 1)

    def __len__(self) -> int:
        """Returns the length of the sequence.

        Examples::

            >>> len(S)
            0

        """
        return self.size

    def clear(self) -> None:
        """Empties the sequence.

        Examples::

            >>> S.clear()

        """
        self.value.fill_(0.0)
        self.size = 0

    @classmethod
    def from_tensor(cls, input: Tensor):
        """Creates a sequence from hypervectors.

        See: :func:`~torchhd.functional.bundle_sequence`.

        Args:
            input (Tensor): Tensor containing hypervectors that form the sequence.

        Examples::
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> S = structures.BundleSequence.from_tensor(letters_hv)

        """
        value = functional.bundle_sequence(input)
        return cls(value, size=input.size(-2))


class BindSequence:
    """Hypervector binding-based sequence data structure.

    Creates an empty sequence of dim dimensions or from an input tensor.

    Args:
        dimensions (int): number of dimensions of the sequence.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Args:
        input (Tensor): tensor representing a binding-based sequence.
        size (int, optional): the length of the sequence provided as input. Default: ``0``.

    Examples::

        >>> DS = structures.BindSequence(10000)
    """

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
            self.value = functional.identity_hv(
                1, dim_or_input, dtype=dtype, device=device
            ).squeeze(0)

    def append(self, input: Tensor) -> None:
        """Appends the input tensor to the right of the sequence.

        Args:
            input (Tensor): Hypervector to append to the sequence.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> DS.append(letters_hv[0])

        """
        rotated_value = functional.permute(self.value, shifts=1)
        self.value = functional.bind(input, rotated_value)
        self.size += 1

    def appendleft(self, input: Tensor) -> None:
        """Appends the input tensor to the left of the sequence.

        Args:
            input (Tensor): Hypervector to append to the right of the sequence.

        Examples::

            >>> DS.appendleft(letters_hv[1])

        """
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.bind(self.value, rotated_input)
        self.size += 1

    def pop(self, input: Tensor) -> None:
        """Pops the input tensor from the right of the sequence.

        Args:
            input (Tensor): Hypervector to pop from the sequence.

        Examples::

            >>> DS.pop(letters_hv[0])

        """
        self.size -= 1
        self.value = functional.unbind(self.value, input)
        self.value = functional.permute(self.value, shifts=-1)

    def popleft(self, input: Tensor) -> None:
        """Pops the input tensor from the left of the sequence.

        Args:
            input (Tensor): Hypervector to pop left from the sequence.

        Examples::

            >>> DS.popleft(letters_hv[1])

        """
        self.size -= 1
        rotated_input = functional.permute(input, shifts=len(self))
        self.value = functional.unbind(self.value, rotated_input)

    def replace(self, index: int, old: Tensor, new: Tensor) -> None:
        """Replace the old hypervector value from the given index, for the new hypervector value.

        Args:
            index (int): Index from the sequence to replace its value.
            old (Tensor): Old value hypervector.
            new (Tensor): New value hypervector.

        Examples::

            >>> DS1 = structures.BindSequence(dimensions=10000)
            >>> DS.concat(DS1)

        """
        rotated_old = functional.permute(old, shifts=self.size - index - 1)
        self.value = functional.unbind(self.value, rotated_old)

        rotated_new = functional.permute(new, shifts=self.size - index - 1)
        self.value = functional.bind(self.value, rotated_new)

    def __len__(self) -> int:
        """Returns the length of the sequence.

        Examples::

            >>> len(DS)
            0

        """
        return self.size

    def clear(self) -> None:
        """Empties the sequence.

        Examples::

            >>> DS.clear()

        """
        self.value.fill_(1.0)
        self.size = 0

    @classmethod
    def from_tensor(cls, input: Tensor):
        """Creates a sequence from tensor.

        See: :func:`~torchhd.functional.bind_sequence`.

        Args:
            input (Tensor): Tensor containing hypervectors that form the sequence.

        Examples::
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> DS = structures.BindSequence.from_tensor(letters_hv)

        """
        value = functional.bind_sequence(input)
        return cls(value, size=input.size(-2))


class Graph:
    """Hypervector-based graph data structure.

    Creates an empty sequence of dim dimensions or from an input tensor.

    Args:
        dimensions (int): number of dimensions of the graph.
        directed (bool, optional): specify if the graph is directed or not. Default: ``False``.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        input (Tensor): tensor representing a graph hypervector.

    Examples::

        >>> G = structures.Graph(10000, directed=True)

    """

    @overload
    def __init__(self, dimensions: int, *, directed=False, device=None, dtype=None):
        ...

    @overload
    def __init__(self, input: Tensor, *, directed=False):
        ...

    def __init__(self, dim_or_input: int, **kwargs):
        self.is_directed = kwargs.get("directed", False)
        if torch.is_tensor(dim_or_input):
            self.value = dim_or_input
        else:
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            device = kwargs.get("device", None)
            self.value = torch.zeros(dim_or_input, dtype=dtype, device=device)

    def add_edge(self, node1: Tensor, node2: Tensor) -> None:
        """Adds an edge to the graph.

        If directed the direction goes from the first node to the second one.

        Args:
            node1 (Tensor): Hypervector representing the first node of the edge.
            node2 (Tensor): Hypervector representing the second node of the edge.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> G.add_edge(letters_hv[0], letters_hv[1])

        """
        edge = self.encode_edge(node1, node2)
        self.value = functional.bundle(self.value, edge)

    def encode_edge(self, node1: Tensor, node2: Tensor) -> Tensor:
        """Returns the encoding of an edge.

        If directed the direction goes from the first node to the second one.

        Args:
            node1 (Tensor): Hypervector representing the first node of the edge.
            node2 (Tensor): Hypervector representing the second node of the edge.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> G.encode_edge(letters_hv[0], letters_hv[1])
            tensor([-1.,  1., -1.,  ...,  1., -1., -1.])

        """
        if self.is_directed:
            return functional.bind(node1, functional.permute(node2))
        else:
            return functional.bind(node1, node2)

    def node_neighbors(self, input: Tensor, outgoing=True) -> Tensor:
        """Returns the multiset of node neighbors of the input node.

        Args:
            input (Tensor): Hypervector representing the node.
            outgoing (bool, optional): if ``True``, returns the neighboring nodes that ``input`` has an edge to. If ``False``, returns the neighboring nodes that ``input`` has an edge from. This only has effect for directed graphs. Default: ``True``.

        Examples::

            >>> G.node_neighbors(letters_hv[0])
            tensor([ 1.,  1.,  1.,  ..., -1., -1.,  1.])

        """
        if self.is_directed:
            if outgoing:
                permuted_neighbors = functional.unbind(self.value, input)
                return functional.permute(permuted_neighbors, shifts=-1)
            else:
                permuted_node = functional.permute(input, shifts=1)
                return functional.unbind(self.value, permuted_node)
        else:
            return functional.unbind(self.value, input)

    def contains(self, input: Tensor) -> Tensor:
        """Returns the cosine similarity of the input vector against the graph.

        Args:
            input (Tensor): Hypervector to compare against the multiset.

        Examples::

            >>> e = G.encode_edge(letters_hv[0], letters_hv[1])
            >>> G.contains(e)
            tensor(1.)
        """
        return functional.cosine_similarity(input, self.value)

    def clear(self) -> None:
        """Empties the graph.

        Examples::

            >>> G.clear()
        """
        self.value.fill_(0.0)


class Tree:
    """Hypervector-based tree data structure.

    Creates an empty tree.

    Args:
        dimensions (int): dimensions of the tree.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Examples::

        >>> T = structures.Tree(10000)

    """

    def __init__(self, dimensions, device=None, dtype=None):
        self.dimensions = dimensions
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)
        self.l_r = functional.random_hv(2, dimensions, dtype=dtype, device=device)

    def add_leaf(self, value: Tensor, path: List[str]) -> None:
        """Adds a leaf to the tree.

        Args:
            value (Tensor): Hypervector representing the first node of the edge.
            path (List[str]): Path of the leaf, using 'l' to refer as left and right 'r'.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> T.add_leaf(letters_hv[0], ['l','l'])

        """
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
        """Returns the left branch of the tree at the corresponding level.

        Examples::

            >>> T.left
            tensor([ 1., -1.,  1.,  ...,  1.,  1., -1.])

        """
        return self.l_r[0]

    @property
    def right(self) -> Tensor:
        """Returns the right branch of the tree at the corresponding level.

        Examples::

            >>> T.right
            tensor([-1., -1.,  1.,  ...,  1., -1., -1.])

        """
        return self.l_r[1]

    def get_leaf(self, path: List[str]) -> Tensor:
        """Returns the value, either subtree or node given by the path.

        Args:
            path (List[str]): Path of the tree or node wanted to get.

        Examples::

            >>> T.get_leaf(['l','l'])
            tensor([ 1., -1.,  1.,  ...,  1.,  1., -1.])

        """
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

        return functional.unbind(self.value, hv_path)

    def clear(self) -> None:
        """Empties the tree.

        Examples::

            >>> T.clear()

        """
        self.value.fill_(0.0)


class FiniteStateAutomata:
    """Hypervector-based finite state automata data structure.

    Creates an empty finite state automata.

    Args:
        dimensions (int): dimensions of the automata.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None``, uses a global default (see ``torch.set_default_tensor_type()``).
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Examples::

        >>> FSA = structures.FiniteStateAutomata(10000)

    """

    def __init__(self, dimensions, device=None, dtype=None):
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.value = torch.zeros(dimensions, dtype=dtype, device=device)

    def add_transition(
        self,
        token: Tensor,
        initial_state: Tensor,
        final_state: Tensor,
    ) -> None:
        """Adds a transition to the automata.

        Args:
            token (Tensor): token used for changing state.
            initial_state (Tensor): initial state of the transition.
            final_state (Tensor): final state of the transition.

        Examples::

            >>> letters = list(string.ascii_lowercase)
            >>> letters_hv = functional.random_hv(len(letters), 10000)
            >>> T.add_transition(letters_hv[0], letters_hv[1], letters_hv[2])

        """
        transition_edge = functional.bind(
            initial_state, functional.permute(final_state)
        )
        transition = functional.bind(token, transition_edge)
        self.value = functional.bundle(self.value, transition)

    def transition(self, state: Tensor, action: Tensor) -> Tensor:
        """Returns the next state off the automata plus some noise.

        Args:
            state (Tensor): initial state of the transition.
            action (Tensor): token used for changing state.

        Examples::

            >>> FSA.transition(letters_hv[1], letters_hv[0])
            tensor([ 1.,  1., -1.,  ..., -1., -1.,  1.])

        """
        next_state = functional.unbind(self.value, state)
        next_state = functional.unbind(next_state, action)
        return functional.permute(next_state, shifts=-1)

    def clear(self) -> None:
        """Empties the tree.

        Examples::

            >>> FSA.clear()

        """
        self.value.fill_(0.0)
