import torch
from typing import Set
from torch import Tensor
import torch.nn.functional as F

from torchhd.tensors.base import VSATensor


def biggest_power_two(n):
    """Returns the biggest power of two <= n"""
    # if n is a power of two simply return it
    if not (n & (n - 1)):
        return n

    # else set only the most significant bit
    return int("1" + (len(bin(n)) - 3) * "0", 2)


class BSCTensor(VSATensor):
    """Binary Spatter Codes

    Proposed in `Binary spatter-coding of ordered K-tuples <https://link.springer.com/chapter/10.1007/3-540-61510-5_146>`_, this model works with binary valued hypervectors.
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    }

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=torch.bool,
        device=None,
        requires_grad=False,
    ) -> "BSCTensor":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a random-hypervector :math:`x`, the result is :math:`\sim x`.
        Because of the low precession of the BSC model an empty set cannot be explicitly represented, therefore the returned hypervectors are identical to random-hypervectors.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BSCTensor.empty(3, 6)
            tensor([[False, False, False, False,  True,  True],
                    [False,  True, False, False,  True,  True],
                    [ True, False,  True,  True, False, False]])

            >>> torchhd.BSCTensor.empty(3, 6, dtype=torch.long)
            tensor([[0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1],
                    [0, 1, 1, 0, 1, 1]])

        """
        if dtype is None:
            dtype = torch.bool

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.bernoulli_(0.5, generator=generator)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.bool,
        device=None,
        requires_grad=False,
    ) -> "BSCTensor":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BSCTensor.identity(3, 6)
            tensor([[False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False]])

            >>> torchhd.BSCTensor.identity(3, 6, dtype=torch.long)
            tensor([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

        """
        if dtype is None:
            dtype = torch.bool

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        sparsity=0.5,
        generator=None,
        dtype=torch.bool,
        device=None,
        requires_grad=False,
    ) -> "BSCTensor":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            sparsity (float, optional): the expected fraction of elements to be in-active. Has no effect on complex hypervectors. Default: ``0.5``.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BSCTensor.random(3, 6)
            tensor([[ True, False, False, False,  True,  True],
                    [False, False,  True, False, False, False],
                    [False, False,  True,  True, False, False]])

            >>> torchhd.BSCTensor.random(3, 6, sparsity=0.1)
            tensor([[ True,  True,  True,  True,  True,  True],
                    [False,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True]])

            >>> torchhd.BSCTensor.random(3, 6, dtype=torch.long)
            tensor([[1, 1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 1],
                    [0, 1, 1, 0, 0, 0]])

        """
        if dtype is None:
            dtype = torch.bool

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.bernoulli_(1.0 - sparsity, generator=generator)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(
        self, other: "BSCTensor", *, generator: torch.Generator = None
    ) -> "BSCTensor":
        r"""Bundle the hypervector with other using majority voting.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Ties in the majority vote are broken at random. For a deterministic result provide a random number generator.

        Args:
            other (BSC): other input hypervector
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.BSCTensor.random(2, 10)
            >>> a
            tensor([ True, False,  True, False, False,  True,  True, False,  True, False])
            >>> b
            tensor([ True, False, False,  True,  True, False, False,  True, False,  True])
            >>> a.bundle(b)
            tensor([ True, False,  True,  True, False,  True,  True, False, False, False])

            >>> a, b = torchhd.BSCTensor.random(2, 10, dtype=torch.long)
            >>> a
            tensor([1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
            >>> b
            tensor([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
            >>> a.bundle(b)
            tensor([1, 0, 1, 0, 0, 0, 1, 1, 0, 1])

        """

        tiebreaker = torch.empty_like(other)
        tiebreaker.bernoulli_(0.5, generator=generator)

        is_majority = self == other
        return self.where(is_majority, tiebreaker)

    def multibundle(self, *, generator: torch.Generator = None) -> "BSCTensor":
        r"""Bundle multiple hypervectors."""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)

        count = self.sum(dim=-2, dtype=torch.long)

        # add a tiebreaker when there are an even number of hvs
        if n % 2 == 0:
            tiebreaker = torch.empty_like(count)
            tiebreaker.bernoulli_(0.5, generator=generator)
            count += tiebreaker
            n += 1

        threshold = n // 2
        return torch.greater(count, threshold).to(self.dtype)

    def bind(self, other: "BSCTensor") -> "BSCTensor":
        r"""Bind the hypervector with other using XOR.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (BSC): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.BSCTensor.random(2, 10)
            >>> a
            tensor([ True, False,  True,  True, False,  True, False, False,  True, False])
            >>> b
            tensor([ True, False, False, False, False,  True, False, False, False, False])
            >>> a.bind(b)
            tensor([False, False,  True,  True, False, False, False, False,  True, False])

            >>> a, b = torchhd.BSCTensor.random(2, 10, dtype=torch.long)
            >>> a
            tensor([1, 0, 0, 1, 0, 1, 0, 0, 0, 0])
            >>> b
            tensor([0, 0, 0, 1, 0, 0, 1, 0, 1, 0])
            >>> a.bind(b)
            tensor([1, 0, 0, 0, 0, 1, 1, 0, 1, 0])

        """

        return self.logical_xor(other).to(other.dtype)

    def multibind(self) -> "BSCTensor":
        """Bind multiple hypervectors."""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        n_ = biggest_power_two(n)
        output = self[..., :n_, :]

        # parallelize many XORs in a hierarchical manner
        # for larger batches this is significantly faster
        while output.size(-2) > 1:
            output = torch.logical_xor(output[..., 0::2, :], output[..., 1::2, :])

        output = output.squeeze(-2)

        # TODO: as an optimization we could also perform the hierarchical XOR
        # on the leftovers in a recursive fashion
        leftovers = torch.unbind(self[..., n_:, :], -2)
        for i in range(n - n_):
            output = torch.logical_xor(output, leftovers[i])

        return output.to(self.dtype)

    def inverse(self) -> "BSCTensor":
        r"""Invert the hypervector for binding.

        Each hypervector in BSC is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.BSCTensor.random(1, 10)
            >>> a
            tensor([[False, False, False,  True,  True, False, False,  True,  True, False]])
            >>> a.inverse()
            tensor([[False, False, False,  True,  True, False, False,  True,  True, False]])

            >>> a = torchhd.BSCTensor.random(1, 10, dtype=torch.long)
            >>> a
            tensor([[0, 1, 0, 1, 1, 1, 1, 0, 1, 1]])
            >>> a.inverse()
            tensor([[0, 1, 0, 1, 1, 1, 1, 0, 1, 1]])

        """

        return self.clone()

    def negative(self) -> "BSCTensor":
        r"""Negate the hypervector for the bundling inverse.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.BSCTensor.random(1, 10)
            >>> a
            tensor([[ True,  True,  True,  True, False, False, False,  True,  True,  True]])
            >>> a.negative()
            tensor([[False, False, False, False,  True,  True,  True, False, False, False]])

            >>> a = torchhd.BSCTensor.random(1, 10, dtype=torch.long)
            >>> a
            tensor([[0, 1, 0, 1, 0, 0, 1, 1, 0, 1]])
            >>> a.negative()
            tensor([[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]])

        """
        out = torch.empty_like(self).as_subclass(BSCTensor)
        return torch.logical_not(self, out=out)

    def permute(self, shifts: int = 1) -> "BSCTensor":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.BSCTensor.random(1, 10)
            >>> a
            tensor([[ True, False, False, False, False, False, False, False, False, False]])
            >>> a.permute()
            tensor([[False,  True, False, False, False, False, False, False, False, False]])

            >>> a = torchhd.BSCTensor.random(1, 10, dtype=torch.long)
            >>> a
            tensor([[1, 1, 0, 0, 1, 1, 0, 0, 1, 1]])
            >>> a.permute()
            tensor([[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]])

        """
        return super().roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "BSCTensor") -> Tensor:
        """Inner product with other hypervectors."""
        dtype = torch.get_default_dtype()
        device = self.device

        min_one = torch.tensor(-1.0, dtype=dtype, device=device)
        plus_one = torch.tensor(1.0, dtype=dtype, device=device)

        self_as_bipolar = torch.where(self.bool(), min_one, plus_one)
        others_as_bipolar = torch.where(others.bool(), min_one, plus_one)

        if others.dim() >= 2:
            others_as_bipolar = others_as_bipolar.mT
        return torch.matmul(self_as_bipolar, others_as_bipolar)

    def cosine_similarity(self, others: "BSCTensor") -> Tensor:
        """Cosine similarity with other hypervectors."""
        d = self.size(-1)
        return self.dot_similarity(others) / d
