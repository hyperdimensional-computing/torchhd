import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Set

from torchhd.base import VSA_Model


class MAP(VSA_Model):
    """Multiply Add Permute

    Proposed in `Multiplicative Binding, Representation Operators & Analogy <https://www.researchgate.net/publication/215992330_Multiplicative_Binding_Representation_Operators_Analogy>`_, this model works with dense bipolar hypervectors with elements from :math:`\{-1,1\}`.
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAP.empty_hv(3, 6)
            tensor([[0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.]])

        """

        if dtype is None:
            dtype = torch.get_default_dtype()

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
    def identity_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAP.identity_hv(3, 6)
            tensor([[1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.]])

        """

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.ones(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAP":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSA_Model.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAP.random_hv(3, 6)
            tensor([[-1.,  1., -1.,  1.,  1., -1.],
                    [ 1., -1.,  1.,  1.,  1.,  1.],
                    [-1.,  1.,  1.,  1., -1., -1.]])
            >>> torchhd.MAP.random_hv(3, 6, dtype=torch.long)
            tensor([[-1,  1, -1, -1,  1,  1],
                    [ 1,  1, -1, -1, -1, -1],
                    [-1, -1, -1,  1, -1, -1]])

        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=device)
        select.bernoulli_(generator=generator)

        result = torch.where(select, -1, +1).to(dtype=dtype, device=device)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "MAP") -> "MAP":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (MAP): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MAP.random_hv(2, 10)
            >>> a
            tensor([-1., -1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.])
            >>> b
            tensor([ 1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,  1.])
            >>> a.bundle(b)
            tensor([ 0., -2.,  0., -2., -2.,  2., -2.,  0.,  2.,  2.])

        """
        return self.add(other)

    def multibundle(self) -> "MAP":
        """Bundle multiple hypervectors"""
        return self.sum(dim=-2, dtype=self.dtype)

    def bind(self, other: "MAP") -> "MAP":
        r"""Bind the hypervector with other using element-wise multiplication.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (MAP): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MAP.random_hv(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """

        return self.mul(other)

    def multibind(self) -> "MAP":
        """Bind multiple hypervectors"""
        return self.prod(dim=-2, dtype=self.dtype)

    def inverse(self) -> "MAP":
        r"""Invert the hypervector for binding.

        Each hypervector in MAP is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAP.random_hv(1, 10)
            >>> a
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])
            >>> a.inverse()
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])

        """

        return self.clone()

    def negative(self) -> "MAP":
        """Negate the hypervector for the bundling inverse

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAP.random_hv(1, 10)
            >>> a
            tensor([[-1., -1.,  1.,  1.,  1., -1.,  1., -1., -1., -1.]])
            >>> a.negative()
            tensor([[ 1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.]])
        """

        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "MAP":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAP.random_hv(1, 10)
            >>> a
            tensor([[ 1.,  1.,  1., -1., -1., -1.,  1., -1., -1.,  1.]])
            >>> a.permute()
            tensor([[ 1.,  1.,  1.,  1., -1., -1., -1.,  1., -1., -1.]])

        """
        return self.roll(shifts=shifts, dims=-1)

    def dot_similarity(self, others: "MAP") -> Tensor:
        """Inner product with other hypervectors"""
        dtype = torch.get_default_dtype()
        return F.linear(self.to(dtype), others.to(dtype))

    def cos_similarity(self, others: "MAP", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        dtype = torch.get_default_dtype()

        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = self_dot.sqrt()

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = others_dot.sqrt()

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = magnitude.clamp(min=eps)
        return F.linear(self.to(dtype), others.to(dtype)) / magnitude
