import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Set, Optional

from torchhd.tensors.base import VSATensor


class TPTensor(VSATensor):
    """Tensor product binding

    Proposed in `Tensor product variable binding and the representation of symbolic structures in connectionist systems <https://www.sciencedirect.com/science/article/pii/000437029090007M>`_.
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
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "TPTensor":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.TPTensor.empty(3, 6)
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
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "TPTensor":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors, is not used.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.TPTensor.identity(3, 6)
            tensor([[1.],
                    [1.],
                    [1.]])

        """

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.ones(
            num_vectors,
            1,
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
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "TPTensor":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.TPTensor.random(3, 6)
            tensor([[-1.,  1., -1.,  1.,  1., -1.],
                    [ 1., -1.,  1.,  1.,  1.,  1.],
                    [-1.,  1.,  1.,  1., -1., -1.]])
            >>> torchhd.TPTensor.random(3, 6, dtype=torch.long)
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

    def bundle(self, other: "TPTensor") -> "TPTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (TPTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.TPTensor.random(2, 10)
            >>> a
            tensor([-1., -1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.])
            >>> b
            tensor([ 1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,  1.])
            >>> a.bundle(b)
            tensor([ 0., -2.,  0., -2., -2.,  2., -2.,  0.,  2.,  2.])

        """
        return torch.add(self, other)

    def multibundle(self) -> "TPTensor":
        """Bundle multiple hypervectors"""
        return torch.sum(self, dim=-2, dtype=self.dtype)

    def bind(self, other: "TPTensor") -> "TPTensor":
        r"""Bind the hypervector with other using the tensor product (outer product).

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (TPTensor): other input hypervector

        Shapes:
            - Self: :math:`(*, f)`
            - Other: :math:`(*, g)`
            - Output: :math:`(*, fg)`

        Examples::

            >>> a, b = torchhd.TPTensor.random(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """
        outer = torch.mul(self.unsqueeze(-1), other.unsqueeze(-2))
        return torch.flatten(outer, start_dim=-2)

    def unbind(
        self, *, left: Optional["TPTensor"] = None, right: Optional["TPTensor"] = None
    ) -> "TPTensor":
        r"""Unind the hypervector either from the left or right side.

        Must specify either left or right but not both.

        Args:
            left (TPTensor, optional): other input hypervector
            right (TPTensor, optional): other input hypervector

        Shapes:
            - Self: :math:`(*, fg)`
            - Other: :math:`(*, g)`
            - Output: :math:`(*, f)`

        Examples::

            >>> a, b = torchhd.TPTensor.random(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """
        if left is None and right is None:
            raise ValueError("Must specify one of left or right.")

        if left is not None and right is not None:
            raise ValueError("Must specify only one of left or right, not both.")

        batch_size = list(self.size()[:-1])

        if left is not None:
            g = left.size(-1)
            output = torch.matmul(left.unsqueeze(-2), self.view(*batch_size, g, -1))
            output = output.squeeze(-2)
        else:
            g = right.size(-1)
            output = torch.matmul(self.view(*batch_size, -1, g), right.unsqueeze(-1))
            output = output.squeeze(-1)

        return output

    # def multibind(self) -> "TPTensor":
    #     """Bind multiple hypervectors"""
    #     return torch.prod(self, dim=-2, dtype=self.dtype)

    def inverse(self) -> "TPTensor":
        r"""Invert the hypervector for binding.

        Each hypervector in MAP is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.TPTensor.random(1, 10)
            >>> a
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])
            >>> a.inverse()
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])

        """

        return torch.clone(self)

    def negative(self) -> "TPTensor":
        """Negate the hypervector for the bundling inverse

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.TPTensor.random(1, 10)
            >>> a
            tensor([[-1., -1.,  1.,  1.,  1., -1.,  1., -1., -1., -1.]])
            >>> a.negative()
            tensor([[ 1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.]])
        """

        return torch.negative(self)

    def clipping(self, kappa) -> "TPTensor":
        """Performs the clipping function that clips the lower and upper values.

        Args:
            kappa (int): specifies the range of the clipping function.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.TPTensor.random(30, 10).multibundle()
            >>> a
            MAP([-8.,  0.,  6.,  8.,  4., -6.,  0., -2.,  0., -4.])
            >>> a.clipping(4)
            MAP([-4.,  0.,  4.,  4.,  4., -4.,  0., -2.,  0., -4.])

        """

        return torch.clamp(self, min=-kappa, max=kappa)

    def dot_similarity(self, others: "TPTensor") -> Tensor:
        """Inner product with other hypervectors"""
        dtype = torch.get_default_dtype()
        if others.dim() >= 2:
            others = others.mT
        return torch.matmul(self.to(dtype), others.to(dtype))

    def cosine_similarity(self, others: "TPTensor", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        dtype = torch.get_default_dtype()

        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = torch.sqrt(others_dot)

        if self.dim() >= 2:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
        else:
            magnitude = self_mag * others_mag

        magnitude = torch.clamp(magnitude, min=eps)
        return self.dot_similarity(others) / magnitude
