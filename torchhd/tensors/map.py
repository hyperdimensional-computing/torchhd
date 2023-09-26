#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Set

from torchhd.tensors.base import VSATensor


class MAPTensor(VSATensor):
    r"""Multiply Add Permute

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
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPTensor":
        r"""Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAPTensor.empty(3, 6)
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
    ) -> "MAPTensor":
        r"""Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAPTensor.identity(3, 6)
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
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPTensor":
        r"""Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.MAPTensor.random(3, 6)
            tensor([[-1.,  1., -1.,  1.,  1., -1.],
                    [ 1., -1.,  1.,  1.,  1.,  1.],
                    [-1.,  1.,  1.,  1., -1., -1.]])
            >>> torchhd.MAPTensor.random(3, 6, dtype=torch.long)
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

    def bundle(self, other: "MAPTensor") -> "MAPTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (MAPTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MAPTensor.random(2, 10)
            >>> a
            tensor([-1., -1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.])
            >>> b
            tensor([ 1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,  1.])
            >>> a.bundle(b)
            tensor([ 0., -2.,  0., -2., -2.,  2., -2.,  0.,  2.,  2.])

        """
        return torch.add(self, other)

    def multibundle(self) -> "MAPTensor":
        """Bundle multiple hypervectors"""
        return torch.sum(self, dim=-2, dtype=self.dtype)

    def bind(self, other: "MAPTensor") -> "MAPTensor":
        r"""Bind the hypervector with other using element-wise multiplication.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (MAPTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MAPTensor.random(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """

        return torch.mul(self, other)

    def multibind(self) -> "MAPTensor":
        """Bind multiple hypervectors"""
        return torch.prod(self, dim=-2, dtype=self.dtype)

    def inverse(self) -> "MAPTensor":
        r"""Invert the hypervector for binding.

        Each hypervector in MAP is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAPTensor.random(1, 10)
            >>> a
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])
            >>> a.inverse()
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])

        """

        return torch.clone(self)

    def negative(self) -> "MAPTensor":
        r"""Negate the hypervector for the bundling inverse

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAPTensor.random(1, 10)
            >>> a
            tensor([[-1., -1.,  1.,  1.,  1., -1.,  1., -1., -1., -1.]])
            >>> a.negative()
            tensor([[ 1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.]])
        """

        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "MAPTensor":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAPTensor.random(1, 10)
            >>> a
            tensor([[ 1.,  1.,  1., -1., -1., -1.,  1., -1., -1.,  1.]])
            >>> a.permute()
            tensor([[ 1.,  1.,  1.,  1., -1., -1., -1.,  1., -1., -1.]])

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def clipping(self, kappa) -> "MAPTensor":
        r"""Performs the clipping function that clips the lower and upper values.

        Args:
            kappa (int): specifies the range of the clipping function.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.MAPTensor.random(30, 10).multibundle()
            >>> a
            MAP([-8.,  0.,  6.,  8.,  4., -6.,  0., -2.,  0., -4.])
            >>> a.clipping(4)
            MAP([-4.,  0.,  4.,  4.,  4., -4.,  0., -2.,  0., -4.])

        """

        return torch.clamp(self, min=-kappa, max=kappa)

    def dot_similarity(self, others: "MAPTensor", *, dtype=None) -> Tensor:
        """Inner product with other hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        if others.dim() >= 2:
            others = others.transpose(-2, -1)

        return torch.matmul(self.to(dtype), others.to(dtype))

    def cosine_similarity(
        self, others: "MAPTensor", *, dtype=None, eps=1e-08
    ) -> Tensor:
        """Cosine similarity with other hypervectors"""
        if dtype is None:
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
        return self.dot_similarity(others, dtype=dtype) / magnitude
