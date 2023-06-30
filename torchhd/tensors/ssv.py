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


class SSVTensor(VSATensor):
    r"""Segmented Sparse Vector

    Proposed in `High-dimensional computing with sparse vectors <https://ieeexplore.ieee.org/document/7348414>`_, this model works with sparse vector segments.
    """
    segment_size: int
    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
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
        segment_size:int=1024,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "SSVTensor":
        r"""Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.
        Because of the low precession of the BSC model an empty set cannot be explicitly represented, therefore the returned hypervectors are identical to random-hypervectors.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.SSVTensor.empty(3, 6)
            tensor([[0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.]])

        """
        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.randint(
            0, 
            segment_size,
            (num_vectors, dimensions),
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        result = result.as_subclass(cls)
        result.segment_size = segment_size
        return result

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        segment_size:int=1024,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "SSVTensor":
        r"""Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.SSVTensor.identity(3, 6)
            tensor([[1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.]])

        """
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

        result = result.as_subclass(cls)
        result.segment_size = segment_size
        return result

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        segment_size:int=1024,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "SSVTensor":
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

            >>> torchhd.SSVTensor.random(3, 6)
            tensor([[-1.,  1., -1.,  1.,  1., -1.],
                    [ 1., -1.,  1.,  1.,  1.,  1.],
                    [-1.,  1.,  1.,  1., -1., -1.]])
            >>> torchhd.SSVTensor.random(3, 6, dtype=torch.long)
            tensor([[-1,  1, -1, -1,  1,  1],
                    [ 1,  1, -1, -1, -1, -1],
                    [-1, -1, -1,  1, -1, -1]])

        """
        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.randint(
            0, 
            segment_size,
            (num_vectors, dimensions),
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        result = result.as_subclass(cls)
        result.segment_size = segment_size
        return result

    def bundle(self, other: "SSVTensor", *, generator=None) -> "SSVTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (SSVTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.SSVTensor.random(2, 10)
            >>> a
            tensor([-1., -1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.])
            >>> b
            tensor([ 1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,  1.])
            >>> a.bundle(b)
            tensor([ 0., -2.,  0., -2., -2.,  2., -2.,  0.,  2.,  2.])

        """
        select = torch.empty_like(self, dtype=torch.bool)
        select.bernoulli_(0.5, generator=generator)
        return torch.where(select, self, other)

    def multibundle(self) -> "SSVTensor":
        """Bundle multiple hypervectors"""
        return torch.mode(self, dim=-2).values

    def bind(self, other: "SSVTensor") -> "SSVTensor":
        r"""Bind the hypervector with other using element-wise multiplication.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (SSVTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.SSVTensor.random(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """

        return torch.remainder(torch.add(self, other), self.segment_size)

    def multibind(self) -> "SSVTensor":
        """Bind multiple hypervectors"""
        return torch.remainder(torch.sum(self, dim=-2, dtype=self.dtype), self.segment_size)

    def inverse(self) -> "SSVTensor":
        r"""Invert the hypervector for binding.

        Each hypervector in MAP is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.SSVTensor.random(1, 10)
            >>> a
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])
            >>> a.inverse()
            tensor([[-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.]])

        """

        return torch.remainder(torch.negative(self), self.segment_size)

    def negative(self) -> "SSVTensor":
        r"""Negate the hypervector for the bundling inverse

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.SSVTensor.random(1, 10)
            >>> a
            tensor([[-1., -1.,  1.,  1.,  1., -1.,  1., -1., -1., -1.]])
            >>> a.negative()
            tensor([[ 1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1.]])
        """

        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "SSVTensor":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.SSVTensor.random(1, 10)
            >>> a
            tensor([[ 1.,  1.,  1., -1., -1., -1.,  1., -1., -1.,  1.]])
            >>> a.permute()
            tensor([[ 1.,  1.,  1.,  1., -1., -1., -1.,  1., -1., -1.]])

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def dot_similarity(self, others: "SSVTensor") -> Tensor:
        """Inner product with other hypervectors"""
        dtype = torch.get_default_dtype()
        return torch.sum(self == others, dim=-1, dtype=dtype)

    def cosine_similarity(self, others: "SSVTensor") -> Tensor:
        """Cosine similarity with other hypervectors"""
        magnitude = self.size(-1)
        return self.dot_similarity(others) / magnitude
