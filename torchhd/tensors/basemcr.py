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


class BaseMCRTensor(VSATensor):
    r"""Base class for VSA Modular Composite Representations (MCR)

    Proposed in `Modular Composite Representation <https://link.springer.com/article/10.1007/s12559-013-9243-y>`_, this model works with modular integer vectors. The base class is used as template for the MCR and the Cyclic Group Representation (CGR), which is very similar to MCR but uses a different bundling operation.
    """

    block_size: int
    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
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
        block_size: int,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "BaseMCRTensor":
        r"""Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.
        Because of the low precession of the MCR model an empty set cannot be explicitly represented, therefore the returned hypervectors are identical to random-hypervectors.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            block_size (int): the number of elements per block which controls the angular granularity.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: ``int64``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BaseMCRTensor.empty(3, 6, block_size=64)
            BaseMCRTensor([[54,  3, 22, 27, 41, 21],
                       [17, 31, 55,  3, 44, 52],
                       [42, 37, 60, 54, 13, 41]])

        """

        if dtype == None:
            dtype = torch.int64

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(
                f"{name} vectors must be one of dtype {options}, got {dtype}."
            )

        result = torch.randint(
            0,
            block_size,
            (num_vectors, dimensions),
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        result = result.as_subclass(cls)
        result.block_size = block_size
        return result

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        block_size: int,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "BaseMCRTensor":
        r"""Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            block_size (int): the number of elements per block which controls the angular granularity.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``int64`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BaseMCRTensor.identity(3, 6, block_size=64)
            BaseMCRTensor([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])

        """
        if dtype == None:
            dtype = torch.int64

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(
                f"{name} vectors must be one of dtype {options}, got {dtype}."
            )

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        result = result.as_subclass(cls)
        result.block_size = block_size
        return result

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        block_size: int,
        generator=None,
        dtype=torch.int64,
        device=None,
        requires_grad=False,
    ) -> "BaseMCRTensor":
        r"""Creates a set of random independent hypervectors.

        The resulting hypervectors sample uniformly random integers between 0 and ``block_size``.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            block_size (int): the number of elements per block which controls the angular granularity.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: ``int64``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.BaseMCRTensor.random(3, 6, block_size=64)
            BaseMCRTensor([[ 7,  1, 39,  8, 55, 22],
                       [51, 38, 59, 45, 13, 29],
                       [19, 26, 30,  5, 15, 51]])
            >>> torchhd.BaseMCRTensor.random(3, 6, block_size=128, dtype=torch.float32)
            BaseMCRTensor([[116.,  25., 100.,  10.,  21.,  86.],
                       [ 69.,  49.,   2.,  56.,  78.,  70.],
                       [ 77.,  47.,  37., 106.,   8.,  30.]])

        """
        if dtype == None:
            dtype = torch.int64

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(
                f"{name} vectors must be one of dtype {options}, got {dtype}."
            )

        result = torch.randint(
            0,
            block_size,
            (num_vectors, dimensions),
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        result = result.as_subclass(cls)
        result.block_size = block_size
        return result

    def to_complex_unit(self):
        angles = 2 * torch.pi * self / self.block_size
        return torch.polar(torch.ones_like(self, dtype=angles.dtype), angles)

    def bundle(self, other: "BaseMCRTensor") -> "BaseMCRTensor":
        """Bundle the hypervector with other"""
        raise NotImplementedError

    def multibundle(self) -> "BaseMCRTensor":
        """Bundle multiple hypervectors"""
        return super().multibundle()

    def bind(self, other: "BaseMCRTensor") -> "BaseMCRTensor":
        r"""Bind the hypervector with other using circular convolution.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (BaseMCRTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.BaseMCRTensor.random(2, 10, block_size=64)
            >>> a
            BaseMCRTensor([18, 55, 40, 62, 39, 26, 35, 24, 49, 41])
            >>> b
            BaseMCRTensor([46, 36, 21, 23, 25, 12, 29, 53, 54, 41])
            >>> a.bind(b)
            BaseMCRTensor([ 0, 27, 61, 21,  0, 38,  0, 13, 39, 18])

        """
        return torch.remainder(torch.add(self, other), self.block_size)

    def multibind(self) -> "BaseMCRTensor":
        """Bind multiple hypervectors"""
        return torch.remainder(
            torch.sum(self, dim=-2, dtype=self.dtype), self.block_size
        )

    def inverse(self) -> "BaseMCRTensor":
        r"""Invert the hypervector for binding.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.BaseMCRTensor.random(1, 10, block_size=64)
            >>> a
            BaseMCRTensor([[ 5, 30, 15, 43, 19, 36,  4, 14, 57, 34]])
            >>> a.inverse()
            BaseMCRTensor([[59, 34, 49, 21, 45, 28, 60, 50,  7, 30]])

        """

        return torch.remainder(torch.negative(self), self.block_size)

    def permute(self, shifts: int = 1) -> "BaseMCRTensor":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.BaseMCRTensor.random(1, 10, block_size=64)
            >>> a
            BaseMCRTensor([[33, 24,  1, 36,  2, 57, 11, 59, 33,  3]])
            >>> a.permute(4)
            BaseMCRTensor([[11, 59, 33,  3, 33, 24,  1, 36,  2, 57]])

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def normalize(self) -> "BaseMCRTensor":
        r"""Normalize the hypervector.

        Each operation on MCR hypervectors ensures it remains normalized, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> x = torchhd.BaseMCRTensor.random(4, 6, block_size=64).multibundle()
            >>> x
            BaseMCRTensor([28, 27, 20, 44, 57, 18])
            >>> x.normalize()
            BaseMCRTensor([28, 27, 20, 44, 57, 18])

        """
        return self.clone()

    def dot_similarity(self, others: "BaseMCRTensor", *, dtype=None) -> Tensor:
        """Based on 'Manhattan Distance in a Modular Space'.
            Distance of two elements devided by the avearage distance of two random numbers.
        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        random_distance = self.block_size/4


        if self.dim() > 1 and others.dim() > 1:
            aminusb = torch.remainder(self.unsqueeze(-2) - others.unsqueeze(-3), self.block_size)
            bminusa = torch.remainder(others.unsqueeze(-3) - self.unsqueeze(-2), self.block_size)
        else:
            aminusb = torch.remainder(self - others, self.block_size)
            bminusa = torch.remainder(others - self, self.block_size)
        distance = torch.min(aminusb,bminusa)
        normalized_distance = distance/random_distance

        return torch.sum(1-normalized_distance, dim=-1, dtype=dtype)

    def cosine_similarity(self, others: "BaseMCRTensor", *, dtype=None) -> Tensor:
        """Cosine similarity with other hypervectors"""
        magnitude = self.size(-1)
        return self.dot_similarity(others, dtype=dtype) / magnitude

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Ensure that all the build-in torch operations on this Tensor subclass maintain the block_size property

        if kwargs is None:
            kwargs = {}

        def _parse_container_for_attr(container, attr):
            s = set()
            for a in container:
                if type(a) is tuple or type(a) is list:
                    s |= _parse_container_for_attr(a, attr)
                else:
                    if hasattr(a, attr):
                        s.add(a.block_size)
            return s

        # Args is a tuple that can contain other tuples or lists. Parse it
        # reccursively to find any BaseMCRTensor object
        block_sizes = _parse_container_for_attr(args, "block_size")

        if len(block_sizes) != 1:
            raise RuntimeError(
                f"Call to {func} must contain exactly one block size, got {list(block_sizes)}"
            )

        # Call with super to avoid infinite recursion
        ret = super().__torch_function__(func, types, args, kwargs)

        if isinstance(ret, BaseMCRTensor):
            ret.block_size = list(block_sizes)[0]
        elif isinstance(ret, (tuple, list)):
            for x in ret:
                if isinstance(x, BaseMCRTensor):
                    x.block_size = list(block_sizes)[0]

        # TODO: handle more return types
        return ret
