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
from typing import Set
import torch
from torch import Tensor
from torch.fft import fft, ifft
import math

from torchhd.tensors.base import VSATensor


class VTBTensor(VSATensor):
    """Vector-Derived Transformation Binding

    Proposed in `Vector-Derived Transformation Binding: An Improved Binding Operation for Deep Symbol-Like Processing in Neural Networks <https://direct.mit.edu/neco/article/31/5/849/8469/Vector-Derived-Transformation-Binding-An-Improved>`_, as an improvement upon Holographic Reduced Representations (HRR), this model also uses real valued hypervectors.
    """

    supported_dtypes: Set[torch.dtype] = {torch.float32, torch.float64}

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "VTBTensor":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a random-hypervector :math:`x`, the result is :math:`x`.
        The empty vector of the VTB model is simply a set of 0 values.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors, must have an integer square root.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.VTBTensor.empty(3, 9)
            VTBTensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            >>> torchhd.VTBTensor.empty(3, 9, dtype=torch.float64)
                VTBTensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float64)

        """

        has_int_sqrt = math.isclose(math.sqrt(dimensions) % 1.0, 0.0, abs_tol=1e-9)
        if not has_int_sqrt:
            raise ValueError(
                f"The dimensionality of VTB tensors must have an integer square root, got {dimensions} with square root {math.sqrt(dimensions)}"
            )

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
    ) -> "VTBTensor":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors, must have an integer square root.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.VTBTensor.identity(3, 9)
                VTBTensor([[0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577],
                        [0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577],
                        [0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577]])
            >>> torchhd.VTBTensor.identity(3, 9, dtype=torch.float64)
                VTBTensor([[0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577],
                        [0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577],
                        [0.577, 0.000, 0.000, 0.000, 0.577, 0.000, 0.000, 0.000, 0.577]],
                        dtype=torch.float64)

        """

        has_int_sqrt = math.isclose(math.sqrt(dimensions) % 1.0, 0.0, abs_tol=1e-9)
        if not has_int_sqrt:
            raise ValueError(
                f"The dimensionality of VTB tensors must have an integer square root, got {dimensions} with square root {math.sqrt(dimensions)}"
            )

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        mag = dimensions ** (-0.25)
        sqrt_d = int(math.sqrt(dimensions))

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
        )
        result[:, 0 :: sqrt_d + 1] = mag
        result.requires_grad = requires_grad
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
    ) -> "VTBTensor":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled at random from a normal with mean 0 and standard deviation 1/dimensions.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors, must have an integer square root.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.VTBTensor.random(3, 9)
                VTBTensor([[-0.420, -0.069,  0.014, -0.226,  0.399,  0.066, -0.606, -0.184, 0.451],
                        [-0.074,  0.270, -0.044,  0.691,  0.456, -0.144, -0.252, -0.170, -0.349],
                        [ 0.197,  0.347,  0.596, -0.323, -0.397, -0.173, -0.317,  0.217, -0.215]])
            >>> torchhd.VTBTensor.random(3, 9, dtype=torch.float64)
                VTBTensor([[ 0.308,  0.055, -0.471, -0.587,  0.054,  0.371, -0.038,  0.432, 0.080],
                        [-0.267, -0.510, -0.301,  0.435,  0.244,  0.122, -0.094,  0.552, 0.033],
                        [-0.571,  0.540,  0.135, -0.112, -0.048, -0.516, -0.019, -0.226, 0.177]], dtype=torch.float64)
        """

        has_int_sqrt = math.isclose(math.sqrt(dimensions) % 1.0, 0.0, abs_tol=1e-9)
        if not has_int_sqrt:
            raise ValueError(
                f"The dimensionality of VTB tensors must have an integer square root, got {dimensions} with square root {math.sqrt(dimensions)}"
            )

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        # Create random unit vector
        result = torch.randn(size, dtype=dtype, device=device, generator=generator)
        result.div_(result.norm(dim=-1, keepdim=True))

        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "VTBTensor") -> "VTBTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (VTB): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.VTBTensor.random(2, 9)
            >>> a
            VTBTensor([ 0.276,  0.206, -0.383,  0.553,  0.050,  0.334,  0.336, -0.067, 0.445])
            >>> b
            VTBTensor([-0.315,  0.626, -0.460, -0.345,  0.194,  0.121, -0.234, -0.207, -0.170])
            >>> a.bundle(b)
            VTBTensor([-0.039,  0.832, -0.842,  0.209,  0.244,  0.455,  0.101, -0.274, 0.275])

            >>> a, b = torchhd.VTBTensor.random(2, 9, dtype=torch.float64)
            >>> a
            VTBTensor([-0.063,  0.154, -0.061, -0.229, -0.880,  0.365,  0.009, -0.057, -0.076], dtype=torch.float64)
            >>> b
            VTBTensor([-0.259, -0.158, -0.167, -0.599,  0.171,  0.033, -0.406,  0.361, -0.443], dtype=torch.float64)
            >>> a.bundle(b)
            VTBTensor([-0.321, -0.003, -0.228, -0.827, -0.709,  0.398, -0.397,  0.304, -0.519], dtype=torch.float64)

        """
        return torch.add(self, other)

    def multibundle(self) -> "VTBTensor":
        """Bundle multiple hypervectors"""
        return torch.sum(self, dim=-2, dtype=self.dtype)

    def bind(self, other: "VTBTensor") -> "VTBTensor":
        r"""Bind the hypervector with other using the proposed binding method.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (VTB): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.VTBTensor.random(2, 9)
            >>> a
            VTBTensor([-0.173, -0.731,  0.026,  0.110,  0.326,  0.147,  0.105,  0.407, -0.343])
            >>> b
            VTBTensor([ 0.170,  0.049, -0.141,  0.873,  0.186,  0.262, -0.108,  0.171, 0.208])
            >>> a.bind(b)
            VTBTensor([-0.119, -0.485, -0.175,  0.024,  0.338,  0.129,  0.149,  0.133, -0.023])

            >>> a, b = torchhd.VTBTensor.random(2, 9, dtype=torch.float64)
            >>> a
            VTBTensor([-0.168,  0.244, -0.418,  0.236,  0.382, -0.369, -0.006, -0.621, 0.123], dtype=torch.float64)
            >>> b
            VTBTensor([-0.284,  0.464,  0.021, -0.459, -0.414, -0.346,  0.241, -0.083, 0.372], dtype=torch.float64)
            >>> a.bind(b)
            VTBTensor([ 0.263,  0.209, -0.374,  0.177, -0.240, -0.194, -0.491,  0.376, 0.166], dtype=torch.float64)

        """
        sqrt_d = int(math.sqrt(self.size(-1)))

        # Reshape the individual vectors as square matrices
        shape1 = list(self.shape)[:-1] + [1, sqrt_d, sqrt_d]
        # Copy each matrix sqrt_d times
        expand = [-1 for _ in shape1]
        expand[-3] = sqrt_d
        # Combine the batch dimensions and the matrix copies
        batches = math.prod(shape1[:-3])
        shape2 = [sqrt_d * batches, sqrt_d, sqrt_d]
        vy = other.reshape(*shape1).expand(*expand).reshape(shape2)

        x = self.unfold(-1, sqrt_d, sqrt_d).reshape(sqrt_d * batches, sqrt_d, 1)

        # Efficient batched block-diagonal matrix-vector multiply
        output = math.sqrt(sqrt_d) * torch.bmm(vy, x).reshape_as(self)
        return output

    def inverse(self) -> "VTBTensor":
        r"""Inversion of the hypervector for binding.

        For VTB the inverse is an approximate opperation.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.VTBTensor.random(1, 9)

            >>> a
            VTBTensor([[-0.117,  0.322, -0.747, -0.196, -0.059,  0.103, -0.099,  0.389, -0.332]])
            >>> a.inverse()
            VTBTensor([[-0.117, -0.196, -0.099,  0.322, -0.059,  0.389, -0.747,  0.103, -0.332]])

            >>> a = torchhd.VTBTensor.random(1, 9, dtype=torch.float64)
            >>> a
            VTBTensor([[-0.307, -0.019, -0.105,  0.174, -0.316,  0.366, -0.675, -0.147, 0.391]], dtype=torch.float64)
            >>> a.inverse()
            VTBTensor([[-0.307,  0.174, -0.675, -0.019, -0.316, -0.147, -0.105,  0.366, 0.391]], dtype=torch.float64)

        """
        sqrt_d = int(math.sqrt(self.size(-1)))

        # Change only the view of the last dimension, not the batch dimensions
        shape = list(self.shape)[:-1] + [sqrt_d, sqrt_d]
        return self.reshape(*shape).transpose(-2, -1).reshape_as(self)

    def negative(self) -> "VTBTensor":
        r"""Negate the hypervector for the bundling inverse.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.VTBTensor.random(1, 9)
            >>> a
            VTBTensor([[ 0.1692, -0.0420,  0.1477,  0.4790, -0.3863, -0.2593,  0.3213, -0.1168, -0.6205]])
            >>> a.negative()
            VTBTensor([[-0.1692,  0.0420, -0.1477, -0.4790,  0.3863,  0.2593, -0.3213, 0.1168,  0.6205]])

            >>> a = torchhd.VTBTensor.random(1, 9, dtype=torch.float64)
            >>> a
            VTBTensor([[ 0.2905,  0.2212,  0.2049,  0.1753, -0.5277, -0.6216, -0.2118, -0.2753, -0.0915]], dtype=torch.float64)
            >>> a.negative()
            VTBTensor([[-0.2905, -0.2212, -0.2049, -0.1753,  0.5277,  0.6216,  0.2118, 0.2753,  0.0915]], dtype=torch.float64)

        """
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "VTBTensor":
        r"""Permute the hypervector.

        The permutation operator is used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.VTBTensor.random(1, 9)
            >>> a
            VTBTensor([[ 0.1913, -0.5592,  0.3474, -0.1385,  0.5625,  0.0223, -0.1456, 0.2509, -0.3313]])
            >>> a.permute()
            VTBTensor([[-0.3313,  0.1913, -0.5592,  0.3474, -0.1385,  0.5625,  0.0223, -0.1456,  0.2509]])

            >>> a = torchhd.VTBTensor.random(1, 9, dtype=torch.float64)
            >>> a
            VTBTensor([[ 0.6705,  0.1316,  0.2595,  0.1193, -0.4884, -0.4052,  0.0730, -0.2022,  0.0513]], dtype=torch.float64)
            >>> a.permute()
            VTBTensor([[ 0.0513,  0.6705,  0.1316,  0.2595,  0.1193, -0.4884, -0.4052, 0.0730, -0.2022]], dtype=torch.float64)

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def dot_similarity(self, others: "VTBTensor") -> Tensor:
        """Inner product with other hypervectors"""
        if others.dim() >= 2:
            others = others.transpose(-2, -1)

        return torch.matmul(self, others)

    def cosine_similarity(self, others: "VTBTensor", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        self_dot = torch.sum(self * self, dim=-1)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(others * others, dim=-1)
        others_mag = torch.sqrt(others_dot)

        if self.dim() >= 2:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
        else:
            magnitude = self_mag * others_mag

        if torch.isclose(magnitude, torch.zeros_like(magnitude), equal_nan=True).any():
            import warnings

            warnings.warn("The norm of a vector is nearly zero, this could indicate a bug.")

        magnitude = torch.clamp(magnitude, min=eps)
        return self.dot_similarity(others) / magnitude
