from typing import Set
import torch
from torch import Tensor
from torch.fft import fft, ifft
import math

from torchhd.tensors.base import VSATensor


class HRRTensor(VSATensor):
    """Holographic Reduced Representation

    Proposed in `Holographic reduced representations <https://ieeexplore.ieee.org/document/377968>`_, this model uses real valued hypervectors.
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
    ) -> "HRRTensor":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a random-hypervector :math:`x`, the result is :math:`x`.
        The empty vector of the HRR model is simply a set of 0 values.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.HRRTensor.empty(3, 6)
            HRR([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]])
            >>> torchhd.HRRTensor.empty(3, 6, dtype=torch.float64)
            HRR([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]], dtype=torch.float64)

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
    ) -> "HRRTensor":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.HRRTensor.identity(3, 6)
            HRR([[1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.]])
            >>> torchhd.HRRTensor.identity(3, 6, dtype=torch.float64)
            HRR([[1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.]], dtype=torch.float64)

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
        )
        result[:, 0] = 1
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
    ) -> "HRRTensor":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled at random from a normal with mean 0 and standard deviation 1/dimensions.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is ``torch.get_default_dtype()``.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.HRRTensor.random(3, 6)
            HRR([[ 0.2520, -0.0048, -0.0351,  0.2067,  0.0638, -0.0729],
                 [-0.2695,  0.0815,  0.0103,  0.2211, -0.1202,  0.2134],
                 [ 0.0086, -0.1748, -0.1715,  0.3215, -0.1353,  0.0044]])
            >>> torchhd.HRRTensor.random(3, 6, dtype=torch.float64)
            HRR([[-0.1327, -0.0396, -0.0065,  0.0886, -0.4665,  0.2656],
                 [-0.2879, -0.1070, -0.0851, -0.4366, -0.1311,  0.3976],
                 [-0.0472,  0.2987, -0.1567,  0.1496, -0.0098,  0.0344]], dtype=torch.float64)
        """

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        result = torch.empty(size, dtype=dtype, device=device)
        result.normal_(0, 1.0 / math.sqrt(dimensions), generator=generator)

        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "HRRTensor") -> "HRRTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (HRR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.HRRTensor.random(2, 6)
            >>> a
            HRR([ 0.1916, -0.1451, -0.0678,  0.0829,  0.3816, -0.0906])
            >>> b
            HRR([-0.2825,  0.3788,  0.0885, -0.1269, -0.0481, -0.3029])
            >>> a.bundle(b)
            HRR([-0.0909,  0.2336,  0.0207, -0.0440,  0.3336, -0.3935])

            >>> a, b = torchhd.HRRTensor.random(2, 6, dtype=torch.float64)
            >>> a
            HRR([ 0.3879, -0.0452, -0.0082, -0.2262, -0.2764,  0.0166], dtype=torch.float64)
            >>> b
            HRR([ 0.0738, -0.0306,  0.4948,  0.1209,  0.1482,  0.1268], dtype=torch.float64)
            >>> a.bundle(b)
            HRR([ 0.4618, -0.0758,  0.4866, -0.1053, -0.1281,  0.1434], dtype=torch.float64)

        """
        return torch.add(self, other)

    def multibundle(self) -> "HRRTensor":
        """Bundle multiple hypervectors"""
        return torch.sum(self, dim=-2, dtype=self.dtype)

    def bind(self, other: "HRRTensor") -> "HRRTensor":
        r"""Bind the hypervector with other using circular convolution.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (HRR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.HRRTensor.random(2, 6)
            >>> a
            HRR([ 0.0101, -0.2474, -0.0097, -0.0788,  0.1541, -0.1766])
            >>> b
            HRR([-0.0338,  0.0340,  0.0289, -0.1498,  0.1178, -0.2822])
            >>> a.bind(b)
            HRR([ 0.0786, -0.0260,  0.0591, -0.0706,  0.0799, -0.0216])

            >>> a, b = torchhd.HRRTensor.random(2, 6, dtype=torch.float64)
            >>> a
            HRR([ 0.0354, -0.0818,  0.0216,  0.0384,  0.2961,  0.1976], dtype=torch.float64)
            >>> b
            HRR([ 0.3640, -0.0640, -0.1033, -0.1454,  0.0999,  0.0299], dtype=torch.float64)
            >>> a.bind(b)
            HRR([-0.0362, -0.0910,  0.0114,  0.0445,  0.1244,  0.0388], dtype=torch.float64)

        """
        result = ifft(torch.mul(fft(self), fft(other)))
        return torch.real(result)

    def multibind(self) -> "HRRTensor":
        """Bind multiple hypervectors"""
        result = ifft(torch.prod(fft(self), dim=-2, dtype=self.dtype))
        return torch.real(result)

    def exact_inverse(self) -> "HRRTensor":
        """Unstable, but exact, inverse"""
        result = ifft(1.0 / torch.conj(fft(self)))
        result = torch.real(result)
        return torch.nan_to_num(result)

    def inverse(self) -> "HRRTensor":
        r"""Stable inversion of the hypervector for binding.

        For HRR the stable inverse of hypervector is its conjugate in the frequency domain, this returns the conjugate of the hypervector.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.HRRTensor.random(1, 6)
            >>> a
            HRR([[ 0.1406,  0.0014, -0.0502,  0.2888,  0.2969, -0.2637]])
            >>> a.inverse()
            HRR([[ 0.1406, -0.2637,  0.2969,  0.2888, -0.0502,  0.0014]])

            >>> a = torchhd.HRRTensor.random(1, 6, dtype=torch.float64)
            >>> a
            HRR([[ 0.0090,  0.2620,  0.0836,  0.0441, -0.2351, -0.1744]], dtype=torch.float64)
            >>> a.inverse()
            HRR([[ 0.0090, -0.1744, -0.2351,  0.0441,  0.0836,  0.2620]], dtype=torch.float64)

        """
        result = ifft(torch.conj(fft(self)))
        return torch.real(result)

    def negative(self) -> "HRRTensor":
        r"""Negate the hypervector for the bundling inverse.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.HRRTensor.random(1, 6)
            >>> a
            HRR([[ 0.2658, -0.2808,  0.1436,  0.1131,  0.1567, -0.1426]])
            >>> a.negative()
            HRR([[-0.2658,  0.2808, -0.1436, -0.1131, -0.1567,  0.1426]])

            >>> a = torchhd.HRRTensor.random(1, 6, dtype=torch.float64)
            >>> a
            HRR([[ 0.0318,  0.1944,  0.1229,  0.0193,  0.0135, -0.2521]], dtype=torch.float64)
            >>> a.negative()
            HRR([[-0.0318, -0.1944, -0.1229, -0.0193, -0.0135,  0.2521]], dtype=torch.float64)

        """
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "HRRTensor":
        r"""Permute the hypervector.

        The permutation operator is used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.HRRTensor.random(1, 6)
            >>> a
            HRR([[-0.2521,  0.1140, -0.1647, -0.1490, -0.2091, -0.0618]])
            >>> a.permute()
            HRR([[-0.0618, -0.2521,  0.1140, -0.1647, -0.1490, -0.2091]])

            >>> a = torchhd.HRRTensor.random(1, 6, dtype=torch.float64)
            >>> a
            HRR([[-0.0495, -0.0318,  0.3923, -0.3205,  0.1587,  0.1926]], dtype=torch.float64)
            >>> a.permute()
            HRR([[ 0.1926, -0.0495, -0.0318,  0.3923, -0.3205,  0.1587]], dtype=torch.float64)

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def dot_similarity(self, others: "HRRTensor") -> Tensor:
        """Inner product with other hypervectors"""
        if others.dim() >= 2:
            others = others.mT
        return torch.matmul(self, others)

    def cosine_similarity(self, others: "HRRTensor", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        self_dot = torch.sum(self * self, dim=-1)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(others * others, dim=-1)
        others_mag = torch.sqrt(others_dot)

        if self.dim() > 1:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(0)
        else:
            magnitude = self_mag * others_mag

        magnitude = torch.clamp(magnitude, min=eps)
        return self.dot_similarity(others) / magnitude
