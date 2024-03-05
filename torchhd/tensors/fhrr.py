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
import math
import torch
from typing import Set
from torch import Tensor
import torch.nn.functional as F

from torchhd.tensors.base import VSATensor

type_conversion = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


class FHRRTensor(VSATensor):
    """Fourier Holographic Reduced Representation

    Proposed in `Holographic Reduced Representation: Distributed Representation for Cognitive Structures <https://philpapers.org/rec/PLAHRR/>`_, this model uses complex phaser hypervectors.
    """

    supported_dtypes: Set[torch.dtype] = {torch.complex64, torch.complex128}

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
    ) -> "FHRRTensor":
        """Creates a set of hypervectors representing empty sets.

        When bundled with a random-hypervector :math:`x`, the result is :math:`x`.
        The empty vector of the FHRR model is a set of 0 values in both real and imaginary part.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is torch.complex64.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.FHRRTensor.empty(3, 6)
            FHRR([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

            >>> torchhd.FHRRTensor.empty(3, 6, dtype=torch.complex128)
            FHRR([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                   dtype=torch.complex128)
        """
        if dtype is None:
            dtype = torch.complex64

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
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
    ) -> "FHRRTensor":
        """Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is torch.complex64.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.FHRRTensor.identity(3, 6)
            FHRR([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                    [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                    [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]])

            >>> torchhd.FHRRTensor.identity(3, 6, dtype=torch.complex128)
            FHRR([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                    [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
                    [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]],
                   dtype=torch.complex128)


        """
        if dtype is None:
            dtype = torch.complex64

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
        dtype=torch.complex64,
        device=None,
        requires_grad=False,
        generator=None,
    ) -> "FHRRTensor":
        """Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``angle`` between -pi and +pi.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` is torch.complex64.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        Examples::

            >>> torchhd.FHRRTensor.random(3, 6)
            FHRR([[ 0.2082-0.9781j,  0.7038-0.7104j, -0.6297-0.7768j, -0.9632-0.2689j, -0.4941+0.8694j,  0.9771-0.2128j],
                    [ 0.9820-0.1887j, -0.0395+0.9992j, -0.5139+0.8579j, -0.8415-0.5402j, -0.6696-0.7427j,  0.2312+0.9729j],
                    [-0.9786+0.2057j,  0.1714+0.9852j, -0.5925-0.8056j, -0.5698-0.8218j, -0.4632-0.8863j,  0.6996-0.7145j]])

            >>> torchhd.FHRRTensor.random(3, 6, dtype=torch.long)
            FHRR([[-0.9996-0.0285j, -0.0688-0.9976j,  0.6900-0.7238j,  0.9519-0.3064j, 0.8131-0.5821j,  0.9942-0.1077j],
                    [-0.9199-0.3922j,  0.8073-0.5902j,  0.8683+0.4960j,  0.1250+0.9922j, 0.6248+0.7808j, -0.2495+0.9684j],
                    [ 0.0178+0.9998j, -0.3006-0.9538j, -0.9346+0.3557j,  0.9017-0.4324j, 0.4029-0.9153j,  0.4818-0.8763j]],
                    dtype=torch.complex128)

        """
        if dtype is None:
            dtype = torch.complex64

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        dtype = type_conversion[dtype]

        size = (num_vectors, dimensions)
        angle = torch.empty(size, dtype=dtype, device=device)
        angle.uniform_(-math.pi, math.pi, generator=generator)

        result = torch.complex(angle.cos(), angle.sin())
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "FHRRTensor") -> "FHRRTensor":
        r"""Bundle the hypervector with other using element-wise sum.

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (FHRR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.FHRRTensor.random(2, 6)
            >>> a
            FHRR([ 0.9556-0.2948j,  0.1746+0.9846j, -0.6270-0.7790j, -0.2423-0.9702j, 0.6358+0.7719j,  0.9965-0.0834j])
            >>> b
            FHRR([-0.9539-0.3000j, -0.1279+0.9918j, -0.4610+0.8874j, -0.3638-0.9315j, 0.9554+0.2952j,  0.8659+0.5003j])
            >>> a.bundle(b)
            FHRR([-1.6885+0.4104j, -0.4094-1.4874j,  0.0090-0.0058j,  0.1039-0.9365j, 0.0413-1.8657j,  0.6276+1.8385j])

            >>> a, b = torchhd.FHRRTensor.random(2, 10, dtype=torch.complex128)
            >>> a
            FHRR([ 0.4521-0.8920j,  0.7917-0.6109j,  0.5414-0.8408j, -0.9550-0.2967j, 0.9320+0.3626j, -0.8509-0.5253j],
            dtype=torch.complex128)
            >>> b
            FHRR([ 0.6954-0.7186j, -0.5621-0.8270j,  0.4685+0.8835j, -0.9319+0.3627j, -0.8310-0.5563j,  0.2545+0.9671j],
            dtype=torch.complex128)
            >>> a.bundle(b)
            FHRR([ 1.1475-1.6106j,  0.2296-1.4379j,  1.0099+0.0427j, -1.8869+0.0660j, 0.1010-0.1937j, -0.5964+0.4417j],
            dtype=torch.complex128)

        """
        return torch.add(self, other)

    def multibundle(self) -> "FHRRTensor":
        """Bundle multiple hypervectors"""
        return torch.sum(self, dim=-2, dtype=self.dtype)

    def bind(self, other: "FHRRTensor") -> "FHRRTensor":
        r"""Bind the hypervector with other using element-wise multiplication.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (FHRR): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.FHRRTensor.random(2, 6)
            >>> a
            FHRR([ 0.9317-0.3632j,  0.7320+0.6813j, -0.8588+0.5123j, -0.9723-0.2339j, -0.9631-0.2692j, -0.4093-0.9124j])
            >>> b
            FHRR([ 0.9983+0.0578j, -0.5043-0.8635j,  0.5505-0.8349j,  0.9805+0.1966j, 0.9656+0.2599j, -0.5609-0.8279j])
            >>> a.bind(b)
            FHRR([ 0.9511-0.3087j,  0.2191-0.9757j, -0.0450+0.9990j, -0.9073-0.4204j, -0.8600-0.5102j, -0.5257+0.8507j])

            >>> a, b = torchhd.FHRRTensor.random(2, 6, dtype=torch.complex128)
            >>> a
            FHRR([ 0.7838-0.6210j, -0.0258+0.9997j,  0.0263+0.9997j,  0.9617+0.2742j, 0.1281-0.9918j, -0.4321+0.9018j],
            dtype=torch.complex128)
            >>> b
            FHRR([-0.9995+0.0308j,  0.4550-0.8905j,  0.2793+0.9602j,  0.0025-1.0000j, 0.4470+0.8946j,  0.8314-0.5557j],
            dtype=torch.complex128)
            >>> a.bind(b)
            FHRR([-0.7643+0.6449j,  0.8785+0.4778j, -0.9525+0.3045j,  0.2766-0.9610j, 0.9444-0.3287j,  0.1419+0.9899j],
            dtype=torch.complex128)

        """
        return torch.mul(self, other)

    def multibind(self) -> "FHRRTensor":
        """Bind multiple hypervectors"""
        return torch.prod(self, dim=-2, dtype=self.dtype)

    def inverse(self) -> "FHRRTensor":
        r"""Invert the hypervector for binding.

        For FHRR the inverse of hypervector is its conjugate, this returns the conjugate of the hypervector.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.FHRRTensor.random(1, 6)
            >>> a
            FHRR([[ 0.9855+0.1698j, -0.0927-0.9957j, -0.8316-0.5554j, -0.1433-0.9897j, 0.8328+0.5536j,  0.2071+0.9783j]])
            >>> a.inverse()
            FHRR([[ 0.9855-0.1698j, -0.0927+0.9957j, -0.8316+0.5554j, -0.1433+0.9897j, 0.8328-0.5536j,  0.2071-0.9783j]])

            >>> a = torchhd.FHRRTensor.random(1, 6, dtype=torch.complex128)
            >>> a
            FHRR([[-0.9983-0.0574j, -0.4825+0.8759j,  0.9631-0.2692j,  0.9066-0.4219j, 0.7099-0.7044j, -0.1313-0.9913j]],
            dtype=torch.complex128)
            >>> a.inverse()
            >>> a.inverse()
            FHRR([[-0.9983+0.0574j, -0.4825-0.8759j,  0.9631+0.2692j,  0.9066+0.4219j, 0.7099+0.7044j, -0.1313+0.9913j]],
            dtype=torch.complex128)

        """
        # Resolve conj to ensure the the returned tensor does not share the same memory
        return torch.conj(self).resolve_conj()

    def negative(self) -> "FHRRTensor":
        r"""Negate the hypervector for the bundling inverse.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.FHRRTensor.random(1, 6)
            >>> a
            FHRR([[-0.0187-0.9998j,  0.1950-0.9808j,  0.5203+0.8540j,  0.8587+0.5124j, 0.9998+0.0203j, -0.6237-0.7816j]])
            >>> a.negative()
            FHRR([[ 0.0187+0.9998j, -0.1950+0.9808j, -0.5203-0.8540j, -0.8587-0.5124j, -0.9998-0.0203j,  0.6237+0.7816j]])

            >>> a = torchhd.FHRRTensor.random(1, 6, dtype=torch.complex128)
            >>> a
            FHRR([[ 0.8255+0.5644j, -0.8352-0.5500j,  0.9751-0.2218j, -0.9808-0.1950j, -0.3840-0.9233j,  0.4106-0.9118j]],
            dtype=torch.complex128)
            >>> a.negative()
            FHRR([[-0.8255-0.5644j,  0.8352+0.5500j, -0.9751+0.2218j,  0.9808+0.1950j, 0.3840+0.9233j, -0.4106+0.9118j]], dtype=torch.complex128)

        """
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "FHRRTensor":
        r"""Permute the hypervector.

        The permutation operator is used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a = torchhd.FHRRTensor.random(1, 6)
            >>> a
            FHRR([[-0.3286-0.9445j,  0.2161-0.9764j, -0.6484+0.7613j, -0.4020+0.9156j, 0.8282-0.5605j, -0.9869+0.1613j]])
            >>> a.permute()
            FHRR([[-0.9869+0.1613j, -0.3286-0.9445j,  0.2161-0.9764j, -0.6484+0.7613j, -0.4020+0.9156j,  0.8282-0.5605j]])

            >>> a = torchhd.FHRRTensor.random(1, 6, dtype=torch.complex128)
            >>> a
            FHRR([[-0.9500-0.3123j, -0.0234+0.9997j, -0.1071-0.9943j, -0.8558-0.5174j, 0.9631-0.2690j,  0.5470-0.8371j]],
            dtype=torch.complex128)
            >>> a.permute()
            FHRR([[ 0.5470-0.8371j, -0.9500-0.3123j, -0.0234+0.9997j, -0.1071-0.9943j, -0.8558-0.5174j,  0.9631-0.2690j]],
            dtype=torch.complex128)

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def dot_similarity(self, others: "FHRRTensor") -> Tensor:
        """Inner product with other hypervectors"""
        if others.dim() >= 2:
            others = others.transpose(-2, -1)

        return torch.real(torch.matmul(self, torch.conj(others)))

    def cosine_similarity(self, others: "FHRRTensor", *, eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        self_dot = torch.sum(torch.real(self * torch.conj(self)), dim=-1)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(torch.real(others * torch.conj(others)), dim=-1)
        others_mag = torch.sqrt(others_dot)

        if self.dim() >= 2:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
        else:
            magnitude = self_mag * others_mag

        if torch.isclose(magnitude, torch.zeros_like(magnitude), equal_nan=True).any():
            import warnings

            warnings.warn(
                "The norm of a vector is nearly zero, this could indicate a bug."
            )

        magnitude = torch.clamp(magnitude, min=eps)
        return self.dot_similarity(others) / magnitude
