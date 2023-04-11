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
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.utils.data as data
from tqdm import tqdm


import torchhd.functional as functional
import torchhd.datasets as datasets
import torchhd.embeddings as embeddings


__all__ = [
    "Centroid",
    "IntRVFL",
]


class Centroid(nn.Module):
    r"""Implements the centroid classification model using class prototypes.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of the output, typically the number of classes.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shape:
        - Input: :math:`(*, d)` where :math:`*` means any number of
          dimensions including none and ``d = in_features``.
        - Output: :math:`(*, n)` where all but the last dimension
          are the same shape as the input and ``n = out_features``.

    Attributes:
        weight: the trainable weights, or class prototypes, of the module of shape
            :math:`(n, d)`. The values are initialized as all zeros.

    Examples::

        >>> m = Centroid(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> output.size()
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Centroid, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cosine_similarity(input, self.weight)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_adapt(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        self.weight.index_add_(0, target, input, alpha=lr)

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, pred, -input)


    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = 1.0 - logit.gather(1, pred.unsqueeze(1))

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * -input)


    @torch.no_grad()
    def add_adjust(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                self.weight.index_add_(0, target, input)
            return

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1
        self.weight.index_add_(0, pred, lr * alpha2 * input)


    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features is not None
        )


class IntRVFL(nn.Module):
    r"""Class implementing integer random vector functional link network (intRVFL) model as described in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        in_features (int): Size of each input sample.
        dimensions (int): The number of hidden dimensions to use.
        out_features (int): The number of output features, typically the number of classes.
        kappa (int, optional): Parameter of the clipping function limiting the range of values; used as the part of transforming input data.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shape:
        - Input: :math:`(*, d)` where :math:`*` means any number of
          dimensions including none and ``d = in_features``.
        - Output: :math:`(*, n)` where all but the last dimension
          are the same shape as the input and ``n = out_features``.

    Attributes:
        weight: the trainable weights, or class prototypes, of the module of shape
            :math:`(n, d)`. The values are initialized as all zeros.

    """

    __constants__ = ["in_features", "dimensions", "out_features", "kappa"]
    in_features: int
    dimensions: int
    out_features: int
    kappa: Optional[int]
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        dimensions: int,
        out_features: int,
        kappa: Optional[int] = None,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(IntRVFL, self).__init__()

        self.in_features = in_features
        self.dimensions = dimensions
        self.out_features = out_features
        self.kappa = kappa

        self.encoding = embeddings.Density(
            in_features, self.dimensions, **factory_kwargs
        )

        weight = torch.empty((out_features, dimensions), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def encode(self, x):
        encodings = self.encoding(x)

        if self.kappa is not None:
            encodings = encodings.clipping(self.kappa)

        return encodings

    def forward(self, x):
        # Make encodings for all data samples in the batch
        encodings = self.encode(x)

        # Get similarity values for each class
        return functional.dot_similarity(encodings, self.weight)

    # Train the model
    @torch.no_grad()
    def fit_ridge_regression(
        self,
        samples: Tensor,
        labels: Tensor,
        alpha: Optional[float] = 1,
    ) -> None:
        r"""Compute the weights (readout matrix) with :func:`~torchhd.ridge_regression`.

        It is a common way to form classifiers wihtin randomized neural networks see, e.g., `Randomness in Neural Networks: An Overview  <https://doi.org/10.1002/widm.1200>`_.

        Args:
            samples (Tensor): The feature vectors.
            labels (LongTensor): The targets vector, typically the class of each sample.
            alpha (float, optional): Scalar for the variance of the samples. Default is 1.

        Shapes:
           - Samples: :math:`(n, f)`
           - Labels: :math:`(n, c)`

        """
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
        n = labels.size(0)

        # Transform to hypervector representations
        encodings = self.encode(samples)

        # Transform classes to one-hot encoding
        one_hot_labels = torch.zeros(n, self.out_features, **factory_kwargs)
        one_hot_labels[torch.arange(n), labels] = 1

        # Compute the readout matrix using the ridge regression
        weights = functional.ridge_regression(encodings, one_hot_labels, alpha=alpha)
        # Assign the obtained classifier to the output
        self.weight.copy_(weights)
