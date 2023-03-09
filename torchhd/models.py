import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init

import torchhd.embeddings
import torchhd.functional as functional
import numpy as np

__all__ = [
    "Centroid",
]


class Centroid(nn.Module):
    r"""Implements the centroid classification model using class prototypes.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

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
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
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
        self.weight = Parameter(weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cos_similarity(input, self.weight)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

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
        # print(logit)
        # print(logit.argmax(1))

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        result = torch.where(select, -1, +1).to()
        # print(result.sum())
        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return
        # print(input)
        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * input)

    @torch.no_grad()
    def add_online2(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

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
        # print(logit)
        # print(logit.argmax(1))

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        result = torch.where(select, -1, +1).to()
        # print(result.sum())
        # cancel update if all predictions were correct
        self.similarity_sum += logit.max(1).values.item()
        self.count += 1

        if self.error_count == 0:
            val = self.similarity_sum/self.count
        else:
            val = (self.error_similarity_sum/self.error_count)-(self.error_similarity_sum/self.error_count)*0.01
        #print(self.similarity_sum/self.count)
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                self.weight.index_add_(0, target, input)
            return
        # print(input)
        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()
        #print('Total',self.similarity_sum / self.count)
        #print('Err',self.error_similarity_sum/self.error_count)

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)


        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)

        '''

        for i in range(self.out_features):
            if i != target.item():
                if i in logit.topk(2).indices and abs(logit[0][pred].item()-logit[0][i].item()) > 7:
                #print(pred.unsqueeze(1), torch.tensor([i]).unsqueeze(1))
                    #print(logit[0][i].item(), self.similarity_sum/self.count, abs(logit[0][pred].item()-logit[0][i].item()))
                    alpha2 = logit.gather(1, torch.tensor([i]).unsqueeze(1)) - 1.0
                    self.weight.index_add_(0, torch.tensor(i), lr * alpha2 * input)
'''



    @torch.no_grad()
    def add_online3(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

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
        # print(logit)
        # print(logit.argmax(1))

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        result = torch.where(select, -1, +1).to()
        # print(result.sum())
        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < 0.8:
                alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
                #print(alpha1)
                self.weight.index_add_(0, target, lr * alpha1 * input)
            return
        # print(input)
        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, lr * alpha1 * input)
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

    def regen_continuous(self, weight, x, size, classes):
        # F.normalize(self.weight.data)
        weight[x, :] = torch.normal(0, 1, size=(1, size))
        self.weight[:, x] = torch.zeros((1, classes))

    @torch.no_grad()
    def regenerate_continuous(self, weight, drop_rate, classes) -> None:
        dimensions = weight.shape[0]
        indices = torch.topk(
            1 / torch.var(self.weight, dim=0), int(dimensions * drop_rate)
        ).indices
        size = weight.shape[1]
        for i in indices:
            self.regen_continuous(weight, i, size, classes)

    def regen_reset(self, weight, x, size):
        weight[x, :] = torch.normal(0, 1, size=(1, size))

    @torch.no_grad()
    def regenerate_reset(self, weight, drop_rate) -> None:
        dimensions = weight.shape[0]
        indices = torch.topk(
            1 / torch.var(self.weight, dim=0), int(dimensions * drop_rate)
        ).indices
        size = weight.shape[1]
        for i in indices:
            self.regen_reset(weight, i, size)


class MemoryModel(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        type="projection",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MemoryModel, self).__init__()
        self.type = type
        self.in_features = in_features
        self.out_features = out_features

        if type == "projection":
            self.classes = torchhd.embeddings.Projection(in_features, out_features)
        elif type == "sinusoid":
            self.classes = torchhd.embeddings.Sinusoid(in_features, out_features)
        elif type == "density":
            self.classes = torchhd.embeddings.Density(in_features, out_features)
        elif type == "hashmap":
            self.classes = torchhd.embeddings.Random(out_features, in_features)
        '''
        self.weight = torch.empty((in_features, in_features), **factory_kwargs)
        '''

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        input = torch.matmul(input, self.weight).sign()
        if dot:
            return functional.dot_similarity(input, self.classes.weight)

        return functional.cos_similarity(input, self.classes.weight)

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        label = torch.index_select(self.classes.weight, 0, target)
        input = torch.matmul(input.T, label)
        self.weight += input
        # self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        iinput = torch.matmul(input, self.weight)
        predictions = functional.cos_similarity(iinput, self.classes.weight)

        if np.argmax(predictions).item() != target.item():
            label = torch.index_select(self.classes.weight, 0, target)
            self.weight += torch.matmul(input.T, label)

        """
        else:
            top_two_pred = predictions.topk(2)
            if abs(top_two_pred[0][0][0].item()/top_two_pred[0][0][1].item()) > 1.1:
                print(abs(top_two_pred[0][0][0].item()/top_two_pred[0][0][1].item()))
                label = torch.index_select(self.classes.weight, 0, target)
                self.weight += torch.matmul(input.T, label)
            #label = torch.index_select(self.classes.weight, 0, target)
            #self.weight += torch.matmul(input.T, label)
        """

    @torch.no_grad()
    def add_online2(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        iinput = torch.matmul(input, self.weight)
        predictions = functional.cos_similarity(iinput, self.classes.weight)
        # print(predictions[0][target.item()])
        # print(predictions[0][target.item()])
        top_two_pred = predictions.topk(2)
        print(abs(top_two_pred[0][0][0].item() - top_two_pred[0][0][1].item()))

        if np.argmax(predictions).item() != target.item():
            if (
                predictions[0][target.item()].item() < 0.7
                or abs(top_two_pred[0][0][0].item() - top_two_pred[0][0][1].item())
                > 0.1
            ):
                label = torch.index_select(self.classes.weight, 0, target)
                self.weight += torch.matmul(input.T, label)
        else:
            if predictions[0][target.item()].item() < 0.5:
                if (
                    abs(top_two_pred[0][0][0].item() - top_two_pred[0][0][1].item())
                    > 0.1
                ):
                    label = torch.index_select(self.classes.weight, 0, target)
                    self.weight += torch.matmul(input.T, label)
            """
            if np.argmax(predictions).item() != target.item():
                #print(predictions[0][target.item()])
                label = torch.index_select(self.classes.weight, 0, target)
                self.weight += torch.matmul(input.T, label)
            else:
                self.weight += torch.matmul(input.T, label)

                print(predictions[0][target.item()])
                """

    def add_online3(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        iinput = torch.matmul(input, self.weight)
        logit = functional.cos_similarity(iinput, self.classes.weight)
        pred = logit.argmax(1)
        is_wrong = target != pred
        # print(logit)
        # print(logit.argmax(1))

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        result = torch.where(select, -1, +1).to()
        # print(result.sum())
        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # print(input)
        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * input)
