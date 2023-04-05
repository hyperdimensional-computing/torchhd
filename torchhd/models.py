import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torchmetrics

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
        self,
        in_features: int,
        out_features: int,
        init_samples=0,
        device=None,
        dtype=None,
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
        weight_base = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight)
        self.weight_base = Parameter(weight_base)
        self.reset_parameters()
        self.train_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=out_features
        )
        self.train_accuracy_base = torchmetrics.Accuracy(
            "multiclass", num_classes=out_features
        )
        self.added_samples = 0
        self.init_samples = init_samples

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(
        self, input: Tensor, dot: bool = False, combined=False, test=False
    ) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cos_similarity(input, self.weight)

    def forward2(
        self, input: Tensor, dot: bool = False, combined=False, test=False
    ) -> Tensor:
        if dot:
            return functional.dot_similarity(
                input, self.weight
            ), functional.dot_similarity(input, self.weight_base)
        return functional.cos_similarity(input, self.weight), functional.cos_similarity(
            input, self.weight_base
        )
        # return functional.cos_similarity(input, self.weight)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)
        logit = self(input)
        return logit

    def mutate_hv(self, input, target, w=1, lr=1):
        idx = torch.nonzero(
            torch.tensor(torchhd.hard_quantize(self.weight[target]) != input),
            as_tuple=True,
        )[1]
        a = torch.zeros((1, self.in_features))
        a[0][idx] = input[0][idx]
        self.weight.index_add_(0, target, a * w)

    @torch.no_grad()
    def add_data_augmentation(
        self, input: Tensor, target: Tensor, lr: float = 1.0
    ) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        for i in range(1):
            self.mutate_hv(input, target)
        logit = self(input)
        pred = logit.argmax(1)
        return pred

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
        l = logit
        pred = logit.argmax(1)
        is_wrong = target != pred
        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)

        if is_wrong.sum().item() == 0:
            return l

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        return l

    @torch.no_grad()
    def add_online2(
        self, input: Tensor, target: Tensor, rep: int = 0, lr: float = 1.0
    ) -> None:
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
        l = logit

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        if is_wrong.sum().item() == 0:
            # print(val, logit.max(1).values.item(), 0.05  > abs(logit.max(1).values.item()-logit[0][1-logit.argmax(1).item()]).item())

            if logit.max(1).values.item() < val:
                self.weight.index_add_(0, target, input)
            # else:
            #    print(logit, val)
            return l

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)
        # print("alpha 1", alpha1, logit.gather(1, target.unsqueeze(1)), "alpha 2", alpha2)

        return l

    @torch.no_grad()
    def add_online2_init(
        self, input: Tensor, target: Tensor, rep: int = 0, lr: float = 1.0
    ) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        if self.added_samples < self.init_samples:
            self.added_samples += 1
            self.weight.index_add_(0, target, input, alpha=lr)
            logit = self(input)
            return logit

        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred
        l = logit

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        if is_wrong.sum().item() == 0:
            # print(val, logit.max(1).values.item(), 0.05  > abs(logit.max(1).values.item()-logit[0][1-logit.argmax(1).item()]).item())

            if logit.max(1).values.item() < val:
                self.weight.index_add_(0, target, input)
            # else:
            #    print(logit, val)
            return l

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)
        # print("alpha 1", alpha1, logit.gather(1, target.unsqueeze(1)), "alpha 2", alpha2)

        return l

    @torch.no_grad()
    def add_online_combined(
        self, input: Tensor, target: Tensor, rep: int = 0, lr: float = 1.0
    ) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py

        logit, logit_base = self.forward2(input)
        pred = logit.argmax(1)
        pred_base = logit_base.argmax(1)
        is_wrong = target != pred
        is_wrong_base = target != pred_base
        l = logit
        l_base = logit_base
        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        self.train_accuracy(logit, target)
        self.train_accuracy_base(logit_base, target)
        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count

        if is_wrong.sum().item() == 0:
            # print(val, logit.max(1).values.item(), 0.05  > abs(logit.max(1).values.item()-logit[0][1-logit.argmax(1).item()]).item())

            if logit.max(1).values.item() < val:
                # or 0.05 > abs(logit.max(1).values.item()-logit[0][1-logit.argmax(1).item()]).item():
                self.weight.index_add_(0, target, input)
            # else:
            #    print(logit, val)
            return l, l_base

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        # logit = logit[is_wrong]
        # logit_base = logit_base[is_wrong_base]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)

        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)

        alpha1_base = 1.0 - logit_base.gather(1, target.unsqueeze(1))
        self.weight_base.index_add_(0, target, lr * alpha1_base * input)

        alpha2_base = logit_base.gather(1, pred_base.unsqueeze(1)) - 1.0
        self.weight_base.index_add_(0, pred_base, lr * alpha2_base * input)
        # print("alpha 1", alpha1, logit.gather(1, target.unsqueeze(1)), "alpha 2", alpha2)

        return l, l_base

    @torch.no_grad()
    def add_online_noise(
        self, input: Tensor, target: Tensor, rep: int = 0, lr: float = 1.0
    ) -> None:
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
        l = logit

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1

        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count

        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                self.mutate_hv(input, target)
            return l

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)
        return l

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
        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        target = target[is_wrong]
        self.mse += (1.0 - logit.gather(1, target.unsqueeze(1)))[0][0].item()

        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        # print(self.similarity_sum/self.count)
        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                # self.weight.index_add_(0, target, input)
                self.weight.index_add_(0, target, lr * alpha1 * input)

            return pred
        # print(input)
        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()
        # print('Total',self.similarity_sum / self.count)
        # print('Err',self.error_similarity_sum/self.error_count)

        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)
        return pred

    @torch.no_grad()
    def add_online4(
        self, input: Tensor, target: Tensor, rep=0, lr: float = 1.0
    ) -> None:
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
        p = pred
        t = target
        ii = input
        tt = target
        # print(logit.argmax(1))
        loss = self.criterion(logit, target)

        select = torch.empty(10000, dtype=torch.bool)
        select.bernoulli_(0.1)
        result = torch.where(select, -1, +1).to()
        # print(result.sum())
        # cancel update if all predictions were correct

        self.mse += (1.0 - logit.gather(1, target.unsqueeze(1)))[0][0].item()

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1

        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        # print(self.similarity_sum/self.count)
        if pred.item() not in self.conf_pred_class:
            self.conf_pred_class[pred.item()] = 0
            self.conf_pred_class_count[pred.item()] = 0

        if is_wrong.sum().item() == 0:
            if self.conf_pred_class_count[pred.item()] == 0 or logit.max(
                1
            ).values.item() < (
                self.conf_pred_class[pred.item()]
                / self.conf_pred_class_count[pred.item()]
            ):
                self.weight.index_add_(0, target, input * (rep + 1))
                self.conf_pred_class[pred.item()] += logit.max(1).values.item()
                self.conf_pred_class_count[pred.item()] += 1
                # if rep < 5:
                #    self.add_online4(ii, tt, rep=rep+1)

            return pred
        # print(input)
        # only update wrongly predicted inputs

        if pred.item() not in self.miss_predict:
            self.miss_predict[pred.item()] = 0
        self.miss_predict[pred.item()] += 1

        # print("pred", logit.max(1).values.item())

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * (rep + 1) * alpha1 * input)

        # if pred.item() != t.item() and not torch.all(self.weight[pred.item()] == 0):
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0
        self.weight.index_add_(0, pred, lr * alpha2 * input)
        """
        for i in range(self.out_features):
            if i != t.item() and not torch.all(self.weight[i] == 0):
                alpha = logit.gather(1, torch.tensor([[i]])) - 1.0
                self.weight.index_add_(0, torch.tensor(i), lr * alpha * input)

        """
        # if rep < 5:
        #    self.add_online4(ii, tt, rep=rep+1)

        return p

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

        norms = self.weight_base.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight_base.div_(norms)

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

        self.classes = torchhd.embeddings.Projection(in_features, out_features)
        self.weight = torch.empty((in_features, in_features), **factory_kwargs)

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
