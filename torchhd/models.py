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
import copy
import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.utils.data as data
from tqdm import tqdm
import torchmetrics
import torchhd
import torchhd.functional as functional
import torchhd.datasets as datasets
import torchhd.embeddings as embeddings


__all__ = ["Centroid", "CentroidMiss", "CentroidIterative", "IntRVFL", "PoolCentroid"]


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


        self.in_features = in_features # dimensions

        self.out_features = out_features
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)

        # QuantHD
        weight_quant = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight_quant = Parameter(weight_quant, requires_grad=requires_grad)

        # SparseHD
        weight_sparse = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight_sparse = Parameter(weight_sparse, requires_grad=requires_grad)

        # DistHD
        self.n_disthd = torch.empty((0, in_features))
        self.m_disthd = torch.empty((0, in_features))

        # MultiCentroidHD
        multi_weight = [torch.empty(1,in_features) for i in range(out_features)]
        self.multi_weight = [Parameter(tensor) for tensor in multi_weight]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        init.zeros_(self.weight_quant)
        init.zeros_(self.weight_sparse)
        for i in self.multi_weight:
            init.zeros_(i)

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

    def quantized_similarity(self, input, model):
        if model == 'binary':
            return functional.hamming_similarity(input, self.weight_quant).float()
        elif model == 'ternary':
            return functional.dot_similarity(input, self.weight_quant)

    def add_quantize(self, input: Tensor, target: Tensor, lr: float = 1.0, model = 'binary') -> None:
        logit = self.quantized_similarity(input, model)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, lr * input)
        self.weight.index_add_(0, pred, lr * -input)

    def binarize_model(self, model):
        if model == 'binary':
            self.weight_quant.data = torch.sign(self.weight.data)
        elif model == 'ternary':
            self.weight_quant.data = torch.where(self.weight.data > 0, torch.tensor(1.0),
                                                 torch.where(self.weight.data < 0, torch.tensor(-1.0),
                                                             torch.tensor(0.0)))

    def add_sparse(self, input: Tensor, target: Tensor, lr: float = 1.0, iter = 0) -> None:
        if iter == 0:
            logit = self(input)
        else:
            logit = self.sparse_similarity(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        if iter == 0:
            self.weight.index_add_(0, target, lr * input)
            self.weight.index_add_(0, pred, lr * -input)
        else:
            self.weight_sparse.index_add_(0, target, lr * input)
            self.weight_sparse.index_add_(0, pred, lr * -input)

    def sparse_similarity(self, input):
        return functional.cosine_similarity(input, self.weight_sparse)

    def sparsify_model(self, model, s, iter):
        if model == 'dimension':
            if iter == 0:
                max_vals, _ = torch.max(self.weight.data, dim=0)
                min_vals, _ = torch.min(self.weight.data, dim=0)
            else:
                max_vals, _ = torch.max(self.weight_sparse.data, dim=0)
                min_vals, _ = torch.min(self.weight_sparse.data, dim=0)
            variation = max_vals - min_vals
            _, dropped_indices = variation.topk(s, largest=False)

            if iter == 0:
                self.weight_sparse.data = self.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0
        if model == 'class':
            if iter == 0:
                _, dropped_indices = torch.topk(self.weight.abs(), k=s, dim=1, largest=False, sorted=True)
            else:
                _, dropped_indices = torch.topk(self.weight_sparse.abs(), k=s, dim=1, largest=False, sorted=True)
            if iter == 0:
                self.weight_sparse.data = self.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0

    def add_neural(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, lr * input)
        self.weight.index_add_(0, pred, lr * -input)

    def neural_regenerate(self, r, encode):
        max_vals, _ = torch.max(self.weight.data, dim=0)
        min_vals, _ = torch.min(self.weight.data, dim=0)

        variation = max_vals - min_vals
        _, dropped_indices = variation.topk(r, largest=False)
        
        self.weight.data[:, dropped_indices] = torch.randn(self.weight.size(0))
        encode.embed.weight[:, dropped_indices] = torch.randn(self.weight.size(0))

    @torch.no_grad()
    def add_dist(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
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
    def eval_dist(self, input: Tensor, target: Tensor, lr: float = 1.0, alpha = 1.0, beta = 1.0, theta = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        _, top_2 = torch.topk(logit, k=2)
        pred1 = top_2[0][0]
        pred2 = top_2[0][1]

        if pred2 == target:
            m1 = torch.abs(input - self.weight[pred1])
            m2 = torch.abs(input - self.weight[pred2])
            self.m_disthd = torch.cat((self.m_disthd, alpha*m1 - beta*m2), dim=0)
        if pred1 != target and pred2 != target:
            n1 = torch.abs(input - self.weight[pred2])
            n2 = torch.abs(input - self.weight[pred1])
            n3 = torch.abs(input - self.weight[target])
            self.n_disthd = torch.cat((self.m_disthd, alpha*n1 + beta*n2 - theta*n3), dim=0)

    def regenerate_dist(self, r, eps=1e-12):
        norms = self.m_disthd.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.m_disthd.div_(norms)

        norms = self.n_disthd.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.n_disthd.div_(norms)

        m_ = torch.sum(self.m_disthd, dim=0)
        n_ = torch.sum(self.n_disthd, dim=0)

        _, top_m_ = m_.topk(r, largest=True)
        _, top_n_ = n_.topk(r, largest=True)
        intersect = torch.unique(torch.cat((top_m_, top_n_), 0))

        dimensions_regenerated = intersect[torch.logical_and(torch.sum(top_m_ == intersect.unsqueeze(1), dim=0) > 0,
                                             torch.sum(top_n_ == intersect.unsqueeze(1), dim=0) > 0)]
        self.weight[:, dimensions_regenerated] = torch.randn(self.weight.size(0))

    def multi_similarity(self, input):
        return torch.cat([functional.cosine_similarity(input, i)[0] for i in self.multi_weight], dim=0)

    @torch.no_grad()
    def add_multi(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self.multi_similarity(input)

        pred = torch.argmax(logit, dim=0)
        row = 0
        col = 0
        for i in self.multi_weight:
            if i.shape[0] >= pred:
                col = i.shape[0] - pred
                break
            else:
                row += 1
                pred -= i.shape[0]
        if row == target:
            
            if self.multi_weight[row].shape[0] == col:
                col -= 1
            self.multi_weight[row][col] += input[0]
        else:
            self.multi_weight[target] = torch.cat([self.multi_weight[target], input], dim=0)
        return torch.tensor([row])

    def drop_classes(self, drop):
        concatenated = torch.cat([i for i in self.multi_weight], dim=0)
        abs_sum = torch.abs(concatenated).sum(dim=1)
        sorted_indices = torch.argsort(abs_sum)[:drop]
        sorted_indices, _ = torch.sort(sorted_indices, descending=False)
        pos = 0
        for r in range(len(self.multi_weight)):
            prev_pos = pos
            pos += self.multi_weight[r].shape[0]

            remove_ind = sorted_indices[sorted_indices < pos]
            remove_ind = remove_ind - prev_pos

            indices_to_keep = [i for i in range(self.multi_weight[r].shape[0]) if i not in remove_ind]
            tensors_to_keep = torch.index_select(self.multi_weight[r], dim=0, index=torch.tensor(indices_to_keep))
            
            self.multi_weight[r] = tensors_to_keep
            sorted_indices = sorted_indices[sorted_indices >= pos]

    def cluster_classes(self, drop):
        concatenated = torch.cat([i for i in self.multi_weight], dim=0)
        abs_sum = torch.abs(concatenated).sum(dim=1)
        sorted_indices = torch.argsort(abs_sum)[:drop]
        sorted_indices, _ = torch.sort(sorted_indices, descending=False)
        pos = 0
        for r in range(len(self.multi_weight)):
            prev_pos = pos
            pos += self.multi_weight[r].shape[0]

            remove_ind = sorted_indices[sorted_indices < pos]
            remove_ind = remove_ind - prev_pos

            to_cluster = torch.index_select(self.multi_weight[r], dim=0, index=remove_ind)
            cluster = torchhd.multiset(to_cluster)

            indices_to_keep = [i for i in range(self.multi_weight[r].shape[0]) if i not in remove_ind]
            tensors_to_keep = torch.index_select(self.multi_weight[r], dim=0, index=torch.tensor(indices_to_keep))

            self.multi_weight[r] = tensors_to_keep
            most_similar = torch.argmax(torchhd.cosine_similarity(cluster, self.multi_weight[r]), dim=0)
            self.multi_weight[r] += most_similar
            sorted_indices = sorted_indices[sorted_indices >= pos]


    def get_subclasses(self):
        sub_classes = 0
        for i in self.multi_weight:
            sub_classes += i.shape[0]
        return sub_classes

    def reduce_subclasses(self, train_loader, device, encode, model, accuracy_full, reduce_subclasses = 'drop', threshold = 0.03) -> None:
        for i in range(10):
            accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.in_features).to(device)

            drop_classes = int(self.get_subclasses()*0.1)
            if reduce_subclasses == 'drop':
                self.drop_classes(drop_classes)
            elif reduce_subclasses == 'cluster':
                self.cluster_classes(drop_classes)
            
            with torch.no_grad():
                for samples, labels in tqdm(train_loader, desc="Training"):
                    samples = samples.to(device)
                    labels = labels.to(device)
                    samples_hv = encode(samples)
                    outputs = model.multi_similarity(samples_hv)

                    pred = torch.argmax(outputs, dim=0)
                    row = 0
                    for i in model.multi_weight:
                        if i.shape[0] >= pred:
                            break
                        else:
                            row += 1
                            pred -= i.shape[0]
                    accuracy.update(torch.tensor([row]), labels)
            new_acc = accuracy.compute().item()
            if accuracy_full - new_acc > threshold:
                print(self.get_subclasses(), accuracy_full, new_acc)
                return


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


class PoolCentroid(nn.Module):
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
        pool_size: int,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PoolCentroid, self).__init__()

        self.pool_size = pool_size
        self.in_features = in_features
        self.out_features = out_features

        self.similarity_sum = torch.empty((pool_size), **factory_kwargs)
        self.count = 0
        self.error_similarity_sum = torch.empty((pool_size), **factory_kwargs)
        self.error_count = torch.empty((pool_size), **factory_kwargs)

        weight = torch.empty((pool_size, out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)

        self.correct_predictions = torch.empty(
            (pool_size, out_features), **factory_kwargs
        )
        self.incorrect_predictions = torch.empty(
            (pool_size, out_features), **factory_kwargs
        )
        self.miss_labeled_predictions = torch.empty(
            (pool_size, out_features, out_features), **factory_kwargs
        )
        self.reset_parameters()

    @torch.no_grad()
    def add(self, input: list, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        for i in range(len(input)):
            sample_hv = input[i]
            logit = self(sample_hv, i)
            pred = logit.argmax(1)
            self.weight[i].index_add_(0, target, sample_hv, alpha=lr)
            # print(target, logit.argmax(1))
            if target == pred:
                self.correct_predictions[i][target] += 1
            else:
                self.incorrect_predictions[i][pred] += 1

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        for i in range(self.pool_size):
            norms = self.weight[i].norm(dim=1, keepdim=True)
            norms.clamp_(min=eps)
            self.weight[i].div_(norms)

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        init.zeros_(self.correct_predictions)
        init.zeros_(self.incorrect_predictions)
        init.zeros_(self.miss_labeled_predictions)
        init.zeros_(self.similarity_sum)
        init.zeros_(self.error_similarity_sum)
        init.zeros_(self.error_count)

    def forward(self, input: Tensor, idx, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight[idx])

        return functional.cosine_similarity(input, self.weight[idx])

    @torch.no_grad()
    def add_online(self, input: list, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        for i in range(len(input)):
            sample_hv = input[i]
            logit = self(sample_hv, i)
            pred = logit.argmax(1)
            is_wrong = target != pred

            if is_wrong.sum().item() == 0:
                return

            alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
            self.weight[i].index_add_(0, target, lr * alpha1 * sample_hv)
            alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1
            self.weight[i].index_add_(0, pred, lr * alpha2 * sample_hv)

    @torch.no_grad()
    def add_adapt(self, input: list, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        for i in range(len(input)):
            sample_hv = input[i]
            logit = self(sample_hv, i)
            pred = logit.argmax(1)
            is_wrong = target != pred

            if target == pred:
                self.correct_predictions[i][target] += 1
            else:
                self.incorrect_predictions[i][pred] += 1

                sample_hv = sample_hv[is_wrong]
                self.miss_labeled_predictions[i][pred[0]][target[0]] += 1

                self.weight[i].index_add_(0, target, sample_hv)
                self.weight[i].index_add_(0, pred, -sample_hv)

    @torch.no_grad()
    def add_adapt2(self, input: list, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        for i in range(len(input)):
            sample_hv = input[i]
            logit = self(sample_hv, i)
            pred = logit.argmax(1)
            is_wrong = target != pred

            self.similarity_sum += logit.max(1).values.item()
            if self.error_count[i] == 0:
                val = self.similarity_sum[i] / self.count
            else:
                val = self.error_similarity_sum[i] / self.error_count[i]
            if is_wrong.sum().item() == 0:
                self.correct_predictions[i][target] += 1
                if logit.max(1).values.item() < val:
                    self.weight[i].index_add_(0, target, sample_hv)
            else:
                self.incorrect_predictions[i][pred] += 1
                self.error_count[i] += 1
                self.error_similarity_sum[i] += logit.max(1).values.item()

                sample_hv = sample_hv[is_wrong]
                self.miss_labeled_predictions[i][pred[0]][target[0]] += 1

                self.weight[i].index_add_(0, target, sample_hv)
                self.weight[i].index_add_(0, pred, -sample_hv)

    @torch.no_grad()
    def add_adjust(self, input: list, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.count += 1

        for i in range(len(input)):
            sample_hv = input[i]
            logit = self(sample_hv, i)
            pred = logit.argmax(1)
            is_wrong = target != pred

            self.similarity_sum += logit.max(1).values.item()
            if self.error_count[i] == 0:
                val = self.similarity_sum[i] / self.count
            else:
                val = self.error_similarity_sum[i] / self.error_count[i]
            if is_wrong.sum().item() == 0:
                if logit.max(1).values.item() < val:
                    self.weight[i].index_add_(0, target, sample_hv)
                return

            self.error_count[i] += 1
            self.error_similarity_sum[i] += logit.max(1).values.item()

            alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
            self.weight[i].index_add_(0, target, lr * alpha1 * sample_hv)
            alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1
            self.weight[i].index_add_(0, pred, lr * alpha2 * sample_hv)

    def normalize_miss_labeled(self):
        return torch.nn.functional.normalize(self.miss_labeled_predictions, p=2, dim=2)


class CentroidMiss(nn.Module):
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
    warmup: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        warmup: int,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(CentroidMiss, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.warmup = warmup * 0.2
        print(self.warmup)
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)

        # patterns de miss predictions
        miss_predicted = torch.empty((out_features, in_features), **factory_kwargs)
        self.miss_predicted = Parameter(miss_predicted, requires_grad=requires_grad)

        miss_predicted_large = torch.empty(
            (out_features, out_features, in_features), **factory_kwargs
        )
        self.miss_predicted_large = Parameter(
            miss_predicted_large, requires_grad=requires_grad
        )

        miss_predicted_counter = torch.empty((out_features, out_features, 1), **factory_kwargs)
        self.miss_predicted_counter = Parameter(miss_predicted_counter, requires_grad=requires_grad)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        init.zeros_(self.miss_predicted)
        init.zeros_(self.miss_predicted_large)
        init.zeros_(self.miss_predicted_counter)

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cosine_similarity(input, self.weight)

    def forward_misspredicted(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.miss_predicted)

        return functional.cosine_similarity(input, self.miss_predicted)

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

        norms = self.miss_predicted.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.miss_predicted.div_(norms)

        for i in self.miss_predicted_large:
            norms = i.norm(dim=1, keepdim=True)
            norms.clamp_(min=eps)
            i.div_(norms)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""

        self.count += 1

        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred
        self.weight.index_add_(0, target, input, alpha=lr)

        if is_wrong.sum().item() == 0:
            return

        self.weight.index_add_(0, target, input, alpha=lr * 3)
        self.weight.index_add_(0, pred, -input, alpha=lr)

        # self.miss_predicted.index_add_(0, target, input, alpha=lr)
        # print(pred, target)
        # print(self.miss_predicted_large[target][0][pred])
        # print("add")
        # self.miss_predicted_large[pred].index_add_(0, target, input, alpha=lr)
        # self.miss_predicted_large[pred][0][target][0].index_add_(0, target, input, alpha=lr)
        self.miss_predicted_large[pred, target, :] += input
        self.miss_predicted_counter[pred, target, :] += 1
        # print(self.miss_predicted_large[pred,target,:], input)


    @torch.no_grad()
    def add_adapt(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.miss_predicted.index_add_(0, target, input, alpha=lr)

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
        self.miss_predicted.index_add_(0, target, input, alpha=lr)

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
        self.miss_predicted.index_add_(0, target, input, alpha=lr)

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        # if self.count > self.warmup:
        #    print(torchhd.cosine_similarity(self.weight[target], input))
        if (
            self.count > self.warmup
            and torchhd.cosine_similarity(self.weight[target], input) < val
        ):
            # print(torchhd.cosine_similarity(self.miss_predicted[target], input))
            self.miss_predicted.index_add_(0, target, alpha1 * input, alpha=lr)

        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1
        self.weight.index_add_(0, pred, lr * alpha2 * input)

    @torch.no_grad()
    def add_adjust_2(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
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

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        self.miss_predicted.index_add_(0, target, input, alpha=lr)

        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, pred, -input)

    @torch.no_grad()
    def add_adjust_3(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        self.miss_predicted.index_add_(0, target, input, alpha=lr)

        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, pred, -input)


class CentroidIterative(nn.Module):
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
        super(CentroidIterative, self).__init__()

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

    '''
    @torch.no_grad()
    def add_high_dimensions(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred
        self.count += 1

        r = 1
        if self.correct >= 1:
            r = 1/(self.correct/self.count)
        self.weight.index_add_(0, target, input, alpha=r)

        if is_wrong:
            mask_indices = torch.eq(torch.sign(input), torch.sign(self.weight[pred]))
            mask = torch.zeros_like(input, dtype=torch.bool)
            mask[mask_indices] = True
            alpha = (1-logit[0][target])
            self.weight[pred] -= (alpha*input*r)
        else:
            self.correct += 1
            mask_indices = torch.eq(torch.sign(input), torch.sign(self.weight[target]))
            mask = torch.zeros_like(input, dtype=torch.bool)
            mask[mask_indices] = True
            alpha = (1-logit.max(1).values)
            self.weight[target] += mask * (alpha*input*r)
    '''

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

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input * lr)
        self.weight.index_add_(0, pred, -input * lr)

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

    def reset(self):
        self.similarity_sum = 0
        self.count = 0
        self.error_count = 0
        self.error_similarity_sum = 0

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
    def add_adjust_2(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
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

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, pred, -input)

    @torch.no_grad()
    def add_adjust_3(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, target, input)
        self.weight.index_add_(0, pred, -input)

    @torch.no_grad()
    def add_adjust_9(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""

        self.count += 1

        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        if is_wrong.sum().item() == 0:
            self.weight.index_add_(0, target, input, alpha=lr)
        else:
            self.weight.index_add_(0, target, input, alpha=lr * 4)
            self.weight.index_add_(0, pred, -input, alpha=lr)

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

    def neural_regeneration(self, k=100):
        variance = torch.var(self.weight, dim=0, keepdim=True)
        _, indices = torch.topk(variance, k, largest=False)
        self.weight[:, indices] = torch.zeros(indices.shape)
        return indices
