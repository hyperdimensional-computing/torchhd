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
from typing import Type, Self, Union, Optional, Literal, Callable, Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch.nn.parameter import Parameter
from collections import deque
import torchmetrics
import torchhd.functional as functional
from torchhd.embeddings import Random, Level, Projection, Sinusoid
from torchhd.models import Centroid

DataLoader = Iterable[Tuple[Tensor, LongTensor]]

__all__ = [
    "Classifier",
    "Vanilla",
    "AdaptHD",
    "OnlineHD",
    "RefineHD",
    "NeuralHD",
    "DistHD",
    "CompHD",
    "SparseHD",
    "QuantHD"
]


class Classifier(nn.Module):
    encoder: Callable[[Tensor], Tensor]
    model: Callable[[Tensor], Tensor]

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def fit(self, data_loader: DataLoader) -> Self:
        raise NotImplementedError()

    def predict(self, samples: Tensor) -> LongTensor:
        return torch.argmax(self(samples), dim=-1)

    def score(self, samples: Tensor, labels: LongTensor) -> float:
        predictions = self.predict(samples)
        return torch.mean(predictions == labels, dtype=torch.float).item()


class Vanilla(Classifier):
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            n_levels: int = 100,
            min_level: int = -1,
            max_level: int = 1,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.batch_size = batch_size

        self.keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, data_loader: DataLoader) -> Self:
        for samples, labels in data_loader:
            encoded = self.encoder(samples)
            self.model.add(encoded, labels)

        return self


class AdaptHD(Classifier):
    r"""Implements `AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8918974>`_."""

    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            n_levels: int = 100,
            min_level: int = -1,
            max_level: int = 1,
            epochs: int = 120,
            lr: float = 0.035,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, data_loader: DataLoader) -> Self:

        for _ in range(self.epochs):
            for samples, labels in data_loader:
                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

        return self


# Adapted from: https://gitlab.com/biaslab/onlinehd/
class OnlineHD(Classifier):
    r"""Implements `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            epochs: int = 120,
            lr: float = 0.035,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader) -> Self:

        for _ in range(self.epochs):
            for samples, labels in data_loader:
                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

        return self


class RefineHD(Classifier):
    r"""Implements `RefineHD: : Accurate and Efficient Single-Pass Adaptive Learning Using Hyperdimensional Computing <https://www.computer.org/csdl/proceedings-article/icrc/2023/10386671/1TJmff9VDYQ>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            epochs: int = 120,
            lr: float = 0.035,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

    def adjust_reset(self):
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

    def fit(self, data_loader: DataLoader) -> Self:
        for _ in range(self.epochs):
            for samples, labels in data_loader:
                encoded = self.encoder(samples)
                self.model.add_adjust(encoded, labels, self.count, self.similarity_sum, self.error_count, self.error_similarity_sum, lr=self.lr)
        return self


# Adapted from: https://gitlab.com/biaslab/neuralhd
class NeuralHD(Classifier):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            regen_freq: int = 20,
            regen_rate: float = 0.04,
            epochs: int = 120,
            lr: float = 0.37,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader) -> Self:

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for samples, labels in data_loader:
            encoded = self.encoder(samples)
            self.model.add(encoded, labels)

        for epoch_idx in range(1, self.epochs):
            for samples, labels in data_loader:
                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[regen_dims].uniform_(0, 2 * math.pi)

        return self


# Adapted from: https://github.com/jwang235/DistHD/
class DistHD(Classifier):
    r"""Implements `DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification <https://ieeexplore.ieee.org/document/10247876>`_."""

    encoder: Projection
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            regen_freq: int = 20,
            regen_rate: float = 0.04,
            alpha: float = 0.5,
            beta: float = 1,
            theta: float = 0.25,
            epochs: int = 120,
            lr: float = 0.05,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Projection(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader) -> Self:

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for epoch_idx in range(self.epochs):
            for samples, labels in data_loader:
                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                scores = 0
                for samples, labels in data_loader:
                    scores += self.regen_score(samples, labels)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()

        return self

    def regen_score(self, samples, labels):
        scores = self(samples)
        top2_preds = torch.topk(scores, k=2).indices
        pred1, pred2 = torch.unbind(top2_preds, dim=-1)
        wrong = pred1 != labels

        samples = samples[wrong]
        pred2 = pred2[wrong]
        labels = labels[wrong]
        pred1 = pred1[wrong]

        weight = F.normalize(self.model.weight, dim=1)

        # partial correct
        partial = pred2 == labels
        dist2corr = torch.abs(weight[labels[partial]] - samples[partial])
        dist2incorr = torch.abs(weight[pred1[partial]] - samples[partial])
        partial_dist = torch.sum(
            (self.beta * dist2incorr - self.alpha * dist2corr), dim=0
        )

        # completely incorrect
        complete = pred2 != labels
        dist2corr = torch.abs(weight[labels[complete]] - samples[complete])
        dist2incorr1 = torch.abs(weight[pred1[complete]] - samples[complete])
        dist2incorr2 = torch.abs(weight[pred2[complete]] - samples[complete])
        complete_dist = torch.sum(
            (
                    self.beta * dist2incorr1
                    + self.theta * dist2incorr2
                    - self.alpha * dist2corr
            ),
            dim=0,
        )

        return 0.5 * partial_dist + complete_dist


class CompHD(Classifier):
    r"""Implements `CompHD: Efficient Hyperdimensional Computing Using Model Compression <https://ieeexplore.ieee.org/document/8824908>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            *,
            chunks=10,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )
        self.chunks = chunks
        self.batch_size = batch_size

        self.position_vectors = None

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        comp_weight = torch.empty((self.model.out_features, int(self.model.in_features / self.chunks)))
        self.comp_weight = Parameter(comp_weight)

    def fit(self, data_loader: DataLoader) -> Self:
        for samples, labels in data_loader:
            encoded = self.encoder(samples)
            self.model.add(encoded, labels)
        self.comp_compress()
        return self

    def comp_compress(self):
        w_re = torch.reshape(
            self.model.weight, (self.model.out_features, self.chunks, int(self.model.in_features / self.chunks))
        )
        self.position_vectors = Random(
            self.chunks, int(self.model.in_features / self.chunks)
        )

        for i in range(self.model.out_features):
            self.comp_weight.data[i] = torch.sum(
                w_re[i] * self.position_vectors.weight, dim=0
            )

    def compress_hv(self, enc):
        return torch.sum(
            torch.reshape(enc, (self.chunks, int(self.n_dimensions / self.chunks)))
            * self.position_vectors.weight,
            dim=0,
        )

    def forward_comp(self, enc):
        return functional.dot_similarity(
            enc, self.comp_weight
        )

    def predict(self, samples: Tensor) -> LongTensor:
        return torch.tensor([torch.argmax(self.forward_comp(self.compress_hv(self.encoder(samples))), dim=-1)])


class SparseHD(Classifier):
    r"""Implements `SparseHD: Algorithm-Hardware Co-optimization for Efficient High-Dimensional Computing <https://ieeexplore.ieee.org/document/8735551>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            epochs: int,
            lr: float,
            sparsity: float = 0.1,
            sparse_type: str = 'dimension',
            *,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.sparsity = sparsity
        self.batch_size = batch_size
        self.sparse_type = sparse_type
        weight_sparse = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight_sparse = Parameter(weight_sparse, requires_grad=requires_grad)

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader) -> Self:
        for epoch_idx in range(self.epochs):
            for samples, labels in data_loader:
                samples_hv = encode(samples)
                model.add_sparse(samples_hv, labels, self.weight_sparse, lr=lr, iter=iter)
            self.sparsify_model(epoch_idx)
        return self

    def sparsify_model(self, epoch_index):
        if self.sparse_type == "dimension":
            if epoch_index == 0:
                max_vals, _ = torch.max(self.model.weight.data, dim=0)
                min_vals, _ = torch.min(self.model.weight.data, dim=0)
            else:
                max_vals, _ = torch.max(self.model.weight_sparse.data, dim=0)
                min_vals, _ = torch.min(self.model.weight_sparse.data, dim=0)
            variation = max_vals - min_vals
            _, dropped_indices = variation.topk(s, largest=False)

            if epoch_index == 0:
                self.weight_sparse.data = self.model.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0
        if self.sparse_type == "class":
            if epoch_index == 0:
                _, dropped_indices = torch.topk(
                    self.model.weight.abs(), k=s, dim=1, largest=False, sorted=True
                )
            else:
                _, dropped_indices = torch.topk(
                    self.weight_sparse.abs(), k=s, dim=1, largest=False, sorted=True
                )
            if iter == 0:
                self.weight_sparse.data = self.model.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0

    def predict(self, samples: Tensor) -> LongTensor:
        return torch.argmax(functional.dot_similarity(samples, self.weight_sparse), dim=-1)


class QuantHD(Classifier):
    r"""Implements `SparseHD: Algorithm-Hardware Co-optimization for Efficient High-Dimensional Computing <https://ieeexplore.ieee.org/document/8735551>`_."""

    encoder: Sinusoid
    model: Centroid

    def __init__(
            self,
            n_features: int,
            n_dimensions: int,
            n_classes: int,
            epochs: int = 3,
            epsilon: float = 0.01,
            lr: float = 1,
            quant_type: str = 'ternary',
            *,
            batch_size: Union[int, None] = 1024,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.quant_type = quant_type
        weight_quant = torch.empty((n_features, n_dimensions))
        self.weight_quant = Parameter(weight_quant)
        self.n_classes = n_classes
        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def binarize_model(self):
        if self.quant_type == "binary":
            self.weight_quant.data = torch.sgn(self.model.weight.data)
        elif self.quant_type == "ternary":
            self.weight_quant.data = torch.where(
                self.model.weight.data > 0,
                torch.tensor(1.0),
                torch.where(
                    self.model.weight.data < 0, torch.tensor(-1.0), torch.tensor(0.0)
                ),
            )

    def fit(self, data_loader: DataLoader) -> Self:
        train_len = len(data_loader)
        validation_set = train_len - math.ceil(train_len * 0.05)

        for samples, labels in data_loader:
            samples_hv = self.encoder(samples)
            self.model.add(samples_hv, labels)
        self.binarize_model()

        q = deque(maxlen=2)
        for epoch_idx in range(self.epochs):
            accuracy_validation = torchmetrics.Accuracy(
                "multiclass", num_classes=self.n_classes
            )
            for samples, labels in data_loader:
                samples_hv = self.encoder(samples)
                if epoch_idx < validation_set:
                    self.model.add_quantize(samples_hv, labels, self.weight_quant, lr=self.lr, model=self.quant_type)
                else:
                    if epoch_idx == validation_set:
                        self.binarize_model(self.quant_type)
                    outputs = self.model.quantized_similarity(
                        samples_hv, self.quant_type
                    ).float()
                    accuracy_validation.update(outputs, labels)
            if len(q) == 2:
                if all(abs(q[i] - q[i - 1]) < self.epsilon for i in range(1, len(q))):
                    return self
                q.append(accuracy_validation.compute().item())
            else:
                q.append(accuracy_validation.compute().item())

        return self

    def predict(self, samples: Tensor) -> LongTensor:
        return torch.argmax(self.model.quantized_similarity(self.encoder(samples), self.quant_type, self.weight_quant), dim=-1)
