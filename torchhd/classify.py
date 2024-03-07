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
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor, LongTensor
from torch.nn.parameter import Parameter

import torchhd.functional as functional
from torchhd.embeddings import Random, Level, Projection, Sinusoid
from torchhd.models import Centroid


__all__ = [
    "Classifier",
    "Vanilla",
    "AdaptHD",
    "OnlineHD",
    "DistHD",
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

    def fit(self, samples: Tensor, labels: LongTensor) -> Self:
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

    def fit(self, samples: Tensor, labels: LongTensor) -> Self:

        loader = DataLoader(
            TensorDataset(samples, labels), self.batch_size, shuffle=False
        )

        for samples, labels in loader:
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

    def fit(self, samples: Tensor, labels: LongTensor) -> Self:

        loader = DataLoader(
            TensorDataset(samples, labels), self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for samples, labels in loader:
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

    def fit(self, samples: Tensor, labels: LongTensor) -> Self:

        loader = DataLoader(
            TensorDataset(samples, labels), self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for samples, labels in loader:
                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

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
        n_regen: int = 20,
        regen_rate: float = 0.04,
        alpha: float = 0.5,
        beta: float = 1,
        theta: float = 0.25,
        epochs: int = 20,
        lr: float = 0.05,
        batch_size: Union[int, None] = 1024,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.n_regen = n_regen
        self.regen_rate = regen_rate
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Projection(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, samples: Tensor, labels: LongTensor) -> Self:

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        loader = DataLoader(
            TensorDataset(samples, labels), self.batch_size, shuffle=True
        )

        for _ in range(self.n_regen):
            for _ in range(self.epochs):
                for samples, labels in loader:
                    encoded = self.encoder(samples)
                    self.model.add_online(encoded, labels, lr=self.lr)

            scores = 0
            for samples, labels in loader:
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
