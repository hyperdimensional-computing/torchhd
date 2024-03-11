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
from typing import Type, Union, Optional, Literal, Callable, Iterable, Tuple
import math
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

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
    "LeHDC",
    "CompHD",
    "SparseHD",
    "QuantHD",
]


class Classifier(nn.Module):
    r"""Base class for all classifiers

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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

    @property
    def device(self) -> torch.device:
        return self.model.weight.device

    def forward(self, samples: Tensor) -> Tensor:
        """Evaluate the logits of the classifier for the given samples.

        Args:
            samples (Tensor): Batch of samples to be classified.

        Returns:
            Tensor: Logits of each samples for each class.

        """
        return self.model(self.encoder(samples))

    def fit(self, data_loader: DataLoader):
        """Fits the classifier to the provided data.

        Args:
            data_loader (DataLoader): Iterable of tuples containing a batch of samples and labels.

        Returns:
            self

        """
        raise NotImplementedError()

    def predict(self, samples: Tensor) -> LongTensor:
        """Predict the class of each given sample.

        Args:
            samples (Tensor): Batch of samples to be classified.

        Returns:
            LongTensor: Index of the predicted class for each sample.

        """
        return torch.argmax(self(samples), dim=-1)

    def accuracy(self, samples: Tensor, labels: LongTensor) -> float:
        """Accuracy in predicting the labels of the samples.

        Args:
            samples (Tensor): Batch of samples to be classified.
            labels (LongTensor): Batch of true labels of the samples.

        Returns:
            float: The accuracy of predicting the true labels.

        """
        predictions = self.predict(samples)
        return torch.mean(predictions == labels, dtype=torch.float).item()


class Vanilla(Classifier):
    r"""Baseline centroid classifier.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

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

    def fit(self, data_loader: DataLoader):
        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            encoded = self.encoder(samples)
            self.model.add(encoded, labels)

        return self


class AdaptHD(Classifier):
    r"""Implements `AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8918974>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr

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

    def fit(self, data_loader: DataLoader):

        for _ in range(self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

        return self


# Adapted from: https://gitlab.com/biaslab/onlinehd/
class OnlineHD(Classifier):
    r"""Implements `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader):

        for _ in range(self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

        return self


class RefineHD(Classifier):
    r"""Implements `RefineHD: : Accurate and Efficient Single-Pass Adaptive Learning Using Hyperdimensional Computing <https://ieeexplore.ieee.org/abstract/document/10386671>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr

        self.adjust_reset()

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def adjust_reset(self):
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

    def fit(self, data_loader: DataLoader):
        for _ in range(self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                logits = self.model(encoded)

                top2_pred = torch.topk(logits, 2)
                pred = top2_pred.indices[:, 0]
                is_wrong = labels != pred

                w = 1 - top2_pred.values[:, 0] - top2_pred.values[:, 1]

                self.similarity_sum += logits.max(1).values.item()
                self.count += 1
                if self.error_count == 0:
                    val = self.similarity_sum / self.count
                else:
                    val = self.error_similarity_sum / self.error_count
                if is_wrong.sum().item() == 0:
                    if logits.max(1).values.item() < val:
                        self.model.weight.index_add_(0, labels, self.lr * w * encoded)
                    return

                self.error_count += 1
                self.error_similarity_sum += logits.max(1).values.item()

                logits = logits[is_wrong]
                encoded = encoded[is_wrong]
                labels = labels[is_wrong]
                pred = pred[is_wrong]

                alpha1 = 1.0 - logits.gather(1, labels.unsqueeze(1))
                alpha2 = logits.gather(1, pred.unsqueeze(1)) - 1

                self.model.weight.index_add_(
                    0, labels, alpha1 * w * encoded, alpha=self.lr
                )
                self.model.weight.index_add_(
                    0, pred, alpha2 * w * encoded, alpha=self.lr
                )

        return self


# Adapted from: https://gitlab.com/biaslab/neuralhd
class NeuralHD(Classifier):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int): The fraction of hidden dimensions to regenerate.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader):

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            encoded = self.encoder(samples)
            self.model.add(encoded, labels)

        for epoch_idx in range(1, self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

            # Regenerate feature dimensions
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
    r"""Implements `DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification <https://ieeexplore.ieee.org/document/10247876>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int): The fraction of hidden dimensions to regenerate.
        alpha (float): Parameter effecting the dimensions to regenerate, see paper for details.
        beta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        theta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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

        self.encoder = Projection(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, data_loader: DataLoader):

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for epoch_idx in range(self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                scores = 0
                for samples, labels in data_loader:
                    samples = samples.to(self.device)
                    labels = labels.to(self.device)

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

        # Partial correct
        partial = pred2 == labels
        dist2corr = torch.abs(weight[labels[partial]] - samples[partial])
        dist2incorr = torch.abs(weight[pred1[partial]] - samples[partial])
        partial_dist = torch.sum(
            (self.beta * dist2incorr - self.alpha * dist2corr), dim=0
        )

        # Completely incorrect
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


class LeHDC(Classifier):
    r"""Implements `DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification <https://ieeexplore.ieee.org/document/10247876>`_.


    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        weight_decay (float): The rate at which the weights of the model are decayed during training.
        dropout_rate (float): The fraction of hidden dimensions to randomly zero-out.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Projection
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
        lr: float = 0.01,
        weight_decay: float = 0.03,
        dropout_rate: float = 0.3,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

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
        self.dropout = torch.nn.Dropout(dropout_rate)
        # Gradient model accumulates gradients
        self.grad_model = Centroid(
            n_dimensions, n_classes, device=device, dtype=dtype, requires_grad=True
        )
        # Regular model is a binarized version of the gradient model
        self.model = Centroid(
            n_dimensions, n_classes, device=device, dtype=dtype, requires_grad=True
        )

    @property
    def device(self) -> torch.device:
        return self.model.weight.device

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.dropout(self.encoder(samples)), dot=True)

    def fit(self, data_loader: DataLoader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.grad_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        for _ in range(self.epochs):
            accumulated_loss = 0

            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                logits = self(samples)
                loss = criterion(logits, labels)
                accumulated_loss += loss.detach().item()

                # Zero out all the gradients
                self.grad_model.zero_grad()
                self.model.zero_grad()

                loss.backward()

                # The gradient model is updated using the gradients from the binarized model
                self.grad_model.weight.grad = self.model.weight.grad
                optimizer.step()

                # Quantize the weights
                with torch.no_grad():
                    self.model.weight.data = self.grad_model.weight.sign()

            scheduler.step(accumulated_loss)

        return self


class CompHD(Classifier):
    r"""Implements `CompHD: Efficient Hyperdimensional Computing Using Model Compression <https://ieeexplore.ieee.org/document/8824908>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        chunks (int): The number of times the model is reduced in size.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        chunks: int = 10,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        if n_dimensions % chunks != 0:
            raise ValueError("n_dimensions must be divisible by chunks.")

        self.chunks = chunks

        self.feat_keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )

        self.model_count = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self.model = Centroid(
            n_dimensions // chunks, n_classes, device=device, dtype=dtype
        )

        n_chunk_keys = max(self.n_dimensions // self.chunks, self.chunks)
        chunk_keys = torch.from_numpy(scipy.linalg.hadamard(n_chunk_keys))
        chunk_keys = chunk_keys.to(self.device)
        self.chunk_keys = chunk_keys[: self.chunks, : self.n_dimensions // self.chunks]

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.feat_keys.weight, self.levels(samples)).sign()

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.compress(self.encoder(samples)))

    def compress(self, input):
        shape = (self.chunks, self.n_dimensions // self.chunks)
        return functional.hash_table(self.chunk_keys, torch.reshape(input, shape))

    def fit(self, data_loader: DataLoader):
        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            encoded = self.encoder(samples)
            self.model_count.add(encoded, labels)

        with torch.no_grad():
            shape = (self.n_classes, self.chunks, self.n_dimensions // self.chunks)
            weight_chunks = torch.reshape(self.model_count.weight, shape)

            keys = self.chunk_keys[None, ...].expand(self.n_classes, -1, -1)
            comp_weights = functional.hash_table(keys, weight_chunks)
            self.model.weight.data = comp_weights

        return self


class SparseHD(Classifier):
    r"""Implements `SparseHD: Algorithm-Hardware Co-optimization for Efficient High-Dimensional Computing <https://ieeexplore.ieee.org/document/8735551>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        sparsity (float): The fraction of weights to be zero.
        sparsity_type (str): The way in which to apply the sparsity, per hidden dimension, or per class.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        sparsity: float = 0.1,
        sparsity_type: Literal["dimension", "class"] = "dimension",
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr
        self.sparsity = sparsity
        self.sparsity_type = sparsity_type

        self.feat_keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
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
        return functional.hash_table(self.feat_keys.weight, self.levels(samples)).sign()

    def fit(self, data_loader: DataLoader):
        for _ in range(self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

            self.sparsify()

        return self

    def sparsify(self) -> None:
        s = round((1 - self.sparsity) * self.n_dimensions)

        if self.sparsity_type == "dimension":
            max_vals, _ = torch.max(self.model.weight.data, dim=0)
            min_vals, _ = torch.min(self.model.weight.data, dim=0)
            variation = max_vals - min_vals

            _, mask = torch.topk(variation, k=s, largest=False, sorted=False)
            self.model.weight.data[:, mask] = 0

        if self.sparsity_type == "class":
            _, mask = torch.topk(
                self.model.weight.abs(), k=s, dim=1, largest=False, sorted=False
            )
            self.model.weight.data.scatter(
                1, mask, torch.zeros_like(self.model.weight.data)
            )


class QuantHD(Classifier):
    r"""Implements `QuantHD: A Quantization Framework for Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8906150>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int): The number of discretized levels for the level-hypervectors.
        min_level (int): The lower-bound of the range represented by the level-hypervectors.
        max_level (int): The upper-bound of the range represented by the level-hypervectors.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

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
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr

        self.feat_keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )

        self.model_count = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.feat_keys.weight, self.levels(samples)).sign()

    def binarize(self):
        self.model.weight.data = torch.sign(self.model_count.weight.data)

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples), dot=True)

    def add_quantize(self, input: Tensor, target: Tensor) -> None:
        logit = self.model(input, dot=True)
        pred = logit.argmax(1)
        is_wrong = target != pred

        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.model_count.weight.index_add_(0, target, input, alpha=self.lr)
        self.model_count.weight.index_add_(0, pred, input, alpha=-self.lr)

    def fit(self, data_loader: DataLoader):

        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            samples_hv = self.encoder(samples)
            self.model_count.add(samples_hv, labels)

        self.binarize()

        for _ in range(1, self.epochs):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                samples_hv = self.encoder(samples)
                self.add_quantize(samples_hv, labels)

            self.binarize()

        return self
