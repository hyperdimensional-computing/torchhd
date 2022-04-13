import math
import torch
import torch.nn as nn

from . import functional
from . import utils


class IdentityEmbedding(nn.Embedding):
    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.identity_hv(
            self.num_embeddings,
            self.embedding_dim,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()


class RandomEmbedding(nn.Embedding):
    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.random_hv(
            self.num_embeddings,
            self.embedding_dim,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()


class LevelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, low=0.0, high=1.0, randomness=0.0, **kwargs):
        self.low_value = low
        self.high_value = high
        self.randomness = randomness

        super(LevelEmbedding, self).__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.level_hv(
            self.num_embeddings,
            self.embedding_dim,
            randomness=self.randomness,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # tranform the floating point input to an index
        # make first variable a copy of the input, then we can reuse the buffer.
        # normalized between 0 and 1
        normalized = (input - self.low_value) / (self.high_value - self.low_value)

        indices = torch.round(normalized * (self.num_embeddings - 1)).long()
        indices.clamp_(0, self.num_embeddings - 1)

        return super(LevelEmbedding, self).forward(indices)


class CircularEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, low=0.0, high=2 * math.pi, randomness=0.0, **kwargs):
        self.low_value = low
        self.high_value = high
        self.randomness = randomness

        super(CircularEmbedding, self).__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        factory_kwargs = {
            "device": self.weight.data.device,
            "dtype": self.weight.data.dtype,
        }
        functional.circular_hv(
            self.num_embeddings,
            self.embedding_dim,
            randomness=self.randomness,
            out=self.weight.data,
            **factory_kwargs
        )

        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # tranform the floating point input to an index
        # make first variable a copy of the input, then we can reuse the buffer.
        # normalized between 0 and 1
        normalized = (input - self.low_value) / (self.high_value - self.low_value)

        indices = torch.round(normalized * self.num_embeddings).long()
        indices.remainder_(self.num_embeddings)

        return super(CircularEmbedding, self).forward(indices)


def query(class_hvs, graph_hv):
    similarities = functional.similarity(graph_hv, class_hvs)
    return similarities, similarities.argmax()


def quantize_count_class_hvs(count_class_hvs, dtype=torch.int8):
    class_hvs = torch.ones(
        count_class_hvs.shape, dtype=dtype, device=count_class_hvs.device
    )

    # majority voting
    class_hvs[count_class_hvs < 0] = -1
    return class_hvs


def train(train_data, encode_fn, num_classes, dim, device=None):
    count_class_hvs = torch.zeros((num_classes, dim), dtype=torch.float, device=device)

    for G in train_data:
        G = G.to(device)
        graph_hv = encode_fn(G)
        count_class_hvs[G.y] += graph_hv.float()

    return count_class_hvs


def train_online(train_data, encode_fn, num_classes, dim, device=None):
    count_class_hvs = torch.zeros((num_classes, dim), dtype=torch.float, device=device)

    for G in train_data:
        G = G.to(device)
        graph_hv = encode_fn(G)
        class_hvs = quantize_count_class_hvs(count_class_hvs[G.y])
        similarity = functional.similarity(graph_hv, class_hvs)
        alpha = 1.0 - similarity
        count_class_hvs[G.y] += graph_hv.float() * alpha

    return count_class_hvs


def retrain(train_data, encode_fn, count_class_hvs, device=None):
    count_class_hvs = count_class_hvs.clone()
    class_hvs = quantize_count_class_hvs(count_class_hvs)

    for G in train_data:
        G = G.to(device)
        graph_hv = encode_fn(G)
        similarities, prediction = query(class_hvs, graph_hv)
        prediction = prediction.to(G.y.device)
        if prediction != G.y:
            graph_hv = graph_hv.float()
            alpha = similarities[prediction] - similarities[G.y]
            count_class_hvs[G.y] += graph_hv * alpha
            count_class_hvs[prediction] -= graph_hv * alpha

            class_hvs = quantize_count_class_hvs(count_class_hvs)

    return count_class_hvs


def test(test_data, encode_fn, count_class_hvs, device=None):
    y_true = torch.zeros(len(test_data), dtype=torch.int)
    y_pred = torch.zeros(len(test_data), dtype=torch.int)
    class_hvs = quantize_count_class_hvs(count_class_hvs)

    for i, G in enumerate(test_data):
        G = G.to(device)
        graph_hv = encode_fn(G)
        similarities, prediction = query(graph_hv, class_hvs)
        y_true[i] = G.y
        y_pred[i] = prediction

    return y_true, y_pred
