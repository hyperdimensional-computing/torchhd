import torch

from . import functional
from . import utils


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
