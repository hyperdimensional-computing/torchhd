import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils
import torchmetrics
from collections import deque
import math


def train_quantHD(
    train_loader,
    test_loader,
    num_classes,
    encode,
    model,
    device,
    name,
    method,
    encoding,
    iterations,
    dimensions,
    lr,
    chunks,
    threshold,
    reduce_subclasses,
    model_quantize,
    epsilon,
    model_sparse,
    s,
    alpha,
    beta,
    theta,
    r,
    partial_data,
    robustness,
    lazy_regeneration,
    model_neural,
    results_file,
):
    train_len = len(train_loader)
    validation_set = train_len - math.ceil(len(train_loader) * 0.05)

    train_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    with torch.no_grad():
        q = deque(maxlen=2)
        for iter in range(iterations):
            accuracy_validation = torchmetrics.Accuracy(
                "multiclass", num_classes=num_classes
            ).to(device)
            for idx, (samples, labels) in enumerate(
                tqdm(train_loader, desc="Training")
            ):
                samples = samples.to(device)
                labels = labels.to(device)
                samples_hv = encode(samples)
                if idx < validation_set:
                    model.add_quantize(samples_hv, labels, lr=lr, model=model_quantize)
                else:
                    if idx == validation_set:
                        model.binarize_model(model_quantize, device)
                    outputs = model.quantized_similarity(
                        samples_hv, model_quantize
                    ).float()
                    accuracy_validation.update(outputs.to(device), labels.to(device))

            if len(q) == 2:
                if all(abs(q[i] - q[i - 1]) < epsilon for i in range(1, len(q))):
                    iterations = iter
                    break
                q.append(accuracy_validation.compute().item())
            else:
                q.append(accuracy_validation.compute().item())

    train_time = time.time() - train_time

    model.normalize()

    utils.test_eval(
        test_loader,
        num_classes,
        encode,
        model,
        device,
        name,
        method,
        encoding,
        iterations,
        dimensions,
        lr,
        chunks,
        threshold,
        reduce_subclasses,
        model_quantize,
        epsilon,
        model_sparse,
        s,
        alpha,
        beta,
        theta,
        r,
        partial_data,
        robustness,
        results_file,
        train_time,
        lazy_regeneration,
        model_neural,
    )
