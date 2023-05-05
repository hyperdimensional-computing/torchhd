import torch
from tqdm import tqdm
import torchmetrics
from collections import deque
import math


def train_sparseHD(
    train_loader,
    device,
    encode,
    model,
    model_sparse,
    iterations,
    num_classes,
    lr=0.05,
    epsilon=0.05,
    s=0.05,
    dimensions=10000,
):
    train_len = len(train_loader)
    validation_set = train_len - math.ceil(len(train_loader) * 0.05)

    with torch.no_grad():
        accuracy_validation = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        ).to(device)
        for iter in range(iterations):
            accuracy_validation_sparse = torchmetrics.Accuracy(
                "multiclass", num_classes=num_classes
            ).to(device)
            for idx, (samples, labels) in enumerate(
                tqdm(train_loader, desc="Training")
            ):
                samples = samples.to(device)
                labels = labels.to(device)
                samples_hv = encode(samples)

                if idx < validation_set:
                    model.add_sparse(samples_hv, labels, lr=lr, iter=iter)
                else:
                    if idx == validation_set:
                        model.sparsify_model(model_sparse, int(s * dimensions), iter)

                    outputs = model.sparse_similarity(samples_hv)
                    if iter == 0:
                        outputs_full = model.forward(samples_hv)
                        accuracy_validation.update(outputs_full.to(device), labels)
                    accuracy_validation_sparse.update(outputs.to(device), labels)
            if iter == 0:
                accuracy = accuracy_validation.compute().item()

            if abs(accuracy_validation_sparse.compute().item() - accuracy) < epsilon:
                return iter
    return iterations


def test_sparseHD(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model.sparse_similarity(samples_hv)
            accuracy.update(outputs.to(device), labels)
