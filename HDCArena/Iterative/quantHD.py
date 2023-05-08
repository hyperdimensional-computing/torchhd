import torch
from tqdm import tqdm
import torchmetrics
from collections import deque
import math


def train_quantHD(
    train_loader,
    device,
    encode,
    model,
    model_quantize,
    iterations,
    num_classes,
    lr=1,
    epsilon=0.01,
):
    train_len = len(train_loader)
    validation_set = train_len - math.ceil(len(train_loader) * 0.05)

    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)
    model.binarize_model(model_quantize, device)

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
                    return iter
                q.append(accuracy_validation.compute().item())
            else:
                q.append(accuracy_validation.compute().item())
    return iterations


def test_quantHD(test_loader, device, encode, model, accuracy, model_quantize):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model.quantized_similarity(samples_hv, model_quantize)
            accuracy.update(outputs.to(device), labels)
