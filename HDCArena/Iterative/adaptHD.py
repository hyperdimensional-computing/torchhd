import torch
from tqdm import tqdm
import torchmetrics
from collections import deque


def train_adaptHD(train_loader, device, encode, model, iterations, num_classes, lr):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    with torch.no_grad():
        q = deque(maxlen=3)
        for iter in range(iterations):
            accuracy_train = torchmetrics.Accuracy(
                "multiclass", num_classes=num_classes
            ).to(device)

            for samples, labels in tqdm(train_loader, desc="Training"):
                samples = samples.to(device)
                labels = labels.to(device)

                samples_hv = encode(samples)
                model.add_adapt(samples_hv, labels, lr=lr)
                outputs = model.forward(samples_hv, dot=False)
                accuracy_train.update(outputs.to(device), labels.to(device))
            lr = (1 - accuracy_train.compute().item()) * 10

            if len(q) == 3:
                if all(abs(q[i] - q[i - 1]) < 0.001 for i in range(1, len(q))):
                    return iter
                q.append(accuracy_train.compute().item())
            else:
                q.append(accuracy_train.compute().item())
    return iterations


def test_adaptHD(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
