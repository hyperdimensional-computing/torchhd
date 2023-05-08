import torch
from tqdm import tqdm
import torchmetrics
from collections import deque
import math


def train_neuralHD(
    train_loader,
    device,
    encode,
    model,
    iterations,
    model_neural,
    lazy_regeneration=5,
    lr=1,
    r=0.05,
    dimensions=10000,
):
    with torch.no_grad():
        for iter in range(iterations):
            for idx, (samples, labels) in enumerate(
                tqdm(train_loader, desc="Training")
            ):
                samples = samples.to(device)
                labels = labels.to(device)
                samples_hv = encode(samples)
                model.add_neural(samples_hv, labels, lr=lr)
            if iter % lazy_regeneration == 0:
                model.neural_regenerate(int(r * dimensions), encode, device)
                model.normalize()
                if model_neural == "reset" and iter != iterations - 1:
                    model.reset_parameters()
    return iterations


def test_neuralHD(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), labels)
