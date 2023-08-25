import torch
from tqdm import tqdm


def train_rvfl(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adapt(samples_hv, labels)


def test_rvfl(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=False)
            accuracy.update(outputs.cpu(), labels)