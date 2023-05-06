import torch
from tqdm import tqdm
from torchhd import functional

def train_rvfl(train_ds, encode, model, device, classes, alpha):
    samples = train_ds.data
    labels = train_ds.targets
    n = len(labels)

    encodings = encode(samples.to(device)).to(device)

    one_hot_labels = torch.zeros(n, classes)
    one_hot_labels[torch.arange(n), labels] = 1

    weights = functional.ridge_regression(encodings, one_hot_labels.to(device), alpha=alpha).to(device)
    model.weight.copy_(weights)


def test_rvfl(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, targets in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            encoded = encode(samples)
            logits = model.dot_similarity(encoded)
            accuracy.update(logits.to(device), targets.to(device))
