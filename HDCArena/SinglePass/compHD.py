import torch
from tqdm import tqdm


def train_compHD(train_loader, device, encode, model, chunks):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)
    model.comp_compress(chunks, device)


def test_compHD(test_loader, device, encode, model, accuracy, chunks):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            samples_hv = model.compress_hv(samples_hv, chunks, device)
            outputs = model.forward_comp(samples_hv).unsqueeze(0).to(device)
            accuracy.update(outputs.to(device), labels.to(device))
