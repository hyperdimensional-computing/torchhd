import torch
from tqdm import tqdm

torch.set_printoptions(threshold=torch.inf)


def train_noiseHD(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adjust(samples_hv, labels)

    non_significant = torch.where(model.noise < torch.mean(model.noise))[1].to(device)
    model.weight.data[:, non_significant] = torch.zeros(
        (model.weight.shape[0], len(non_significant))
    ).to(device)
    # l = 1

    # for i in range(model.weight.size(1)):
    #    _, topk_indices = torch.topk(torch.abs(model.weight[:,i]), k=l, dim=0, largest=False)
    #    model.weight[:,i][topk_indices] = 0


def test_noiseHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
