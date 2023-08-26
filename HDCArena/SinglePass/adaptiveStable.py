import torch
from tqdm import tqdm
import torchhd
import scipy


def train_vanillaHD(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_stable(samples_hv, labels)

    # norms = model.weight_err.norm(dim=1, keepdim=True)
    # norms.clamp_(min=1e-12)
    # model.weight_err.div_(norms)
    # model.normalize()
    for i in range(len(model.weight.data)):
        # print(model.weight.data[i])
        # print(model.weight_err.data[i])
        print(scipy.stats.kstest(model.weight.data[i], model.weight_err.data[i]))
        print(
            torchhd.functional.cosine_similarity(
                model.weight.data[i], model.weight_err.data[i]
            )
        )
        m = torch.nonzero(
            abs(torch.sign(model.weight.data[i]) - torch.sign(model.weight_err.data[i]))
        ).squeeze(1)
        s = 1 - torchhd.functional.cosine_similarity(
            model.weight.data[i], model.weight_err.data[i]
        )
        model.weight.data[i][m] = torch.zeros(len(m))
    # model.weight.data -= model.weight_err.data


def test_vanillaHD(test_loader, device, encode, model, accuracy):
    model.normalize()

    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
