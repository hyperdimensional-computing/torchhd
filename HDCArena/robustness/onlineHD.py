import torch
from tqdm import tqdm
import torchmetrics
import csv


def train_onlineHD(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_online(samples_hv, labels)


def test_onlineHD(test_loader, device, encode, model, accuracy, name, results_file, dimensions, method, encoding, failure):
    model.normalize()
    acc = torchmetrics.Accuracy("multiclass", num_classes=model.out_features).to(
                device
            )
    accuracies = [torchmetrics.Accuracy("multiclass", num_classes=model.out_features).to(
                device
            ) for i in range(len(failure))]
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            acc.update(outputs.to(device), labels.to(device))

            for i, f in enumerate(failure):
                samples_hv = encode(samples)

                num_dim = int((f/100)*dimensions)
                f_mask = torch.randperm(dimensions - 0)[:num_dim]
                samples_hv[0][f_mask] = samples_hv[0][f_mask] * -torch.ones(num_dim).to(device)
                #print(samples_hv, num_dim)
                outputs = model(samples_hv, dot=True)

                accuracies[i].update(outputs.to(device), labels.to(device))

    for i, f in enumerate(failure):
        v = acc.compute().item()-accuracies[i].compute().item()

        with open(results_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    name,
                    v,
                    0,
                    0,
                    dimensions,
                    method,
                    encoding,
                    f,
                ]
            )