import torch
from tqdm import tqdm
import torchmetrics
import csv

def train_adaptHD(train_loader, device, encode, model, test_loader, name, results_file, method, encoding):
    with torch.no_grad():
        l = int(len(train_loader) / 10)
        c = 0
        per = 0
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=model.out_features).to(
            device
        )
        for iter, (samples, labels) in enumerate(tqdm(train_loader, desc="Training")):
            c += 1

            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adapt(samples_hv, labels)
            if c % l == 0:
                test_adaptHD(test_loader, device, encode, model, accuracy)
                per += 1
                if per != 10:
                    with open(results_file, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                name,
                                accuracy.compute().item(),
                                0,
                                0,
                                model.in_features,
                                method,
                                encoding,
                                1,
                                per/10
                            ]
                        )
                accuracy.reset()
    test_adaptHD(test_loader, device, encode, model, accuracy)
    with open(results_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                name,
                accuracy.compute().item(),
                0,
                0,
                model.in_features,
                method,
                encoding,
                1,
                per / 10
            ]
        )

def test_adaptHD(test_loader, device, encode, model, accuracy):
    #model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=False)
            accuracy.update(outputs.to(device), labels.to(device))
