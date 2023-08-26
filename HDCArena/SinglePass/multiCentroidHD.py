import torch
from tqdm import tqdm
import torchmetrics


def train_multicentroidHD(
    train_loader,
    device,
    encode,
    model,
    num_classes,
    reduce_subclasses="drop",
    threshold=0.03,
):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            pred = model.add_multi(samples_hv, labels, device)
            accuracy.update(pred.to(device), labels)
        model.reduce_subclasses(
            train_loader,
            device,
            encode,
            model,
            num_classes,
            accuracy.compute().item(),
            reduce_subclasses=reduce_subclasses,
            threshold=threshold,
        )


def test_multicentroidHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model.multi_similarity(samples_hv, device)

            pred = torch.argmax(outputs, dim=0)
            row = 0
            for i in model.multi_weight:
                if i.shape[0] >= pred:
                    break
                else:
                    row += 1
                    pred -= i.shape[0]
            accuracy.update(torch.tensor([row]).to(device), labels.to(device))
