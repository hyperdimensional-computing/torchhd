import torch
from tqdm import tqdm
import torchmetrics


def validate(val_loader, encodings, device, model):
    acc = []
    for _i in range(len(encodings)):
        acc.append(
            torchmetrics.Accuracy("multiclass", num_classes=model.out_features).to(
                device
            )
        )

    for samples, labels in tqdm(val_loader, desc="Validation"):
        samples = samples.to(device)
        labels = labels.to(device)

        for index, i in enumerate(encodings):
            outputs = model.forward_index(i(samples), index, device)
            acc[index].update(outputs.to(device), labels.to(device))
    for index, i in enumerate(encodings):
        acc[index] = acc[index].compute().item()

    print(acc)
    conf = model.confidence / model.count_w
    encode_idx = torch.argmax(torch.tensor(acc))
    model.weight.data = model.ww[encode_idx]
    return encode_idx


def train_embeddingHD(
    train_loader, val_loader, test_loader, device, encodings, e, model
):
    warmup = int(len(train_loader) * 0.3)
    with torch.no_grad():
        for idx, (samples, labels) in enumerate(tqdm(train_loader, desc="Training")):
            samples = samples.to(device)
            labels = labels.to(device)

            if idx < warmup:
                for index, i in enumerate(encodings):
                    samples_hv = i(samples)
                    model.add_index(samples_hv, labels, index, device=device)
            else:
                if idx == warmup:
                    encode_idx = validate(test_loader, encodings, device, model)
                    encode_idx = validate(val_loader, encodings, device, model)
                samples_hv = encodings[encode_idx](samples).to(device)
                model.add(samples_hv, labels)

        for samples, labels in tqdm(val_loader, desc="Validation"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encodings[encode_idx](samples).to(device)
            model.add(samples_hv, labels)

        return encodings[encode_idx], encode_idx


def test_embeddingHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
