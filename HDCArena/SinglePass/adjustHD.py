import torch
from tqdm import tqdm


def train_adjustHD(train_loader, device, encode, model):
    with torch.no_grad():
        tr = int(len(train_loader)/20)
        t2 = int(len(train_loader)/10)
        for iter, (samples, labels) in enumerate(tqdm(train_loader, desc="Training")):
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            if iter < tr:
                model.add_adjust(samples_hv, labels)
            elif iter < t2:
                model.add_adjust_ad(samples_hv, labels)
                model.add_adjust(samples_hv, labels)
            else:
                model.add_adjust_ad2(samples_hv, labels)
                model.add_adjust_ad(samples_hv, labels)
                model.add_adjust(samples_hv, labels)
        model.merge_adjust()

def test_adjustHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
