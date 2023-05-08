import torch
from tqdm import tqdm

torch.set_printoptions(threshold=torch.inf)


def train_highHD(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    unique_counts = torch.tensor(
        [len(torch.unique(model.weight[:, i])) for i in range(model.weight.shape[1])]
    )
    repeated_dim = torch.nonzero(unique_counts <= 4).T.long()
    model.weight[:, repeated_dim] = torch.zeros(
        (model.weight.shape[0], 1, len(repeated_dim))
    )
    print(repeated_dim.shape)
    print(model.weight.shape)
    # l = 1

    # for i in range(model.weight.size(1)):
    #    _, topk_indices = torch.topk(torch.abs(model.weight[:,i]), k=l, dim=0, largest=False)
    #    model.weight[:,i][topk_indices] = 0


def test_highHD(test_loader, device, encode, model, accuracy):
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=False)
            accuracy.update(outputs.to(device), labels.to(device))
