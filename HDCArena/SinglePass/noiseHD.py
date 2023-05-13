import torch
from tqdm import tqdm
from collections import Counter

torch.set_printoptions(threshold=torch.inf)


def train_noiseHD(train_loader, device, encode, model):
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_noise(samples_hv, labels, device=device)


    counts = torch.bincount(torch.tensor(model.noise.squeeze(0).int()))
    nonzero_indices = torch.nonzero(counts)
    first_nonzero_index = nonzero_indices[0][0]
    non_significant = torch.where(model.noise <= first_nonzero_index)[1].to(device)
    #print(len(non_significant))
    model.weight.data[:, non_significant] = torch.zeros(
        (model.weight.shape[0], len(non_significant))
    ).to(device)


    '''
        for i in range(model.out_features):
        counts = torch.bincount(torch.tensor(model.noise[i].squeeze(0).int()))
        nonzero_indices = torch.nonzero(counts)
        #first_nonzero_index = nonzero_indices[0][-1]
        #print(model.noise[i])
        #print(model.noise[i] > 1)
        #print(model.noise[i], first_nonzero_index)
        #print(model.noise[i] <= first_nonzero_index)
        non_significant = torch.where(model.noise[i] > 1)[0]
        #print(non_significant)
        model.weight.data[i, non_significant] = torch.zeros(
            len(non_significant)
        ).to(device)
        #print(model.weight.data[i])
    '''

    # l = 1

    # for i in range(model.weight.size(1)):
    #    _, topk_indices = torch.topk(torch.abs(model.weight[:,i]), k=l, dim=0, largest=False)
    #    model.weight[:,i][topk_indices] = 0


def test_noiseHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    #print(model)
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
