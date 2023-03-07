import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
import numpy as np
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid, MemoryModel
from torchhd.datasets import EMGHandGestures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1000
method = "HashmapProjection"
levels = 100
BATCH_SIZE = 1
# corr 0.7791290461273878
# corr2 0.7779620416436337
# wrong inferred {0: 1845, 1: 1753}
# {0: 24720, 1: 7841}
# Testing accuracy of 73.558%


class Encoder(nn.Module):
    def __init__(self, size, levels):
        super(Encoder, self).__init__()
        self.keys = embeddings.Random(size, DIMENSIONS)
        self.values = embeddings.Level(levels, DIMENSIONS)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.hash_table(self.keys.weight, self.values(x))
        return torchhd.hard_quantize(sample_hv)


def create_min_max_normalize(min, max):
    def normalize(input):
        return torch.nan_to_num((input - min) / (max - min))

    return normalize


def normalize(w, eps=1e-12) -> None:
    """Transforms all the class prototype vectors into unit vectors.

    After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
    Training further after calling this method is not advised.
    """
    norms = w.norm(dim=1, keepdim=True)
    norms.clamp_(min=eps)
    w.div_(norms)


def experiment():
    train = torchhd.datasets.Adult("../../data", download=True, train=True)
    test = torchhd.datasets.Adult("../../data", download=True, train=False)
    # Number of features in the dataset.
    # Number of classes in the dataset.
    num_classes = len(train.classes)

    # Get values for min-max normalization and add the transformation
    min_val = torch.min(train.data, 0).values.to(device)
    max_val = torch.max(train.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    train.transform = transform
    test.transform = transform

    batch = len(train)
    print(batch)

    # Set up data loaders
    train_loader = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=BATCH_SIZE)

    train_data = data.DataLoader(train, batch_size=batch, shuffle=True)
    m = MemoryModel(DIMENSIONS, num_classes)
    proj = embeddings.Projection(train[0][0].size(-1), DIMENSIONS)

    for i in train_data:
        # print(m.classes.weight.shape)
        # print(i[1].shape)
        inputs = proj(i[0])
        torchhd.hard_quantize(inputs)
        labels = torch.index_select(m.classes.weight, 0, i[1])
        # print('i',i[0].shape,'i',i[1][0])
        print(inputs.shape, labels.shape)

        res = torch.matmul(inputs.T, labels).data
    print(normalize(res))
    print(res)
    count = 0
    error = 0
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

            input = proj(samples)
            input = torchhd.hard_quantize(input)
            a = torch.matmul(input, res)

            print(torchhd.dot_similarity(a, m.classes.weight))
            print(m.classes)
            predict = np.argmax(torchhd.dot_similarity(a, m.classes.weight)).item()
            if predict == labels.item():
                count += 1
            else:
                error += 1
            break
    print("Accuracy", count / (count + error))
    '''
    num_models = 1
    models = []
    encoders = []
    for i in range(num_models):
        encode = Encoder(train[0][0].size(-1), levels)
        encode = encode.to(device)
        encoders.append(encode)
        model = Centroid(DIMENSIONS, num_classes)
        model = model.to(device)
        models.append(model)

    added_classes = {}
    wrong_inferred = {}




    t = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            for i in range(num_models):
                samples_hv = encoders[i](samples)
                models[i].add_online(samples_hv, labels)

            if labels.item() not in added_classes:
                added_classes[labels.item()] = 1
            else:
                added_classes[labels.item()] += 1

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    errors = 0
    corr = 0

    errors2 = 0
    corr2 = 0
    with torch.no_grad():

        for i in models:
            i.normalize()

        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            outputs = []
            vals = []
            vals2 = []
            differ = []
            for i in range(num_models):
                samples_hv = encoders[i](samples)
                out = models[i](samples_hv, dot=True)
                vals.append(np.argmax(out).item())
                vals2.append(np.argmin(torch.cdist(encoders[i](samples), models[i].weight, p=3.0)).item())
                differ.append(abs(out[0][0].item() - out[0][1].item()))
                outputs.append(out)

            if np.argmax(outputs[np.argmax(differ)]).item() != labels.item():
                errors2 += 1
            else:
                corr2 += 1
            predic = np.argmax(np.bincount(vals))
            predic2 = np.argmax(np.bincount(vals2))
            if predic2 != predic:
                print(labels.item(), predic, predic2)
            if predic != labels.item():
                errors += 1
                # print('out', np.argmax(outputs).item())
                # print('out2', np.argmax(outputs2).item())
                if labels.item() not in wrong_inferred:
                    wrong_inferred[labels.item()] = 1
                wrong_inferred[labels.item()] += 1
            else:
                corr += 1
            """
            for i in model.weight:
                out2.append(torch.nn.functional.pairwise_distance(i, samples_hv))
            if np.argmax(out2).item() != labels.item():
                errors += 1
            if np.argmax(out).item() != labels.item():
                print('prediction', out)
                for i in model.weight:
                    print('dist',torch.nn.functional.pairwise_distance(i,samples_hv))
                print('label', labels)
                if labels.item() not in wrong_inferred:
                    wrong_inferred[labels.item()] = 1
                wrong_inferred[labels.item()] += 1
            """
            accuracy.update(outputs[0].cpu(), labels)
    print("corr", corr / (corr + errors))
    print("corr2", corr2 / (corr2 + errors2))
    print("wrong inferred", wrong_inferred)
    print(added_classes)
    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    '''


experiment()
