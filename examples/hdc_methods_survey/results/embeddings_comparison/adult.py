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
from torchhd.models import Centroid
from torchhd.datasets import EMGHandGestures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS=10000
method="HashmapProjection"
levels=100
BATCH_SIZE = 1

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

def experiment():
    train = torchhd.datasets.Adult(
        "../../data", download=True, train=True
    )
    test = torchhd.datasets.Adult(
        "../../data", download=True, train=False
    )
    # Number of features in the dataset.
    # Number of classes in the dataset.
    num_classes = len(train.classes)

    # Get values for min-max normalization and add the transformation
    min_val = torch.min(train.data, 0).values.to(device)
    max_val = torch.max(train.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    train.transform = transform
    test.transform = transform

    # Set up data loaders
    train_loader = data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = data.DataLoader(test, batch_size=BATCH_SIZE)
    encode = Encoder(train[0][0].size(-1), levels)
    encode = encode.to(device)

    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    model2 = Centroid(DIMENSIONS, num_classes)
    model2 = model2.to(device)
    encode2 = Encoder(train[0][0].size(-1), levels)
    encode2 = encode2.to(device)

    model3 = Centroid(DIMENSIONS, num_classes)
    model3 = model3.to(device)
    encode3 = Encoder(train[0][0].size(-1), levels)
    encode3 = encode3.to(device)
    added_classes = {}
    wrong_inferred = {}
    count1 = 0
    count2 = 0
    t = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            samples_hv2 = encode2(samples)
            samples_hv3 = encode3(samples)

            model.add_online(samples_hv, labels)
            model2.add_online(samples_hv2, labels)
            model3.add_online(samples_hv3, labels)
            if labels.item() not in added_classes:
                added_classes[labels.item()] = 1
            else:
                added_classes[labels.item()] += 1

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    errors = 0
    corr = 0
    with torch.no_grad():
        model.normalize()

        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)

            samples_hv2 = encode2(samples)
            samples_hv3 = encode3(samples)
            outputs2 = model2(samples_hv2, dot=True)
            outputs3 = model3(samples_hv3, dot=True)
            predic = 0
            if (np.argmax(outputs).item() + np.argmax(outputs2).item() + np.argmax(outputs3).item()) >= 2:
                predic = 1
            if predic != labels.item():
                errors += 1
                #print('out', np.argmax(outputs).item())
                #print('out2', np.argmax(outputs2).item())
                if labels.item() not in wrong_inferred:
                    wrong_inferred[labels.item()] = 1
                wrong_inferred[labels.item()] += 1
            else:
                corr += 1
            '''
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
            '''
            accuracy.update(outputs.cpu(), labels)
    print('corr',corr/(corr+errors))
    print('wrong inferred',wrong_inferred)
    print(added_classes)
    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


experiment()
