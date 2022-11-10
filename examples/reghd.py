import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.datasets import AirfoilSelfNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_FEATURES = 5  # number of features in dataset

ds = AirfoilSelfNoise("../data", download=False)

# Get necessary statistics for data and target transform
STD_DEVS = ds.data.std(0)
MEANS = ds.data.mean(0)
TARGET_STD = ds.targets.std(0)
TARGET_MEAN = ds.targets.mean(0)


def transform(x):
    x = x - MEANS
    x = x / STD_DEVS
    return x


def target_transform(x):
    x = x - TARGET_MEAN
    x = x / TARGET_STD
    return x


ds.transform = transform
ds.target_transform = target_transform

# Split the dataset into 70% training and 30% testing
train_size = int(len(ds) * 0.7)
test_size = len(ds) - train_size
train_ds, test_ds = data.random_split(ds, [train_size, test_size])

train_dl = data.DataLoader(train_ds, batch_size=1, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=1)

# Model based on RegHD application for Single model regression
class SingleModel(nn.Module):
    def __init__(self, num_classes, size):
        super(SingleModel, self).__init__()

        self.lr = 0.00001
        self.M = torch.zeros(1, DIMENSIONS)
        self.project = embeddings.Projection(size, DIMENSIONS)
        self.project.weight.data.normal_(0, 1)
        self.bias = nn.parameter.Parameter(torch.empty(DIMENSIONS), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi)

    def encode(self, x):
        enc = self.project(x)
        sample_hv = torch.cos(enc + self.bias) * torch.sin(enc)
        return torchhd.hard_quantize(sample_hv)

    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0)

        self.M = update

    def forward(self, x):
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res


model = SingleModel(1, NUM_FEATURES)
model = model.to(device)

# Model training
with torch.no_grad():
    for _ in range(10):
        for samples, labels in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.model_update(samples_hv, labels)

# Model accuracy
mse = torchmetrics.MeanSquaredError()

with torch.no_grad():
    for samples, labels in tqdm(test_dl, desc="Testing"):
        samples = samples.to(device)

        predictions = model(samples)
        predictions = predictions * TARGET_STD + TARGET_MEAN
        labels = labels * TARGET_STD + TARGET_MEAN
        mse.update(predictions.cpu(), labels)

print(f"Testing mean squared error of {(mse.compute().item()):.3f}")
