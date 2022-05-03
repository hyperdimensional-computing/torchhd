import time

start_time = time.time()

# The following two lines are only needed because of this repository organization
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import torch.nn as nn
import torch.nn.functional as F
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

from torchhd import functional
from torchhd import embeddings
from torchhd.datasets.isolet import ISOLET

device = torch.device("cuda:2")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        sample_hv = functional.bind(self.id.weight, self.value(x))
        sample_hv = functional.batch_bundle(sample_hv)
        return functional.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


train_ds = ISOLET("../data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = ISOLET("../data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = Model(len(train_ds.classes), train_ds[0][0].size(-1))
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[labels] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

accuracy = torchmetrics.Accuracy()

with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        accuracy.update(predictions.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print("Duration", time.time() - start_time)
