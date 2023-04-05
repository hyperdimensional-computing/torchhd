import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchhd.datasets import UCIHAR
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
NUM_LEVELS = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

ds = UCIHAR(
        "../data", download=True
)

train_size = int(len(ds) * 0.8)
test_size = len(ds) - train_size
train_ds, test_ds = data.random_split(ds, [train_size, test_size])

train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)



'''class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        #x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        #sample_hv = torchhd.ngrams(sample_hv, 4)
        return torchhd.hard_quantize(sample_hv)'''

class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.embed = embeddings.Density(size, DIMENSIONS)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)

encode = Encoder(DIMENSIONS, ds[0][0].size(-1))
encode = encode.to(device)

num_classes =len(ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add(samples_hv, labels)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
