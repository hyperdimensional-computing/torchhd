import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

# Note: this example requires the accelerate library: https://github.com/huggingface/accelerate
from accelerate import Accelerator
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

accelerator = Accelerator()
device = accelerator.device
print("Using {} device".format(device))

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


class Model(nn.Module):
    def __init__(self, dimensions, num_classes, size, levels):
        super(Model, self).__init__()
        self.encode = Encoder(dimensions, size, levels)
        self.classify = Centroid(dimensions, num_classes)

    def forward(self, x, dot=False):
        y = self.encode(x)
        return self.classify(y, dot=dot)


num_classes = len(train_ds.classes)
model = Model(DIMENSIONS, num_classes, IMG_SIZE, NUM_LEVELS)
model.to(device)

model, train_ld, test_ld = accelerator.prepare(model, train_ld, test_ld)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        model.classify.add(samples_hv, labels)

accuracy = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=num_classes)

with torch.no_grad():
    model.classify.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        outputs = model(samples, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
