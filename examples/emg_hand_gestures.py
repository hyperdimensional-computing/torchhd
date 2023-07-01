import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets import EMGHandGestures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 21
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
WINDOW = 256
N_GRAM_SIZE = 4
DOWNSAMPLE = 5
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))


def transform(x):
    return x[SUBSAMPLES]


class Encoder(nn.Module):
    def __init__(self, size):
        super(Encoder, self).__init__()
        self.embed = embeddings.Density(size, DIMENSIONS)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)


def experiment(subjects=[0]):
    print("List of subjects " + str(subjects))
    ds = EMGHandGestures(
        "../data", download=True, subjects=subjects, transform=transform
    )

    train_size = int(len(ds) * 0.7)
    test_size = len(ds) - train_size
    train_ds, test_ds = data.random_split(ds, [train_size, test_size])

    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(ds[0][0].size(-1) * ds[0][0].size(-2))
    encode = Encoder(ds[0][0].size(-1) * ds[0][0].size(-2))
    encode = encode.to(device)

    num_classes = len(ds.classes)
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)

    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            targets = targets.to(device)

            sample_hv = encode(samples)
            print(sample_hv.shape)
            print(targets.shape)
            model.add_online(sample_hv, targets)

            break

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        model.normalize()

        for samples, targets in tqdm(test_ld, desc="Testing"):
            break
            samples = samples.to(device)

            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            print(output.cpu(), targets)
            accuracy.update(output.cpu(), targets)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


# Make a model for each subject
for i in range(5):
    experiment([i])
