import math
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.datasets import BeijingAirQuality

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

# Hardcoded dataset statistics [year, month, day, hour]
MIN_DATE = torch.tensor([2013, 1, 1, 0], dtype=torch.float)
MAX_DATE = torch.tensor([2017, 12, 31, 23], dtype=torch.float)

MIN_TEMPERATURE = -19.9000
MAX_TEMPERATURE = 41.6000


def transform(x):
    date = x.categorical[:4].float()
    date -= MIN_DATE
    date /= MAX_DATE - MIN_DATE

    temperature = x.continuous[6]
    return date, temperature


ds = BeijingAirQuality("../data", transform=transform, download=True)

# Remove samples with nan temperature value
has_temperature = ~ds.continuous_data[:, 6].isnan()
subset = torch.arange(0, len(ds))[has_temperature].tolist()
filtered_ds = data.Subset(ds, subset)

# Split data in 70% train and 30% test
train_size = int(len(filtered_ds) * 0.7)
test_size = len(filtered_ds) - train_size
train_ds, test_ds = data.random_split(filtered_ds, [train_size, test_size])

train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.target = embeddings.Level(
            500, DIMENSIONS, low=MIN_TEMPERATURE, high=MAX_TEMPERATURE
        )
        self.project = embeddings.Sinusoid(size, DIMENSIONS)

        self.regression = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.regression.weight.data.fill_(0.0)

    def encode(self, x):
        sample_hv = self.project(x)
        return torchhd.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)

        # Get the approximate target hv from the model
        target_hv = torchhd.bind(self.regression.weight, enc)

        # Get the index of the most similar target vector
        sim = torchhd.dot_similarity(target_hv, self.target.weight)
        index = torch.argmax(sim, dim=-1)

        # Convert the index of the hypervector back to the value it represents
        slope = MAX_TEMPERATURE - MIN_TEMPERATURE
        pred = index / 499 * slope + MIN_TEMPERATURE
        return pred


model = Model(1, 4)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        target_hv = model.target(labels)
        model.regression.weight.data += torchhd.bind(samples_hv, target_hv)

    model.regression.weight.copy_(F.normalize(model.regression.weight))

mse = torchmetrics.MeanSquaredError()

with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        predictions = model(samples)
        mse.update(predictions.cpu(), labels)

print(f"Testing mean squared error of {(mse.compute().item()):.3f}")
