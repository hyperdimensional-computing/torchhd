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


def ngram(input, n, permute_hv):
    input = torchhd.as_vsa_model(input)
    n_gram = torchhd.bind(
        input[..., : -(n - 1), :],
        torch.unsqueeze(permute_hv[n - 2].repeat((input.shape[1] - (n - 1)), 1), 0),
    )
    for i in range(1, n):
        stop = None if i == (n - 1) else -(n - i - 1)
        if n - i - 1 == 0:
            sample = input[..., i:stop, :]
        else:
            sample = torchhd.bind(
                input[..., i:stop, :],
                torch.unsqueeze(
                    permute_hv[n - i - 2].repeat((input.shape[1] - (n - 1)), 1), 0
                ),
            )
        n_gram = torchhd.bind(n_gram, sample)
    return torchhd.multiset(n_gram)


class Encoder(nn.Module):
    def __init__(self, out_features, timestamps, channels):
        super(Encoder, self).__init__()

        self.channels = embeddings.Random(channels, out_features)
        self.timestamps = embeddings.Random(timestamps, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)
        self.permute_hv = torchhd.circular_hv(N_GRAM_SIZE - 1, out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        samples = torchhd.bind(signal, self.timestamps.weight.unsqueeze(1))

        samples = torchhd.multiset(samples)
        # sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        sample_hv = ngram(samples, N_GRAM_SIZE, self.permute_hv)

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

    encode = Encoder(DIMENSIONS, ds[0][0].size(-2), ds[0][0].size(-1))
    encode = encode.to(device)

    num_classes = len(ds.classes)
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)

    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            targets = targets.to(device)

            sample_hv = encode(samples)
            model.add(sample_hv, targets)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        model.normalize()

        for samples, targets in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            accuracy.update(output.cpu(), targets)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


# Make a model for each subject
for i in range(5):
    experiment([i])
