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
from torchhd.datasets import EuropeanLanguages as Languages

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
# cap maximum sample size to 128 characters (including spaces)
MAX_INPUT_SIZE = 128
PADDING_IDX = 0

ASCII_A = ord("a")
ASCII_Z = ord("z")
ASCII_SPACE = ord(" ")
NUM_TOKENS = ASCII_Z - ASCII_A + 3  # a through z plus space and padding


def char2int(char: str) -> int:
    """Map a character to its integer identifier"""
    ascii_index = ord(char)

    if ascii_index == ASCII_SPACE:
        # Remap the space character to come after "z"
        return ASCII_Z - ASCII_A + 1

    return ascii_index - ASCII_A


def transform(x: str) -> torch.Tensor:
    char_ids = x[:MAX_INPUT_SIZE]
    char_ids = [char2int(char) + 1 for char in char_ids.lower()]

    if len(char_ids) < MAX_INPUT_SIZE:
        char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

    return torch.tensor(char_ids, dtype=torch.long)


train_ds = Languages("../data", train=True, transform=transform, download=True)
train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = Languages("../data", train=False, transform=transform, download=True)
test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, out_features, size):
        super(Encoder, self).__init__()
        self.symbol = embeddings.Random(size, out_features, padding_idx=PADDING_IDX)

    def forward(self, x):
        print(self.symbol.weight)

        symbols = self.symbol(x)
        sample_hv = torchhd.ngrams(symbols, n=3)
        return torchhd.hard_quantize(sample_hv)


encode = Encoder(DIMENSIONS, NUM_TOKENS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
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
        labels = labels.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
