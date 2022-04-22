# The following two lines are only needed because of this repository organization
from random import sample
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hdc import functional
from hdc import embeddings
from hdc import metrics
from hdc.datasets import EuropeanLanguages as Languages

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
    char_ids = [char2int(char) + 1 for char in x.lower()]

    if len(char_ids) < MAX_INPUT_SIZE:
        char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

    return torch.tensor(char_ids, dtype=torch.long)


train_ds = Languages("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = Languages("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.symbol = embeddings.Random(size, DIMENSIONS, padding_idx=PADDING_IDX)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        symbols = self.symbol(x)

        first = functional.permute(symbols[:, 0:-2], shifts=2)
        second = functional.permute(symbols[:, 1:-1])
        third = symbols[:, 2:None]

        sample_hv = functional.bind(first, second)
        sample_hv = functional.bind(sample_hv, third)
        sample_hv = functional.batch_bundle(sample_hv)
        return functional.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


model = Model(len(train_ds.classes), NUM_TOKENS)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[labels] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

accuracy = metrics.Accuracy()

with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)

        accuracy.step(labels, predictions)

print(f"Testing accuracy of {(accuracy.value().item() * 100):.3f}%")
