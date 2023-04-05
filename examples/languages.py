import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from torchhd import functional
from torchhd import embeddings
from torchhd.datasets import EuropeanLanguages as Languages
import sys
import time

device = torch.device("cpu")
DIMENSIONS = int(sys.argv[1])
BATCH_SIZE = 1
MAX_INPUT_SIZE = 128
PADDING_IDX = 0

ASCII_A = ord("a")
ASCII_Z = ord("z")
ASCII_SPACE = ord(" ")
NUM_TOKENS = ASCII_Z - ASCII_A + 3


def char2int(char: str) -> int:
    """Map a character to its integer identifier"""
    ascii_index = ord(char)
    if ascii_index == ASCII_SPACE:
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

t = time.time()


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.symbol = embeddings.Random(size, DIMENSIONS, padding_idx=PADDING_IDX)
        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        symbols = self.symbol(x)
        sample_hv = functional.ngrams(symbols, n=3)
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

correct_pred = 0

with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)
        labels = labels.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        if predictions == labels:
            correct_pred += 1
print(
    "language recognition,"
    + str(DIMENSIONS)
    + ","
    + str(time.time() - t)
    + ","
    + str((correct_pred / 21000)),
    end="",
)
