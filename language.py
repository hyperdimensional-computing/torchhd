import os
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

from hdc import functional
from hdc import embeddings
from hdc.datasets import EuropeanLanguages as Languages

DIMENSIONS = 10000
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



def experiment(device=None):
    train_ds = Languages("data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

    test_ds = Languages("data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=24, shuffle=False)

    num_classes = len(train_ds.classes)

    model = Model(num_classes, NUM_TOKENS)
    model = model.to(device)

    start_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Train"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.classify.weight[labels] += samples_hv

        model.classify.weight[:] = F.normalize(model.classify.weight)

    end_time = time.time()
    train_duration = end_time - start_time
    print(f"Training took {train_duration:.3f}s for {len(train_ds)} items")

    accuracy = torchmetrics.Accuracy()

    start_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=-1)
            accuracy.update(predictions.cpu(), labels)

    end_time = time.time()
    test_duration = end_time - start_time
    print(f"Testing took {test_duration:.2f}s for {len(test_ds)} items")
    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

    metrics = dict(
        accuracy=accuracy.value().item(),
        train_duration=train_duration,
        train_set_size=len(train_ds),
        test_duration=test_duration,
        test_set_size=len(test_ds),
        dimensions=DIMENSIONS,
        device=str(device),
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-file")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)

    args = parser.parse_args()

    if os.path.isfile(args.result_file) and not args.append:
        raise FileExistsError(
            "The result file already exists. Run with --append flag to append data to the existing file."
        )

    os.makedirs(os.path.dirname(os.path.expanduser(args.result_file)), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # if a new file is generated first open then append
    is_first_result_write = not args.append

    for _ in range(args.repeats):
        metrics = experiment(device=device)

        metrics = pd.DataFrame(metrics, index=[0])
        metrics["dataset"] = "European Languages"

        mode = "w" if is_first_result_write else "a"
        metrics.to_csv(
            args.result_file,
            mode=mode,
            header=is_first_result_write,
            index=False,
        )

        # make sure that the next results are appended to the same file
        is_first_result_write = False