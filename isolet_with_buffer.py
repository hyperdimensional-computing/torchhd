import os
import argparse
import time
import random
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hdc import functional
from hdc import embeddings
from hdc import metrics
from hdc.datasets.isolet import Isolet

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
LEARNING_RATE = 0.005


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
        return functional.soft_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


def experiment(settings, device=None):
    train_ds = Isolet("../data", train=True, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds = Isolet("../data", train=False, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_ds.classes)
    sample_size = train_ds[0][0].size(-1)

    model = Model(num_classes, sample_size)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    cache = torch.zeros(num_classes, DIMENSIONS, device=device, dtype=torch.float)
    dirty_bit = torch.tensor([False] * num_classes, dtype=torch.bool, device=device)

    start_time = time.time()
    for samples, labels in tqdm(train_ld, desc="Train"):
        samples = samples.to(device)
        labels = labels.to(device)

        if random.random() < settings["resources"]:
            # zero the parameter gradients
            optimizer.zero_grad()

            enc = model.encode(samples)
            enc = functional.bundle(enc, cache[labels])
            cache[labels] = 0
            dirty_bit[labels] = False

            outputs = model.classify(enc)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                samples_hv = model.encode(samples)
                cache[labels] += samples_hv
                dirty_bit[labels] = True

    # Apply all accumulated samples
    # zero the parameter gradients
    optimizer.zero_grad()

    outputs = model.classify(cache[dirty_bit])
    labels = torch.arange(0, num_classes, device=device, dtype=torch.long)
    labels = labels[dirty_bit]

    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    end_time = time.time()
    train_duration = end_time - start_time
    print(f"Training took {train_duration:.3f}s for {len(train_ds)} items")

    accuracy = metrics.Accuracy()

    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=-1)

            accuracy.step(labels, predictions)

    end_time = time.time()
    test_duration = end_time - start_time
    print(f"Testing took {test_duration:.2f}s for {len(test_ds)} items")
    print(f"Testing accuracy of {(accuracy.value().item() * 100):.3f}%")

    metrics = dict(
        accuracy=accuracy.value().item(),
        resources=settings["resources"],
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
        for resources in torch.linspace(0, 1, 11).tolist():
            settings = dict(resources=resources)
            metrics = experiment(settings, device=device)

            metrics = pd.DataFrame(metrics, index=[0])
            metrics["dataset"] = "ISOLET"
            metrics["has_buffer"] = True

            mode = "w" if is_first_result_write else "a"
            metrics.to_csv(
                args.result_file,
                mode=mode,
                header=is_first_result_write,
                index=False,
            )

            # make sure that the next results are appended to the same file
            is_first_result_write = False
