import os
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from tqdm import tqdm

import hdc
from hdc import functional
from hdc import embeddings

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.flatten = torch.nn.Flatten()

        self.position = embeddings.Random(size * size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        x = self.flatten(x)

        sample_hv = functional.bind(self.position.weight, self.value(x))
        sample_hv = functional.batch_bundle(sample_hv)
        return functional.soft_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


def experiment(settings, device=None):
    transform = torchvision.transforms.ToTensor()

    train_ds = MNIST("data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

    test_ds = MNIST("data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=24, shuffle=False)

    num_classes = len(train_ds.classes)

    model = Model(num_classes, IMG_SIZE)
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

    accuracy = hdc.metrics.Accuracy()

    start_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

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
        for resources in torch.linspace(1, 0, 11).tolist():
            settings = dict(resources=resources)
            metrics = experiment(settings, device=device)

            metrics = pd.DataFrame(metrics, index=[0])
            metrics["dataset"] = "MNIST"

            mode = "w" if is_first_result_write else "a"
            metrics.to_csv(
                args.result_file,
                mode=mode,
                header=is_first_result_write,
                index=False,
            )

            # make sure that the next results are appended to the same file
            is_first_result_write = False
