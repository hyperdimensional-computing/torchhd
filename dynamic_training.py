import os
import argparse
import time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import sklearn.metrics
from tqdm import tqdm

import hdc
import hdc.functional as HDF

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # To recreate an online learning setting
LEARNING_RATE = 0.005


class Model(nn.Module):
    def __init__(self, num_classes, size, train_embedding=False):
        super(Model, self).__init__()

        self.size = size

        self.pos_embed = hdc.embeddings.Random(size * size, DIMENSIONS)
        self.pos_embed.weight.requires_grad = train_embedding

        self.lum_embed = hdc.embeddings.Level(NUM_LEVELS, DIMENSIONS)
        self.lum_embed.weight.requires_grad = train_embedding

        self.classify = nn.Linear(DIMENSIONS, num_classes)
        self.classify.weight.data.fill_(0.0)
        self.classify.bias.data.fill_(0.0)

    def encode(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.size * self.size)

        luminosities = self.lum_embed(x)

        sample_hv = HDF.bind(self.pos_embed.weight, luminosities)
        sample_hv = torch.sum(sample_hv, dim=-2)

        return HDF.soft_quantize(sample_hv)  # cap between -1 and +1

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


def experiment(settings, device=None):

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    train_ds = torchvision.datasets.MNIST(
        "data", train=True, transform=transform, download=True
    )
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_ds = torchvision.datasets.MNIST(
        "data", train=False, transform=transform, download=True
    )
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_ds.classes)

    model = Model(num_classes, IMG_SIZE)
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
            enc = HDF.bundle(enc, cache[labels])
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

    pred_labels = []
    true_labels = []

    start_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Test"):
            samples = samples.to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=-1)

            pred_labels.append(predictions)
            true_labels.append(labels)

    end_time = time.time()
    test_duration = end_time - start_time
    print(f"Testing took {test_duration:.2f}s for {len(test_ds)} items")

    pred_labels = torch.cat(pred_labels).cpu()
    true_labels = torch.cat(true_labels).cpu()

    accuracy = sklearn.metrics.accuracy_score(pred_labels, true_labels)
    print(f"Testing accuracy of {(accuracy * 100):.3f}%")

    metrics = dict(
        accuracy=accuracy,
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
