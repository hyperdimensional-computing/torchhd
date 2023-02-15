import torch
import torch.nn as nn
import csv
import time
import torchmetrics
from tqdm import tqdm
import torch.utils.data as data

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets import UCIClassificationBenchmark

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using {} device".format(device))

BATCH_SIZE = 1


def experiment(
    DIMENSIONS=10000,
    method="SinusoidProjectionRegenerativeReset",
    epochs=5,
    drop_rate=0.2,
):
    def create_min_max_normalize(min, max):
        def normalize(input):
            return torch.nan_to_num((input - min) / (max - min))

        return normalize

    class Encoder(nn.Module):
        def __init__(self, size):
            super(Encoder, self).__init__()
            self.embed = embeddings.Sinusoid(size, DIMENSIONS)
            self.flatten = torch.nn.Flatten()

        def forward(self, x):
            x = self.flatten(x)
            sample_hv = self.embed(x).sign()
            return torchhd.hard_quantize(sample_hv)

    benchmark = UCIClassificationBenchmark("../data", download=True)
    results_file = "results/results" + str(time.time()) + ".csv"
    with open(results_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Name", "Accuracy", "Time", "Size", "Classes", "Dimensions", "Method"]
        )

    for dataset in benchmark.datasets():
        # print(dataset.name)

        # Number of features in the dataset.
        num_feat = dataset.train[0][0].size(-1)
        # Number of classes in the dataset.
        num_classes = len(dataset.train.classes)

        # Get values for min-max normalization and add the transformation
        min_val = torch.min(dataset.train.data, 0).values.to(device)
        max_val = torch.max(dataset.train.data, 0).values.to(device)
        transform = create_min_max_normalize(min_val, max_val)
        dataset.train.transform = transform
        dataset.test.transform = transform

        # Set up data loaders
        train_loader = data.DataLoader(
            dataset.train, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = data.DataLoader(dataset.test, batch_size=BATCH_SIZE)

        encode = Encoder(dataset.train[0][0].size(-1))
        encode = encode.to(device)

        model = Centroid(DIMENSIONS, num_classes)
        model = model.to(device)
        t = time.time()
        for i in range(epochs):
            with torch.no_grad():
                for samples, labels in tqdm(train_loader, desc="Training"):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    model.add_online(samples_hv, labels)
            model.normalize()

            if i < epochs - 1:
                model.regenerate_reset(encode.embed.weight, drop_rate)

        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():
            model.normalize()

            for samples, labels in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)

                samples_hv = encode(samples)
                outputs = model(samples_hv, dot=True)
                accuracy.update(outputs.cpu(), labels)

        with open(results_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    dataset.name,
                    accuracy.compute().item(),
                    time.time() - t,
                    len(dataset.train) + len(dataset.train),
                    num_classes,
                    DIMENSIONS,
                    method,
                ]
            )

experiment()