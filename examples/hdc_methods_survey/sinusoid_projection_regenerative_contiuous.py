import torch
import torch.nn as nn
import csv
import time
import torchmetrics
from tqdm import tqdm
import torch.utils.data as data
import json
import os
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets import UCIClassificationBenchmark
import numpy as np

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using {} device".format(device))

BATCH_SIZE = 1


def experiment(
    DIMENSIONS=10000,
    method="SinusoidProjectionRegenerativeContinuous",
    epochs=5,
    drop_rate=0.2,
    filename="exp",
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

        added_classes = {}
        wrong_inferred = {}

        for i in range(epochs):
            with torch.no_grad():
                for samples, labels in tqdm(train_loader, desc="Training"):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    model.add_online(samples_hv, labels)
                    if labels.item() not in added_classes:
                        added_classes[labels.item()] = 1
                    else:
                        added_classes[labels.item()] += 1
            model.normalize()

            if i < epochs - 1:
                model.regenerate_continuous(encode.embed.weight, drop_rate, num_classes)

        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():
            model.normalize()

            for samples, labels in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)

                samples_hv = encode(samples)
                outputs = model(samples_hv, dot=True)
                out = outputs.cpu()
                if np.argmax(out).item() != labels.item():
                    if labels.item() not in wrong_inferred:
                        wrong_inferred[labels.item()] = 1
                    wrong_inferred[labels.item()] += 1
                accuracy.update(outputs.cpu(), labels)

            op = "r+"
            if os.path.exists("results/missclassified" + filename + ".json") == False:
                op = "x+"

            with open("results/missclassified" + filename + ".json", op) as outfile:
                try:
                    file_data = json.load(outfile)
                except:
                    file_data = {}
                if method not in file_data:
                    file_data[method] = {}
                    file_data = json.loads(json.dumps(file_data))
                if dataset.name not in file_data[method]:
                    file_data[method][dataset.name] = {}

                for i in wrong_inferred.keys():
                    if str(i) not in file_data[method][dataset.name]:
                        file_data[method][dataset.name][str(i)] = wrong_inferred[i]
                    else:
                        file_data[method][dataset.name][str(i)] += wrong_inferred[i]
                outfile.seek(0)
                # convert back to json.
                json.dump(file_data, outfile, indent=4)

            with open("results/trainsamples" + filename + ".json", op) as outfile:
                try:
                    file_data = json.load(outfile)
                except:
                    file_data = {}
                if method not in file_data:
                    file_data[method] = {}
                    file_data = json.loads(json.dumps(file_data))
                if dataset.name not in file_data[method]:
                    file_data[method][dataset.name] = {}

                for i in added_classes.keys():
                    if str(i) not in file_data[method][dataset.name]:
                        file_data[method][dataset.name][str(i)] = added_classes[i]
                    else:
                        file_data[method][dataset.name][str(i)] += added_classes[i]
                outfile.seek(0)
                # convert back to json.
                json.dump(file_data, outfile, indent=4)

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
