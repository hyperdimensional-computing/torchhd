import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm
import torchmetrics
import torchhd
from torchhd.datasets import HDCArena, UCIClassificationBenchmark
from torchhd import embeddings
from torchhd.models import Centroid, CentroidIterative
import time
import csv
import torch.nn.functional as F
import pandas as pd
from SinglePass import vanillaHD, adaptHD, onlineHD, multiCentroidHD
from Iterative import adaptHD as adaptHDiterative
from Iterative import onlineHD as onlineHDiterative
from Iterative import quantHD as quantHDiterative
from Iterative import sparseHD as sparseHDiterative
from Iterative import neuralHD as neuralHDiterative
from Iterative import distHD as distHDiterative

pd.set_option("display.max_columns", None)
from collections import deque


# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input.to(device) - min) / (max - min))

    return normalize


# Specify device to be used for Torch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# Specifies batch size to be used for the model.
class Encoder(nn.Module):
    def __init__(self, size, dimensions, encoding, name):
        super(Encoder, self).__init__()
        self.encoding = encoding
        if self.encoding == "bundle":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "hashmap":
            levels = 100
            if name == "EuropeanLanguages":
                self.keys = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.keys = embeddings.Random(size, dimensions)
            self.embed = embeddings.Level(levels, dimensions)
        if self.encoding == "ngram":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "sequence":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "random":
            self.embed = embeddings.Projection(size, dimensions)
        if self.encoding == "sinusoid":
            self.embed = embeddings.Sinusoid(size, dimensions)
        if self.encoding == "density":
            self.embed = embeddings.Density(size, dimensions)
        if self.encoding == "flocet":
            self.embed = embeddings.DensityFlocet(size, dimensions)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x).float()
        if self.encoding == "bundle":
            sample_hv = torchhd.multiset(self.embed(x.long()))
        if self.encoding == "hashmap":
            sample_hv = torchhd.hash_table(self.keys.weight, self.embed(x))
        if self.encoding == "ngram":
            sample_hv = torchhd.ngrams(self.embed(x.long()), n=3)
        if self.encoding == "sequence":
            sample_hv = torchhd.ngrams(self.embed(x.long()), n=1)
        if self.encoding == "random":
            sample_hv = self.embed(x).sign()
        if self.encoding == "sinusoid":
            sample_hv = self.embed(x).sign()
        if self.encoding == "density":
            sample_hv = self.embed(x).sign()
        if self.encoding == "flocet":
            sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)

    def neural_regeneration(self, idx):
        nn.init.normal_(self.embed.weight[idx], 0, 1)
        self.embed.weight[idx] = F.normalize(self.embed.weight)[idx]


# Get an instance of the UCI benchmark

# Perform evaluation
results_file = "results/results" + str(time.time()) + ".csv"

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Name",
            "Accuracy",
            "TrainTime",
            "TestTime",
            "Dimensions",
            "Method",
            "Encoding",
            "Iterations",
        ]
    )


def exec_arena(
    method="add",
    encoding="density",
    iterations=1,
    dimensions=10,
    repeats=1,
    batch_size=1,
    lr=5,
):
    for dataset in benchmark.datasets():
        for r in range(repeats):
            print(dataset.name)
            if dataset.name == "EuropeanLanguages":
                from torchhd.datasets import EuropeanLanguages as Languages

                MAX_INPUT_SIZE = 128
                PADDING_IDX = 0

                ASCII_A = ord("a")
                ASCII_Z = ord("z")
                ASCII_SPACE = ord(" ")

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

                num_feat = MAX_INPUT_SIZE

                train_ds = Languages(
                    "../data", train=True, transform=transform, download=True
                )

                test_ds = Languages(
                    "../data", train=False, transform=transform, download=True
                )
                num_classes = len(train_ds.classes)
                train_loader = data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True
                )
                test_loader = data.DataLoader(test_ds, batch_size=batch_size)
            elif dataset.name in ["PAMAP", "EMGHandGestures"]:
                if dataset.name == "EMGHandGestures":
                    num_feat = dataset.train[0][0].size(-1) * dataset.train[0][0].size(
                        -2
                    )
                else:
                    num_feat = dataset.train[0][0].size(-1)

                # Number of classes in the dataset.
                num_classes = len(dataset.train.classes)
                # Number of training samples in the dataset.
                num_train_samples = len(dataset.train)

                # Get values for min-max normalization and add the transformation
                min_val = torch.min(dataset.train.data, 0).values.to(device)
                max_val = torch.max(dataset.train.data, 0).values.to(device)
                transform = create_min_max_normalize(min_val, max_val)
                dataset.train.transform = transform

                train_size = int(len(dataset.train) * 0.7)
                test_size = len(dataset.train) - train_size
                train_ds, test_ds = data.random_split(
                    dataset.train, [train_size, test_size]
                )
                train_loader = data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True
                )
                test_loader = data.DataLoader(test_ds, batch_size=batch_size)
            else:
                # Number of features in the dataset.
                if dataset.name not in ["MNIST", "CIFAR10"]:
                    num_feat = dataset.train[0][0].size(-1)
                else:
                    if dataset.name == "MNIST":
                        num_feat = dataset.train[0][0].size(-1) * dataset.train[0][
                            0
                        ].size(-1)
                    elif dataset.name == "CIFAR10":
                        num_feat = 3072
                # Number of classes in the dataset.
                num_classes = len(dataset.train.classes)
                # Number of training samples in the dataset.
                num_train_samples = len(dataset.train)
                # Get values for min-max normalization and add the transformation

                if dataset.name not in ["MNIST", "CIFAR10"]:
                    min_val = torch.min(dataset.train.data, 0).values.to(device)
                    max_val = torch.max(dataset.train.data, 0).values.to(device)
                    transform = create_min_max_normalize(min_val, max_val)
                    dataset.train.transform = transform
                    dataset.test.transform = transform

                # Set up data loaders
                train_loader = data.DataLoader(
                    dataset.train, batch_size=batch_size, shuffle=True
                )
                test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

            encode = Encoder(num_feat, dimensions, encoding, dataset.name)
            encode = encode.to(device)

            model = Centroid(dimensions, num_classes)
            model = model.to(device)

            # TRAIN #
            t = time.time()
            if method == "add":
                vanillaHD.train_vanillaHD(train_loader, device, encode, model)
                iterations_executed = 1
            elif method == "adapt":
                adaptHD.train_adaptHD(train_loader, device, encode, model)
                iterations_executed = 1
            elif method == "online":
                onlineHD.train_onlineHD(train_loader, device, encode, model)
                iterations_executed = 1
            elif method == "adapt_iterative":
                iterations_executed = adaptHDiterative.train_adaptHD(
                    train_loader, device, encode, model, iterations, num_classes, lr
                )
            elif method == "online_iterative":
                iterations_executed = onlineHDiterative.train_onlineHD(
                    train_loader, device, encode, model, iterations, num_classes, lr
                )
            elif method == "quant_iterative":
                # Suggested lr in the paper
                lr = 1.5
                epsilon = 0.01
                model_quantize = "binary"
                iterations_executed = quantHDiterative.train_quantHD(
                    train_loader,
                    device,
                    encode,
                    model,
                    model_quantize,
                    iterations,
                    num_classes,
                    lr,
                    epsilon,
                )

            elif method == "sparse_iterative":
                # Suggested lr in the paper
                lr = 1
                epsilon = 0.01
                model_sparse = "class"
                sparsity = 0.5
                iterations_executed = sparseHDiterative.train_sparseHD(
                    train_loader,
                    device,
                    encode,
                    model,
                    model_sparse,
                    iterations,
                    num_classes,
                    lr,
                    epsilon,
                    s=sparsity,
                    dimensions=dimensions,
                )
            elif method == "neural_iterative":
                # Suggested lr in the paper
                lr = 1
                model_neural = "reset"
                iterations_executed = neuralHDiterative.train_neuralHD(
                    train_loader,
                    device,
                    encode,
                    model,
                    iterations,
                    model_neural,
                    lr,
                    dimensions=dimensions,
                )
            elif method == "dist_iterative":
                # Suggested lr in the paper
                lr = 1
                iterations = 30
                iterations_executed = distHDiterative.train_distHD(
                    train_loader,
                    device,
                    encode,
                    model,
                    iterations,
                    lr=1,
                    r=0.05,
                    alpha=4,
                    beta=2,
                    theta=1,
                    dimensions=dimensions,
                )
            elif method == "multicentroid":
                iterations_executed = 1
                reduce_subclasses = "drop"
                threshold = 0.03
                multiCentroidHD.train_multicentroidHD(
                    train_loader,
                    device,
                    encode,
                    model,
                    num_classes,
                    reduce_subclasses=reduce_subclasses,
                    threshold=threshold,
                )

            train_time = time.time() - t

            # TEST #

            accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(
                device
            )
            t = time.time()
            if method == "add":
                vanillaHD.test_vanillaHD(test_loader, device, encode, model, accuracy)
            elif method == "adapt":
                adaptHD.test_adaptHD(test_loader, device, encode, model, accuracy)
            elif method == "online":
                onlineHD.test_onlineHD(test_loader, device, encode, model, accuracy)
            elif method == "adapt_iterative":
                adaptHDiterative.test_adaptHD(
                    test_loader, device, encode, model, accuracy
                )
            elif method == "online_iterative":
                onlineHDiterative.test_onlineHD(
                    test_loader, device, encode, model, accuracy
                )
            elif method == "quant_iterative":
                quantHDiterative.test_quantHD(
                    test_loader, device, encode, model, accuracy, model_quantize
                )
            elif method == "sparse_iterative":
                sparseHDiterative.test_sparseHD(
                    test_loader, device, encode, model, accuracy
                )
            elif method == "neural_iterative":
                neuralHDiterative.test_neuralHD(
                    test_loader, device, encode, model, accuracy
                )
            elif method == "dist_iterative":
                distHDiterative.test_distHD(
                    test_loader, device, encode, model, accuracy
                )
            elif method == "multicentroid":
                multiCentroidHD.test_multicentroidHD(
                    test_loader, device, encode, model, accuracy
                )
            test_time = time.time() - t

            benchmark.report(dataset, accuracy.compute().item())
            with open(results_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        dataset.name,
                        accuracy.compute().item(),
                        train_time,
                        test_time,
                        dimensions,
                        method,
                        encoding,
                        iterations_executed,
                    ]
                )


BATCH_SIZE = 1
REPEATS = 1
DIMENSIONS = [10000]

# ENCODINGS = ["bundle", "sequence", "ngram", "hashmap", "flocet", "density", "random", "sinusoid"]
ENCODINGS = [
    "hashmap",
]
# METHODS = ["add",
# "adapt",
# "online",
# "adapt_iterative",
# "online_iterative",
# "quant_iterative",
# "sparse_iterative",
# "neural_iterative",
# "dist_iterative",
# "multicentroid"]
METHODS = [
    # "quant_iterative",
    "sparse_iterative",
    "neural_iterative",
    "dist_iterative",
    # "multicentroid",
]

ITERATIONS = 30
arena = True

if arena:
    benchmark = HDCArena("../data", download=True)

else:
    benchmark = UCIClassificationBenchmark("../data", download=True)

print(benchmark.datasets())

for i in DIMENSIONS:
    for j in ENCODINGS:
        for k in METHODS:
            exec_arena(
                encoding=j,
                method=k,
                dimensions=i,
                repeats=REPEATS,
                batch_size=BATCH_SIZE,
                iterations=ITERATIONS,
            )
