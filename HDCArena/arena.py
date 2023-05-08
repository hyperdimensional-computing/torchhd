import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm
import torchmetrics
import torchhd
from torchhd.datasets import HDCArena, UCIClassificationBenchmark
from torchhd import embeddings
from torchhd.models import Centroid, CentroidIterative, IntRVFL
import time
import csv
import torch.nn.functional as F
import pandas as pd
from SinglePass import vanillaHD, highHD, adaptHD, onlineHD, multiCentroidHD, intRVFL
from Iterative import adaptHD as adaptHDiterative
from Iterative import onlineHD as onlineHDiterative
from Iterative import quantHD as quantHDiterative
from Iterative import sparseHD as sparseHDiterative
from Iterative import neuralHD as neuralHDiterative
from Iterative import distHD as distHDiterative

pd.set_option("display.max_columns", None)
from collections import deque

INT_RVFL_HYPER = {
    "abalone": (1450, 32, 15),
    "acute-inflammation": (50, 0.0009765625, 1),
    "acute-nephritis": (50, 0.0009765625, 1),
    "adult": (1150, 0.0625, 3),
    "annealing": (1150, 0.015625, 7),
    "arrhythmia": (1400, 0.0009765625, 7),
    "audiology-std": (950, 16, 3),
    "balance-scale": (50, 32, 7),
    "balloons": (50, 0.0009765625, 1),
    "bank": (200, 0.001953125, 7),
    "blood": (50, 16, 7),
    "breast-cancer": (50, 32, 1),
    "breast-cancer-wisc": (650, 16, 3),
    "breast-cancer-wisc-diag": (1500, 2, 3),
    "breast-cancer-wisc-prog": (1450, 0.01562500, 3),
    "breast-tissue": (1300, 0.1250000, 1),
    "car": (250, 32, 3),
    "cardiotocography-10clases": (1350, 0.0009765625, 3),
    "cardiotocography-3clases": (900, 0.007812500, 15),
    "chess-krvk": (800, 4, 1),
    "chess-krvkp": (1350, 0.01562500, 3),
    "congressional-voting": (100, 32, 15),
    "conn-bench-sonar-mines-rocks": (1100, 0.01562500, 3),
    "conn-bench-vowel-deterding": (1350, 8, 3),
    "connect-4": (1100, 0.5, 3),
    "contrac": (50, 8, 7),
    "credit-approval": (200, 32, 7),
    "cylinder-bands": (1100, 0.0009765625, 7),
    "dermatology": (900, 8, 3),
    "echocardiogram": (250, 32, 15),
    "ecoli": (350, 32, 3),
    "energy-y1": (650, 0.1250000, 3),
    "energy-y2": (1000, 0.0625, 7),
    "fertility": (150, 32, 7),
    "flags": (900, 32, 15),
    "glass": (1400, 0.03125000, 3),
    "haberman-survival": (100, 32, 3),
    "hayes-roth": (50, 16, 1),
    "heart-cleveland": (50, 32, 15),
    "heart-hungarian": (50, 16, 15),
    "heart-switzerland": (50, 8, 15),
    "heart-va": (1350, 0.1250000, 15),
    "hepatitis": (1300, 0.03125000, 1),
    "hill-valley": (150, 0.01562500, 1),
    "horse-colic": (850, 32, 1),
    "ilpd-indian-liver": (1200, 0.25, 7),
    "image-segmentation": (650, 8, 1),
    "ionosphere": (1150, 0.001953125, 1),
    "iris": (50, 4, 3),
    "led-display": (50, 0.0009765625, 7),
    "lenses": (50, 0.03125000, 1),
    "letter": (1500, 32, 1),
    "libras": (1250, 0.1250000, 3),
    "low-res-spect": (1400, 8, 7),
    "lung-cancer": (450, 0.0009765625, 1),
    "lymphography": (1150, 32, 1),
    "magic": (800, 16, 3),
    "mammographic": (150, 16, 7),
    "miniboone": (650, 0.0625, 15),
    "molec-biol-promoter": (1250, 32, 1),
    "molec-biol-splice": (1000, 8, 15),
    "monks-1": (50, 4, 3),
    "monks-2": (400, 32, 1),
    "monks-3": (50, 4, 15),
    "mushroom": (150, 0.25, 3),
    "musk-1": (1300, 0.001953125, 7),
    "musk-2": (1150, 0.007812500, 7),
    "nursery": (1000, 32, 3),
    "oocytes_merluccius_nucleus_4d": (1500, 1, 7),
    "oocytes_merluccius_states_2f": (1500, 0.0625, 7),
    "oocytes_trisopterus_nucleus_2f": (1450, 0.003906250, 3),
    "oocytes_trisopterus_states_5b": (1450, 2, 7),
    "optical": (1100, 32, 7),
    "ozone": (50, 0.003906250, 1),
    "page-blocks": (800, 0.001953125, 1),
    "parkinsons": (1200, 0.5, 1),
    "pendigits": (1500, 0.1250000, 1),
    "pima": (50, 32, 1),
    "pittsburg-bridges-MATERIAL": (100, 8, 1),
    "pittsburg-bridges-REL-L": (1200, 0.5, 1),
    "pittsburg-bridges-SPAN": (450, 4, 7),
    "pittsburg-bridges-T-OR-D": (1000, 16, 1),
    "pittsburg-bridges-TYPE": (50, 32, 7),
    "planning": (50, 32, 1),
    "plant-margin": (1350, 2, 7),
    "plant-shape": (1450, 0.25, 3),
    "plant-texture": (1500, 4, 7),
    "post-operative": (50, 32, 15),
    "primary-tumor": (950, 32, 3),
    "ringnorm": (1500, 0.125, 3),
    "seeds": (550, 32, 1),
    "semeion": (1400, 32, 15),
    "soybean": (850, 1, 3),
    "spambase": (1350, 0.0078125, 15),
    "spect": (50, 32, 1),
    "spectf": (1100, 0.25, 15),
    "statlog-australian-credit": (200, 32, 15),
    "statlog-german-credit": (500, 32, 15),
    "statlog-heart": (50, 32, 7),
    "statlog-image": (950, 0.125, 1),
    "statlog-landsat": (1500, 16, 3),
    "statlog-shuttle": (100, 0.125, 7),
    "statlog-vehicle": (1450, 0.125, 7),
    "steel-plates": (1500, 0.0078125, 3),
    "synthetic-control": (1350, 16, 3),
    "teaching": (400, 32, 3),
    "thyroid": (300, 0.001953125, 7),
    "tic-tac-toe": (750, 8, 1),
    "titanic": (50, 0.0009765625, 1),
    "trains": (100, 16, 1),
    "twonorm": (1100, 0.0078125, 15),
    "vertebral-column-2clases": (250, 32, 3),
    "vertebral-column-3clases": (200, 32, 15),
    "wall-following": (1200, 0.00390625, 3),
    "waveform": (1400, 8, 7),
    "waveform-noise": (1300, 0.0009765625, 15),
    "wine": (850, 32, 1),
    "wine-quality-red": (1100, 32, 1),
    "wine-quality-white": (950, 8, 3),
    "yeast": (1350, 4, 1),
    "zoo": (400, 8, 7),
}


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
        if self.encoding == "generic":
            levels = 100
            self.keys = embeddings.Random(size, dimensions)
            self.embed = embeddings.Level(levels, dimensions)
        if self.encoding == "hashmapGen":
            levels = 100
            self.keys = embeddings.Sinusoid(size, dimensions)
            self.embed = embeddings.Level(levels, dimensions)
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
        if self.encoding == "generic":
            sample_hv = torchhd.functional.generic(
                self.keys.weight.to(device), self.embed(x).to(device), 3, device
            ).to(device)
        if self.encoding == "hashmapGen":
            sample_hv = torchhd.hash_table(
                self.keys(x).sign(), self.embed(x).to(device)
            )
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
                    train_ds = dataset.train
                    dataset.test.transform = transform
                    test_ds = dataset.test

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
            elif method == "add_high":
                highHD.train_highHD(train_loader, device, encode, model)
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
            elif method == "rvfl":
                if not arena:
                    _, alpha, kappa = INT_RVFL_HYPER[dataset.train.name]
                else:
                    alpha = 1
                    kappa = 3
                iterations_executed = 1
                model = IntRVFL(
                    num_feat, dimensions, num_classes, kappa=kappa, device=device
                )
                intRVFL.train_rvfl(train_ds, encode, model, device, num_classes, alpha)
            train_time = time.time() - t

            # TEST #

            accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(
                device
            )
            t = time.time()
            if method == "add":
                vanillaHD.test_vanillaHD(test_loader, device, encode, model, accuracy)
            elif method == "add_high":
                highHD.test_highHD(test_loader, device, encode, model, accuracy)
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
            elif method == "rvfl":
                intRVFL.test_rvfl(test_loader, device, encode, model, accuracy)
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
            # print(accuracy.compute().item())


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
    "add",
    "adapt",
    "online",
    "adapt_iterative",
    "online_iterative",
    "quant_iterative",
    "sparse_iterative",
    "neural_iterative",
    "dist_iterative",
    "multicentroid"
    "rvfl",
]

ITERATIONS = 30
arena = False

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
