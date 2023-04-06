import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm
import torchmetrics
import torchhd
from torchhd.datasets import HDCArena
from torchhd import embeddings
from torchhd.models import Centroid
import time
import csv

# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input - min) / (max - min))

    return normalize

# Specify device to be used for Torch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
# Specifies batch size to be used for the model.



class Encoder(nn.Module):

    def __init__(self, size, dimensions, method):
        super(Encoder, self).__init__()
        self.method = method
        if self.method == 'bundle':
            self.symbol = embeddings.Random(size, dimensions)
        if self.method == 'hashmap':
            levels = 1
            self.keys = embeddings.Random(size, dimensions)
            self.values = embeddings.Level(levels, dimensions)
        if self.method == 'ngram':
            self.symbol = embeddings.Random(size, dimensions)
        if self.method == 'sequence':
            self.symbol = embeddings.Random(size, dimensions)
        if self.method == 'random':
            self.embed = embeddings.Projection(size, dimensions)
        if self.method == 'sinusoid':
            self.embed = embeddings.Projection(size, dimensions)
        if self.method == 'density':
            self.embed = embeddings.Density(size, dimensions)

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x).float()
        if self.method == 'bundle':
            sample_hv = torchhd.multiset(self.symbol(x.long()))
        if self.method == 'hashmap':
            sample_hv = torchhd.hash_table(self.keys.weight, self.values(x))
        if self.method == 'ngram':
            sample_hv = torchhd.ngrams(self.symbol(x.long()), n=3)
        if self.method == 'sequence':
            sample_hv = torchhd.ngrams(self.symbol(x.long()), n=1)
        if self.method == 'random':
            sample_hv = self.embed(x).sign()
        if self.method == 'sinusoid':
            sample_hv = self.embed(x).sign()
        if self.method == 'density':
            sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)



# Get an instance of the UCI benchmark
benchmark = HDCArena("../data", download=True)
# Perform evaluation
results_file = "results/results" + str(time.time()) + ".csv"

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Name", "Accuracy", "Time", "Dimensions", "Method"]
    )

def exec_arena(method='density', dimensions=1,repeats=1,batch_size=1):
    for dataset in benchmark.datasets():
        print(dataset.name)
        if dataset.name == 'EuropeanLanguages':
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

            train_ds = Languages("../data", train=True, transform=transform, download=True)
            train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

            test_ds = Languages("../data", train=False, transform=transform, download=True)
            test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            num_classes = len(train_ds.classes)


        elif dataset.name in ['PAMAP','EMGHandGestures']:
            if dataset.name == 'EMGHandGestures':
                num_feat = dataset.train[0][0].size(-1)*dataset.train[0][0].size(-2)
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
            train_ds, test_ds = data.random_split(dataset.train, [train_size, test_size])

            train_loader = data.DataLoader(train_ds, batch_size=batch_size)
            test_loader = data.DataLoader(test_ds, batch_size=batch_size)
        else:
            # Number of features in the dataset.
            if dataset.name not in ['MNIST','CIFAR10']:
                num_feat = dataset.train[0][0].size(-1)
            else:
                if dataset.name == 'MNIST':
                    num_feat = dataset.train[0][0].size(-1)*dataset.train[0][0].size(-1)
                elif dataset.name == 'CIFAR10':
                    num_feat = 3072
            # Number of classes in the dataset.
            num_classes = len(dataset.train.classes)
            # Number of training samples in the dataset.
            num_train_samples = len(dataset.train)
            # Get values for min-max normalization and add the transformation

            if dataset.name not in ['MNIST', 'CIFAR10']:
                min_val = torch.min(dataset.train.data, 0).values.to(device)
                max_val = torch.max(dataset.train.data, 0).values.to(device)
                transform = create_min_max_normalize(min_val, max_val)
                dataset.train.transform = transform
                dataset.test.transform = transform

            # Set up data loaders
            train_loader = data.DataLoader(dataset.train, batch_size=batch_size)
            test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

        encode = Encoder(num_feat, dimensions, method)
        encode = encode.to(device)

        model = Centroid(dimensions, num_classes)
        model = model.to(device)

        # Run for the requested number of simulations
        for r in range(repeats):
            t = time.time()
            with torch.no_grad():
                for samples, labels in tqdm(train_loader, desc="Training"):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    model.add(samples_hv, labels)
            accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

            with torch.no_grad():
                model.normalize()

                for samples, labels in tqdm(test_loader, desc="Testing"):
                    samples = samples.to(device)

                    samples_hv = encode(samples)
                    outputs = model(samples_hv, dot=True)
                    accuracy.update(outputs.cpu(), labels)


            benchmark.report(dataset, accuracy.compute().item())
            with open(results_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        dataset.name,
                        accuracy.compute().item(),
                        time.time() - t,
                        dimensions,
                        method,
                    ]
                )
            # print(f"{dataset.name} accuracy: {(accuracy.compute().item() * 100):.2f}%")


    # Returns a dictionary with names of the datasets and their respective accuracy that is averaged over folds (if applicable) and repeats
    #benchmark_accuracy = benchmark.score()

    #print(benchmark_accuracy)

BATCH_SIZE = 128
# Specifies how many random initializations of the model to evaluate for each dataset in the collection.
REPEATS = 5
# DIMENSIONS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
DIMENSIONS = [10000]
METHODS = ['bundle','hashmap','ngram','sequence','random','sinusoid','density']
#METHODS = ['bundle','sequence']
print(benchmark.datasets())

for i in DIMENSIONS:
    for j in METHODS:
        exec_arena(method=j, dimensions=i,repeats=REPEATS,batch_size=BATCH_SIZE)