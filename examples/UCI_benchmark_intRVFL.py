import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
import torchhd
from torchhd.datasets import UCIClassificationBenchmark
from torchhd.models import IntRVFLRidge


# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input - min) / (max - min))

    return normalize


# Specify device to be used for Torch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
# Specifies batch size to be used for the model.
batch_size = 10
# Specifies how many random initializations of the model to evaluate for each dataset in the collection.
repeats = 5


# Get an instance of the UCI benchmark
benchmark = UCIClassificationBenchmark("../data", download=True)
# Perform evaluation
for dataset in benchmark.datasets():
    print(dataset.name)

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
    train_loader = data.DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

    # Run for the requested number of simulations
    for r in range(repeats):
        # Creates a model to be evaluated. The model should specify both transformation of input data as weel as the algortihm for forming the classifier.
        model = IntRVFLRidge(
            getattr(torchhd.datasets, dataset.name), num_feat, device
        ).to(device)

        # Obtain the classifier for the model
        model.fit(train_loader)
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():
            for samples, targets in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)
                # Make prediction
                predictions = model(samples)
                accuracy.update(predictions.cpu(), targets)

        benchmark.report(dataset, accuracy.compute().item())

# Returns a dictionary with names of the datasets and their respective accuracy that is averaged over folds (if applicable) and repeats
benchmark_accuracy = benchmark.score()
print(benchmark_accuracy)
