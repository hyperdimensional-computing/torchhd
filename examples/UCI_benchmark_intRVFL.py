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


# These values of hyperparameters were found via the grid search for intRVFL model as described in the article.
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

    # get hyperparameters
    dims, alpha, kappa = INT_RVFL_HYPER[dataset.train.name]

    # Number of features in the dataset.
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
    dataset.test.transform = transform

    # Set up data loaders
    train_loader = data.DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

    one_hot_targets = torch.zeros(num_train_samples, num_classes, device=device)
    one_hot_targets[torch.arange(num_train_samples), dataset.train.targets] = 1

    # Run for the requested number of simulations
    for r in range(repeats):
        # Creates a model to be evaluated. The model should specify both transformation of input data as well as the algorithm for forming the classifier.
        model = IntRVFLRidge(num_feat, dims, num_classes, kappa=kappa, device=device)

        # Obtain the classifier for the model
        model.fit_ridge_regression(dataset.train.data, one_hot_targets)
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():
            for samples, targets in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)
                # Make prediction
                logits = model(samples)
                accuracy.update(logits.cpu(), targets)

        benchmark.report(dataset, accuracy.compute().item())
        print(f"{dataset.name} accuracy: {(accuracy.compute().item() * 100):.2f}%")

# Returns a dictionary with names of the datasets and their respective accuracy that is averaged over folds (if applicable) and repeats
benchmark_accuracy = benchmark.score()
print(benchmark_accuracy)
