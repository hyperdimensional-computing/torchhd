import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
from torchhd.datasets import UCIClassificationBenchmark

torch.manual_seed(0)
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
import numpy as np
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid, MemoryModel
from torchhd.datasets import EMGHandGestures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
method = "MemoryModel"
BATCH_SIZE = 1


class Encoder(nn.Module):
    def __init__(self, size, type):
        super(Encoder, self).__init__()
        self.type = type
        self.keys = None
        if self.type == "hashmap":
            self.keys = embeddings.Random(size, DIMENSIONS)
            self.values = embeddings.Level(size, DIMENSIONS)
        elif self.type == "projection":
            self.proj = embeddings.Projection(size, DIMENSIONS)
        elif self.type == "sinusoid":
            self.proj = embeddings.Sinusoid(size, DIMENSIONS)
        elif self.type == "density":
            self.proj = embeddings.Density(size, DIMENSIONS)
        # self.proj = embeddings.Projection(size, DIMENSIONS)

    def forward(self, x):
        if self.type == "hashmap":
            sample_hv = torchhd.hash_table(self.keys.weight, self.values(x))
        else:
            sample_hv = self.proj(x).sign()
        return torchhd.hard_quantize(sample_hv)


def create_min_max_normalize(min, max):
    def normalize(input):
        return torch.nan_to_num((input - min) / (max - min))

    return normalize


def normalize(w, eps=1e-12) -> None:
    """Transforms all the class prototype vectors into unit vectors.

    After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
    Training further after calling this method is not advised.
    """
    norms = w.norm(dim=1, keepdim=True)
    norms.clamp_(min=eps)
    w.div_(norms)


def experiment():
    train = torchhd.datasets.OocytesMerlucciusNucleus4d(
        "../../data", download=True, train=True, fold=0
    )
    test = torchhd.datasets.OocytesMerlucciusNucleus4d(
        "../../data", download=True, train=False, fold=0
    )

    train_size = int(0.8 * len(train))
    test_size = len(train) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train, [train_size, test_size]
    )

    num_classes = len(train.classes)

    min_val = torch.min(train.data, 0).values.to(device)
    max_val = torch.max(train.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    train.transform = transform
    test.transform = transform
    train_data = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    validation_loader = data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = data.DataLoader(test, batch_size=BATCH_SIZE)
    types = ["density"]

    for t in types:
        model = Centroid(DIMENSIONS, num_classes)
        encode = Encoder(train[0][0].size(-1), t)
        encode = encode.to(device)
        count = 0
        epochs = 10

        with torch.no_grad():
            for i in range(epochs):
                validate_accuracy = 0
                train_accuracy = 0
                test_accuracy = 0
                model.mse = 0
                for samples, labels in train_loader:
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    pred = model.add_online(samples_hv, labels)
                    if pred.item() == labels[0].item():
                        train_accuracy += 1
                model.normalize()

                matri = torchhd.cos_similarity(model.weight, model.weight)
                print(torch.sum(torch.triu(abs(matri), diagonal=1)))
                print(torch.det(matri))
                # hinge loss
                for samples, labels in validation_loader:
                    samples = samples.to(device)
                    labels = labels.to(device)
                    samples_hv = encode(samples)
                    outputs = model(samples_hv, dot=True)
                    if torch.argmax(outputs).item() == labels[0].item():
                        validate_accuracy += 1
                print("VALIDATE ACC ", validate_accuracy / len(validation_dataset))

                print("TRAIN ACC ", train_accuracy / len(train_data))

                # print(model.error_similarity_sum / model.error_count)
                # print(model.similarity_sum / model.count)
                # print('VALIDATE ACC ', validate_accuracy/total_val)
                model.error_similarity_sum = 0
                model.error_count = 0
                model.similarity_sum = 0
                model.count = 0

                for samples, labels in test_loader:
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    outputs = model(samples_hv, dot=True)

                    if outputs.argmax(1).item() == labels[0].item():
                        test_accuracy += 1

                print(f"TEST ACC {(test_accuracy/len(test)):.3f}%")
                print(f"MSE {(model.mse/len(train_dataset)):.3f}%")
                print()


experiment()
