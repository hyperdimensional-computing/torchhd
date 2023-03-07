import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
from torchhd.datasets import UCIClassificationBenchmark
torch.manual_seed(20)
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
    def __init__(self, size):
        super(Encoder, self).__init__()
        self.proj = embeddings.Projection(size, DIMENSIONS)

    def forward(self, x):
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
    train = torchhd.datasets.Yeast("../../data", download=True, train=True, fold=1)
    test = torchhd.datasets.Yeast("../../data", download=True, train=False, fold=1)
    added = 0
    #test = torchhd.datasets.AcuteInflammation("../../data", download=True, train=False)
    # Number of features in the dataset.
    # Number of classes in the dataset.
    num_classes = len(train.classes)

    # Get values for min-max normalization and add the transformation
    min_val = torch.min(train.data, 0).values.to(device)
    max_val = torch.max(train.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    train.transform = transform
    test.transform = transform

    # Set up data loaders
    train_loader = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=BATCH_SIZE)

    model = Centroid(DIMENSIONS, num_classes)

    encode = Encoder(train[0][0].size(-1))
    encode = encode.to(device)

    count = 0
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc='Testing'):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            #print("labels", labels)
            model.add_online(samples_hv, labels)
            #if count == 10:
                #break
            count += 1
        model.normalize()

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)



    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)

            accuracy.update(outputs.cpu(), labels)
    print("Added samples ", added)
    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


experiment()