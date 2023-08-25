import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid, CentroidMiss
from torchhd.datasets.isolet import ISOLET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones


class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.embed = embeddings.Sinusoid(size, DIMENSIONS)

    def forward(self, x):
        sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)


train_ds = ISOLET("/Users/verges/Documents/PhD/TorchHd/torchhd/data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = ISOLET("/Users/verges/Documents/PhD/TorchHd/torchhd/data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

encode = Encoder(DIMENSIONS, train_ds[0][0].size(-1))
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = CentroidMiss(DIMENSIONS, num_classes)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add_adjust(samples_hv, labels)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
accuracy2 = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        outputs_misspredict = model.forward_misspredicted(samples_hv, dot=True)
        #print(outputs_misspredict)
        #print(labels, torch.argmax(outputs), torch.max(outputs), torch.argmax(outputs_misspredict), torch.max(outputs_misspredict))
        accuracy.update(outputs.cpu(), labels)
        if torch.max(outputs) > torch.max(outputs_misspredict):
            accuracy2.update(outputs.cpu(), labels)
        else:
            accuracy2.update(outputs_misspredict.cpu(), labels)


print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print(f"Testing accuracy of {(accuracy2.compute().item() * 100):.3f}%")
