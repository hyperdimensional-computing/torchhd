import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import copy

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets.isolet import ISOLET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
# number of hypervector dimensions
NUM_LEVELS = 10
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones


"""class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        #sample_hv = torchhd.functional.ngrams(sample_hv, 3)
        return torchhd.hard_quantize(sample_hv)
    """


class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.embed = embeddings.Density(size, DIMENSIONS)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)


train_ds = ISOLET("../data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = ISOLET("../data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

encode = Encoder(DIMENSIONS, train_ds[0][0].size(-1))
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)
accuracy_train = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
recall_train = torchmetrics.Recall(
    task="multiclass", average="macro", num_classes=num_classes
)
precision_train = torchmetrics.Precision(
    task="multiclass", average="macro", num_classes=num_classes
)
f1_train = torchmetrics.F1Score(
    task="multiclass", average="macro", num_classes=num_classes
)
confusion = torchmetrics.ConfusionMatrix(num_classes=num_classes)

accuracy_test = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
recall_test = torchmetrics.Recall(
    task="multiclass", average="macro", num_classes=num_classes
)
precision_test = torchmetrics.Precision(
    task="multiclass", average="macro", num_classes=num_classes
)
f1_test = torchmetrics.F1Score(
    task="multiclass", average="macro", num_classes=num_classes
)
confusion_test = torchmetrics.ConfusionMatrix(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for i in range(1):
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            pred = model.add_online2(samples_hv, labels)
            accuracy_train.update(pred.cpu(), labels)
            recall_train.update(pred.cpu(), labels)
            precision_train.update(pred.cpu(), labels)
            f1_train.update(pred.cpu(), labels)
            confusion.update(pred.cpu(), labels)

        # print("Loss train", sum(model.losses)/len(model.losses))
        print("accuracy_train:", (accuracy_train.compute().item() * 100))
        print("recall_train:", (recall_train.compute().item() * 100))
        print("precision_train:", (precision_train.compute().item() * 100))
        print("f1_train:", (f1_train.compute().item() * 100))
        # print("confusion:", (confusion.compute()))

    loss_test = []
    with torch.no_grad():
        model.normalize()
        # model2 = copy.deepcopy(model)
        # model2.normalize()
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            loss = criterion(outputs, labels)

            loss_test.append(loss.item())

            accuracy_test.update(outputs.cpu(), labels)
            recall_test.update(outputs.cpu(), labels)
            precision_test.update(outputs.cpu(), labels)
            f1_test.update(outputs.cpu(), labels)
            confusion_test.update(outputs.cpu(), labels)
    print("Loss test", sum(loss_test) / len(loss_test))
    print("accuracy_test:", (accuracy_test.compute().item() * 100))
    print("recall_test:", (recall_test.compute().item() * 100))
    print("precision_test:", (precision_test.compute().item() * 100))
    print("f1_test:", (f1_test.compute().item() * 100))
    # print("confusion:", (confusion.compute()))
