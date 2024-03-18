import torch
import torchhd
from torchhd.datasets.isolet import ISOLET

classifiers = [
    "Vanilla",
    "AdaptHD",
    "OnlineHD",
    "NeuralHD",
    "DistHD",
    "CompHD",
    "SparseHD",
    "QuantHD",
    "LeHDC",
    "IntRVFL",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1024  # number of hypervector dimensions
BATCH_SIZE = 12  # for GPUs with enough memory we can process multiple images at ones

train_ds = ISOLET("../data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = ISOLET("../data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

num_features = train_ds[0][0].size(-1)
num_classes = len(train_ds.classes)

std, mean = torch.std_mean(train_ds.data, dim=0, keepdim=False)


def transform(sample):
    return (sample - mean) / std


train_ds.transform = transform
test_ds.transform = transform

params = {
    "Vanilla": {},
    "AdaptHD": {
        "epochs": 10,
    },
    "OnlineHD": {
        "epochs": 10,
    },
    "NeuralHD": {
        "epochs": 10,
        "regen_freq": 5,
    },
    "DistHD": {
        "epochs": 10,
        "regen_freq": 5,
    },
    "CompHD": {},
    "SparseHD": {
        "epochs": 10,
    },
    "QuantHD": {
        "epochs": 10,
    },
    "LeHDC": {
        "epochs": 10,
    },
    "IntRVFL": {},
}

for classifier in classifiers:
    print()
    print(classifier)

    model_cls = getattr(torchhd.classifiers, classifier)
    model: torchhd.classifiers.Classifier = model_cls(
        num_features, DIMENSIONS, num_classes, device=device, **params[classifier]
    )

    model.fit(train_ld)
    accuracy = model.accuracy(test_ld)
    print(f"Testing accuracy of {(accuracy * 100):.3f}%")
