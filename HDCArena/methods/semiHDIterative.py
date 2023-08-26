import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils
import torchmetrics
from collections import deque
import torch.utils.data as data
from torch.utils.data import Dataset, Subset


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        # Return data and label at the given index
        return self.data[index], self.labels[index]

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)


def train_semiHD(
    train_ds,
    test_ds,
    train_loader,
    test_loader,
    num_classes,
    encode,
    model,
    device,
    name,
    method,
    encoding,
    iterations,
    dimensions,
    lr,
    chunks,
    threshold,
    reduce_subclasses,
    model_quantize,
    epsilon,
    model_sparse,
    s,
    alpha,
    beta,
    theta,
    r,
    partial_data,
    robustness,
    lazy_regeneration,
    model_neural,
    weight_decay,
    learning_rate,
    dropout_rate,
    results_file,
):
    if int(0.1 * len(train_ds)) > 0:
        train_ds, unlabeled_ds = torch.utils.data.random_split(
            train_ds,
            [int(0.1 * len(train_ds)), len(train_ds) - int(0.1 * len(train_ds))],
        )

        train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True)
        unlabeled_loader = data.DataLoader(unlabeled_ds, batch_size=1, shuffle=True)
        test_loader = data.DataLoader(test_ds, batch_size=1)

    train_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    if int(0.1 * len(train_ds)) > 0:
        with torch.no_grad():
            q = deque(maxlen=3)
            for iter in range(iterations):
                accuracy_train = torchmetrics.Accuracy(
                    "multiclass", num_classes=num_classes
                ).to(device)

                diff = []

                for samples, labels in tqdm(unlabeled_loader, desc="Training"):
                    samples = samples.to(device)

                    samples_hv = encode(samples)
                    outputs = torch.topk(model.forward(samples_hv, dot=False), k=2)[1][
                        0
                    ]
                    diff.append(outputs[0] - outputs[1])

                top_amount = int(len(diff) * s)
                top_data = torch.topk(torch.tensor(diff), k=top_amount)
                new_data = []
                new_labels = []
                update = False
                for index, (samples, labels) in enumerate(
                    tqdm(unlabeled_loader, desc="Training")
                ):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    if index in top_data.indices:
                        model.add(samples_hv, labels)
                        outputs = model.forward(samples_hv, dot=False)
                        accuracy_train.update(outputs.to(device), labels.to(device))
                        update = True
                    else:
                        new_data.append(samples)
                        new_labels.append(labels)
                if not update:
                    break
                unlabeled_ds = CustomDataset(
                    torch.stack(new_data), torch.tensor(new_labels)
                )
                unlabeled_loader = data.DataLoader(
                    unlabeled_ds, batch_size=1, shuffle=True
                )

                lr = (1 - accuracy_train.compute().item()) * 10
                if len(q) == 3:
                    if all(abs(q[i] - q[i - 1]) < 0.001 for i in range(1, len(q))):
                        iterations = iter
                    q.append(accuracy_train.compute().item())
                else:
                    q.append(accuracy_train.compute().item())

    train_time = time.time() - train_time

    model.normalize()

    utils.test_eval(
        test_loader,
        num_classes,
        encode,
        model,
        device,
        name,
        method,
        encoding,
        iterations,
        dimensions,
        lr,
        chunks,
        threshold,
        reduce_subclasses,
        model_quantize,
        epsilon,
        model_sparse,
        s,
        alpha,
        beta,
        theta,
        r,
        partial_data,
        robustness,
        results_file,
        train_time,
        lazy_regeneration,
        model_neural,
        weight_decay,
        learning_rate,
        dropout_rate,
    )
