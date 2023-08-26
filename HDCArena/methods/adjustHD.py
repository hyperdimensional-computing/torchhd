import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils
import random
import torch.utils.data as data
import torchmetrics

import torch
from torch.utils.data import random_split
import random


def make_class_balanced_splits(dataset, num_classes, num_samples_per_split, num_splits):
    num_samples_per_class = num_samples_per_split // num_classes

    # Create a list to store the indices of each class
    class_indices = [[] for _ in range(num_classes)]

    # Iterate through the dataset and collect the indices of each class
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_indices[label].append(i)

    # Shuffle the indices of each class
    for class_idx in range(num_classes):
        random.shuffle(class_indices[class_idx])

    # Create empty lists to store the splits
    splits = [[] for _ in range(num_splits)]

    # For each class, distribute the indices equally across the splits
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]

        # Divide the indices into splits
        for split_idx in range(num_splits):
            start = split_idx * num_samples_per_class
            end = (split_idx + 1) * num_samples_per_class
            splits[split_idx].extend(indices[start:end])

    # Convert the indices to DatasetSplits
    dataset_splits = [torch.utils.data.Subset(dataset, split) for split in splits]

    return dataset_splits


def validate(test_loader, device, encode, model, num_classes):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)
    for samples, labels in tqdm(test_loader, desc="Testing"):
        samples = samples.to(device)
        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=False)
        accuracy.update(outputs.to(device), labels.to(device))
    return accuracy.compute().item()


def test(test_loader, device, encode, model, num_classes):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)
    for samples, labels in tqdm(test_loader, desc="Testing"):
        samples = samples.to(device)
        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=False)
        accuracy.update(outputs.to(device), labels.to(device))
    return accuracy.compute().item()


def train_adjustHD(
    train_ds,
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
    results_file,
):
    validation_ds, _ = data.random_split(
        train_ds, [int(len(train_ds) * 0.1), len(train_ds) - int(len(train_ds) * 0.1)]
    )
    if len(validation_ds) > 0:
        validation_loader = data.DataLoader(validation_ds, shuffle=True)
    num_samples_per_split = num_classes * 3
    num_splits = 4
    while num_splits * num_samples_per_split > len(train_ds):
        num_splits -= 1
    splits = make_class_balanced_splits(
        train_ds, num_classes, num_samples_per_split, num_splits
    )

    accuracies = []
    accuracies_test = []
    for i in range(num_splits):
        for samples, labels in tqdm(
            data.DataLoader(splits[i], shuffle=True), desc="Training"
        ):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adjust(samples_hv, labels, lr=lr)
        if len(validation_ds) > 0:
<<<<<<< HEAD
<<<<<<< HEAD
            accuracies.append(
                validate(validation_loader, device, encode, model, num_classes)
            )
=======
            accuracies.append(validate(validation_loader, device, encode, model, num_classes))
            accuracies_test.append(validate(test_loader, device, encode, model, num_classes))
>>>>>>> 66fb0df (New arena)
=======
            accuracies.append(
                validate(validation_loader, device, encode, model, num_classes)
            )
>>>>>>> ee169fd ([github-action] formatting fixes)
        model.reset_parameters()

    print("acc",accuracies)
    print(accuracies_test)

    if len(validation_ds) > 0:
        for samples, labels in tqdm(
            data.DataLoader(
                splits[torch.argmax(torch.tensor(accuracies))], shuffle=True
            ),
            desc="Training",
        ):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adjust(samples_hv, labels, lr=lr)

    train_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add_adjust(samples_hv, labels, lr=lr)
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
    )
