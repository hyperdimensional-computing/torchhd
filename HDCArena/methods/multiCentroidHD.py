import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import torchmetrics
import time
import utils


def train_multicentroidHD(
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
    train_time = time.time()
    with torch.no_grad():
        accuracy_train = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        ).to(device)
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            pred = model.add_multi(samples_hv, labels, device)
            accuracy_train.update(pred.to(device), labels)

        model.reduce_subclasses(
            train_loader,
            device,
            encode,
            model,
            num_classes,
            accuracy_train.compute().item(),
            reduce_subclasses=reduce_subclasses,
            threshold=threshold,
        )

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
