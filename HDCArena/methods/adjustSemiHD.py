import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils


def train_adjustSemiHD(
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
    hvs = []
    err_ = 0
    err = 0
    cor_ = 0
    cor = 0
    train_time = time.time()
    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            add, v = model.add_adjustSemi(samples_hv, labels, lr=lr)
            if add:
               hvs.append((samples_hv, labels))
               err += 1
               err_ += v
            else:
                cor += 1
                cor_ += v
        print(name, err_/err, cor_/cor)
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
        for samples_hv, labels in hvs:
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
        weight_decay,
        learning_rate,
        dropout_rate,
    )
