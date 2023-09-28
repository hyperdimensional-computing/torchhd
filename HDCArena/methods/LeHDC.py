import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils
import torchmetrics


def train_LeHDC(
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
    train_time = time.time()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=weight_decay, lr=learning_rate
    )

    loss_accum = 0
    loss_log = 0

    for i in range(iterations):
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples, device)
            samples_hv = model.binarize(samples_hv)
            pred = model(samples_hv)
            loss = criterion(pred, labels)
            if torch.rand(1) < 0.2:
                loss_accum += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 1 == 0 or i == iterations - 1:
            if loss_accum > loss_log:
                optimizer.param_groups[0]["lr"] *= 0.5
            loss_log = loss_accum
            loss_accum = 0

    train_time = time.time() - train_time

    model.norm_weights()

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
