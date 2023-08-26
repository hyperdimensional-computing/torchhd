import sys

sys.path.append("methods")
from tqdm import tqdm
import time
import utils


def train_neuralHD(
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

    for iter in range(iterations):
        for idx, (samples, labels) in enumerate(tqdm(train_loader, desc="Training")):
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            model.add_neural(samples_hv, labels, lr=lr)
        if iter % lazy_regeneration == 0:
            model.neural_regenerate(int(r * dimensions), encode, device)
            model.normalize()
            if model_neural == "reset" and iter != iterations - 1:
                model.reset_parameters()

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
