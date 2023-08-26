import sys

sys.path.append("methods")
from tqdm import tqdm
import time
import utils


def train_distHD(
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
            model.add_dist(samples_hv, labels, lr=lr)
            model.eval_dist(
                samples_hv, labels, device, alpha=alpha, beta=beta, theta=theta
            )
        model.regenerate_dist(int(r * dimensions), encode, device)

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
