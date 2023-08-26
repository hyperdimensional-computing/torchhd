import csv
import time
import torchmetrics
from tqdm import tqdm
import torch


def write_results(
    results_file,
    name,
    accuracy,
    train_time,
    test_time,
    dimensions,
    method,
    encoding,
    iterations,
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
    amount_data,
    failure,
    lazy_regeneration,
    model_neural,
):
    with open(results_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                name,
                accuracy,
                train_time,
                test_time,
                dimensions,
                method,
                encoding,
                iterations,
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
                lazy_regeneration,
                model_neural,
                amount_data,
                failure,
            ]
        )


def test_eval(
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
):
    for f in robustness:
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(
            device
        )
        test_time = time.time()
        with torch.no_grad():
            for samples, labels in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)
                samples_hv = encode(samples)

                if method == "comp":
                    samples_hv = model.compress_hv(samples_hv, chunks, device)
                    num_dim = int((f / 100) * (dimensions / chunks))
                    f_mask = torch.randperm(int(dimensions / chunks) - 0)[:num_dim]
                    samples_hv[f_mask] = samples_hv[f_mask] * -torch.ones(num_dim).to(
                        device
                    )
                else:
                    num_dim = int((f / 100) * dimensions)
                    f_mask = torch.randperm(dimensions - 0)[:num_dim]
                    samples_hv[0][f_mask] = samples_hv[0][f_mask] * -torch.ones(
                        num_dim
                    ).to(device)

                if method == "comp":
                    outputs = (
                        model.forward_comp(samples_hv, device).unsqueeze(0).to(device)
                    )
                elif method == "quant_iterative":
                    outputs = model.quantized_similarity(samples_hv, model_quantize)
                elif method == "sparse_iterative":
                    outputs = model.sparse_similarity(samples_hv)
                elif method == "multicentroid":
                    outputs = model.multi_similarity(samples_hv, device)
                    pred = torch.argmax(outputs, dim=0)
                    row = 0
                    for i in model.multi_weight:
                        if i.shape[0] >= pred:
                            break
                        else:
                            row += 1
                            pred -= i.shape[0]
                    outputs = torch.tensor([row])
                elif method == "rvfl":
                    outputs = model(samples, dimensions=dimensions, f=f, device=device)
                else:
                    outputs = model(samples_hv, dot=False)
                accuracy.update(outputs.to(device), labels.to(device))

        test_time = time.time() - test_time
        write_results(
            results_file,
            name,
            accuracy.compute().item(),
            train_time,
            test_time,
            dimensions,
            method,
            encoding,
            iterations,
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
            f,
            lazy_regeneration,
            model_neural,
        )
        accuracy.reset()
