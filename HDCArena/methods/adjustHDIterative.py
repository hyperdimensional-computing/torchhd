import sys

sys.path.append("methods")
import torch
from tqdm import tqdm
import time
import utils
import torchmetrics
from collections import deque
from torchhd import functional

t = True

def orthogonality(tensor_of_tensors, classes, name):
    dot_products = torch.zeros(classes, classes)

    for i in range(classes):
        for j in range(i + 1, classes):
            dot_product = functional.cos(tensor_of_tensors[i], tensor_of_tensors[j])
            dot_products[i, j] = dot_product
            dot_products[j, i] = dot_product

    # Print the result
    #print(torch.sum(dot_products)/((classes*classes-classes)/2))
    orthogonalit = torch.sum(dot_products/((classes*classes-classes)))
    #print(orthogonalit, classes, name)
    return orthogonalit

def train_adjustHD(
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
    q = deque(maxlen=3)

    with torch.no_grad():
        for iter in range(iterations):
            accuracy_train = torchmetrics.Accuracy(
                "multiclass", num_classes=num_classes
            ).to(device)

            for samples, labels in tqdm(train_loader, desc="Training", disable=t):
                samples = samples.to(device)
                labels = labels.to(device)

                samples_hv = encode(samples)
                model.add_adjust(samples_hv, labels, lr=lr)
                outputs = model.forward(samples_hv, dot=False)
                accuracy_train.update(outputs.to(device), labels.to(device))
            model.adjust_reset()
            lr = (1 - accuracy_train.compute().item()) * 10
            if len(q) == 3:
                if all(abs(q[i] - q[i - 1]) < 0.001 for i in range(1, len(q))):
                    iterations = iter
                    break
                q.append(accuracy_train.compute().item())
            else:
                q.append(accuracy_train.compute().item())
            accuracy_train.reset()
    train_time = time.time() - train_time

    # model.normalize()

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
