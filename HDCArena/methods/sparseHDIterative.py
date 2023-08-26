import sys
sys.path.append('methods')
from tqdm import tqdm
import time
import utils
import torchmetrics
import math


def train_sparseHD(train_loader, test_loader, num_classes, encode, model, device, name, method, encoding,
                  iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                   model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file):
    train_len = len(train_loader)
    validation_set = train_len - math.ceil(len(train_loader) * 0.05)
    accuracy_validation = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)

    train_time = time.time()

    for iter in range(iterations):
        accuracy_validation_sparse = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        ).to(device)
        for idx, (samples, labels) in enumerate(
                tqdm(train_loader, desc="Training")
        ):
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)

            if idx < validation_set:
                model.add_sparse(samples_hv, labels, lr=lr, iter=iter)
            else:
                if idx == validation_set:
                    model.sparsify_model(model_sparse, int(s * dimensions), iter)

                outputs = model.sparse_similarity(samples_hv)
                if iter == 0:
                    outputs_full = model.forward(samples_hv)
                    accuracy_validation.update(outputs_full.to(device), labels)
                accuracy_validation_sparse.update(outputs.to(device), labels)
        if iter == 0:
            accuracy = accuracy_validation.compute().item()

        if abs(accuracy_validation_sparse.compute().item() - accuracy) < epsilon:
            iterations = iter
            break

    train_time = time.time() - train_time

    model.normalize()

    utils.test_eval(test_loader, num_classes, encode, model, device, name, method, encoding, iterations, dimensions,
                    lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon, model_sparse, s, alpha, beta, theta, r,
                    partial_data, robustness, results_file, train_time, lazy_regeneration, model_neural)

