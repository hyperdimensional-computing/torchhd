import sys

sys.path.append("methods")
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import utils
import torch
from sklearn.metrics import accuracy_score

def train_random_forest(
    train_data,
    test_data,
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
    train_data = []
    train_labels = []

    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            # Reshape samples into 2D vectors (flatten)
            samples_flattened = samples.view(samples.size(0), -1).cpu().numpy()

            train_data.append(samples_flattened)
            train_labels.append(labels.cpu().numpy())

    # Convert lists to NumPy arrays
    train_data = np.vstack(train_data)
    train_labels = np.concatenate(train_labels)

    # Initialize and train the random forest classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_data, train_labels)

    train_time = time.time() - train_time

    test_time = time.time()

    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Inference"):
            samples = samples.to(device)
            labels = labels.to(device)

            # Reshape samples into 2D vectors (flatten)
            samples_flattened = samples.view(samples.size(0), -1).cpu().numpy()

            test_predictions.append(samples_flattened)
            test_true_labels.append(labels.cpu().numpy())

    # Convert lists to NumPy arrays
    test_predictions = np.vstack(test_predictions)
    test_true_labels = np.concatenate(test_true_labels)

    test_predictions = model.predict(test_predictions)
    accuracy = accuracy_score(test_true_labels, test_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(accuracy)
    test_time = time.time() - test_time
    utils.write_results(
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
        partial_data,
        0,
        lazy_regeneration,
        model_neural,
        weight_decay,
        learning_rate,
        dropout_rate,
    )
