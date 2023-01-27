import torch
from tqdm import tqdm

import torchhd
from torchhd.map import MAP

__all__ = [
    "classifier_ridge_regression",
]

# Function that forms the classifier (readout matrix) with the ridge regression
def classifier_ridge_regression(
    train_ld: torch.utils.data.dataloader.DataLoader,
    dimensions: int,
    num_classes: int,
    lamb: float,
    encoding_function,
    data_type: torch.dtype,
    device: torch.device,
):

    # Get number of training samples
    num_train = len(train_ld.dataset)
    # Collects high-dimensional represetations of data in the train data
    total_samples_hv = torch.zeros(
        num_train,
        dimensions,
        dtype=data_type,
        device=device,
    )
    # Collects one-hot encodings of class labels
    labels_one_hot = torch.zeros(
        num_train,
        num_classes,
        dtype=data_type,
        device=device,
    )

    with torch.no_grad():
        count = 0
        for samples, labels in tqdm(train_ld, desc="Training"):

            samples = samples.to(device)
            labels = labels.to(device)
            # Make one-hot encoding
            labels_one_hot[torch.arange(count, count + samples.size(0)), labels] = 1

            # Make transformation into high-dimensional space
            samples_hv = encoding_function(samples)
            total_samples_hv[count : count + samples.size(0), :] = samples_hv

            count += samples.size(0)

        # Compute the readout matrix using the ridge regression
        Wout = (
            torch.t(labels_one_hot)
            @ total_samples_hv
            @ torch.linalg.pinv(
                torch.t(total_samples_hv) @ total_samples_hv
                + lamb * torch.diag(torch.var(total_samples_hv, 0))
            )
        )

    return Wout
