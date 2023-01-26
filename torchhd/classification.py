import torch
from tqdm import tqdm

import torchhd
from torchhd.map import MAP

__all__ = [
    "EncodingDensityClipped",
    "classifier_ridge_regression",
]


class EncodingDensityClipped:
    """Class that performs the transformation of input data into hypervectors according to intRVFL model. See details in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        dimensions (int): Dimensionality of vectors used when transforming input data.
        num_feat (int): Number of features in the dataset.
        kappa (int): Parameter of the clipping function used as the part of transforming input data.
        key (torchhd.map.MAP): A set of random vectors used as unique IDs for features of the dataset.
        density_encoding (torchhd.embeddings.Thermometer): Thermometer encoding used for transforming input data.
    """

    def __init__(
        self,
        dimensions: int,
        num_feat: int,
        kappa: int,
    ):
        super(EncodingDensityClipped, self).__init__()

        self.key = torchhd.random_hv(num_feat, dimensions, model=MAP)
        self.density_encoding = torchhd.embeddings.Thermometer(
            dimensions + 1, dimensions, low=0, high=1
        )
        self.kappa = kappa

    # Specify the steps needed to perform the encoding
    def encode(self, x):
        # Perform binding of key and value vectors
        sample_hv = MAP.bind(self.key, self.density_encoding(x))
        # Perform the superposition operation on the bound key-value pairs
        sample_hv = MAP.multibundle(sample_hv)
        # Perform clipping function on the result of the superposition operation and return
        return sample_hv.clipping(self.kappa)


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
