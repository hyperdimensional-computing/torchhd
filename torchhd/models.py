import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.utils.data as data
from tqdm import tqdm


import torchhd.functional as functional
import torchhd.datasets as datasets
import torchhd.embeddings as embeddings


__all__ = [
    "Centroid",
    "IntRVFLRidge",
    "classifier_ridge_regression",
]

class Centroid(nn.Module):
    r"""Implements the centroid classification model using class prototypes.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(*, d)` where :math:`*` means any number of
          dimensions including none and ``d = in_features``.
        - Output: :math:`(*, n)` where all but the last dimension
          are the same shape as the input and ``n = out_features``.

    Attributes:
        weight: the trainable weights, or class prototypes, of the module of shape
            :math:`(n, d)`. The values are initialized as all zeros.

    Examples::

        >>> m = Centroid(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> output.size()
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Centroid, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cosine_similarity(input, self.weight)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * input)

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features is not None
        )

class IntRVFLRidge(nn.Module):
    """Class implementing integer random vector functional link network (intRVFL) model as described in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        dataset (torchhd.datasets.CollectionDataset): Specifies a dataset to be evaluted by the model.
        num_feat (int): Number of features in the dataset.
        device (torch.device, optional): Specifies device to be used for Torch.
    """

    # These values of hyperparameters were found via the grid search for intRVFL model as described in the article.
    INT_RVFL_HYPER = {
        "abalone": (1450, 32, 15),
        "acute-inflammation": (50, 0.0009765625, 1),
        "acute-nephritis": (50, 0.0009765625, 1),
        "adult": (1150, 0.0625, 3),
        "annealing": (1150, 0.015625, 7),
        "arrhythmia": (1400, 0.0009765625, 7),
        "audiology-std": (950, 16, 3),
        "balance-scale": (50, 32, 7),
        "balloons": (50, 0.0009765625, 1),
        "bank": (200, 0.001953125, 7),
        "blood": (50, 16, 7),
        "breast-cancer": (50, 32, 1),
        "breast-cancer-wisc": (650, 16, 3),
        "breast-cancer-wisc-diag": (1500, 2, 3),
        "breast-cancer-wisc-prog": (1450, 0.01562500, 3),
        "breast-tissue": (1300, 0.1250000, 1),
        "car": (250, 32, 3),
        "cardiotocography-10clases": (1350, 0.0009765625, 3),
        "cardiotocography-3clases": (900, 0.007812500, 15),
        "chess-krvk": (800, 4, 1),
        "chess-krvkp": (1350, 0.01562500, 3),
        "congressional-voting": (100, 32, 15),
        "conn-bench-sonar-mines-rocks": (1100, 0.01562500, 3),
        "conn-bench-vowel-deterding": (1350, 8, 3),
        "connect-4": (1100, 0.5, 3),
        "contrac": (50, 8, 7),
        "credit-approval": (200, 32, 7),
        "cylinder-bands": (1100, 0.0009765625, 7),
        "dermatology": (900, 8, 3),
        "echocardiogram": (250, 32, 15),
        "ecoli": (350, 32, 3),
        "energy-y1": (650, 0.1250000, 3),
        "energy-y2": (1000, 0.0625, 7),
        "fertility": (150, 32, 7),
        "flags": (900, 32, 15),
        "glass": (1400, 0.03125000, 3),
        "haberman-survival": (100, 32, 3),
        "hayes-roth": (50, 16, 1),
        "heart-cleveland": (50, 32, 15),
        "heart-hungarian": (50, 16, 15),
        "heart-switzerland": (50, 8, 15),
        "heart-va": (1350, 0.1250000, 15),
        "hepatitis": (1300, 0.03125000, 1),
        "hill-valley": (150, 0.01562500, 1),
        "horse-colic": (850, 32, 1),
        "ilpd-indian-liver": (1200, 0.25, 7),
        "image-segmentation": (650, 8, 1),
        "ionosphere": (1150, 0.001953125, 1),
        "iris": (50, 4, 3),
        "led-display": (50, 0.0009765625, 7),
        "lenses": (50, 0.03125000, 1),
        "letter": (1500, 32, 1),
        "libras": (1250, 0.1250000, 3),
        "low-res-spect": (1400, 8, 7),
        "lung-cancer": (450, 0.0009765625, 1),
        "lymphography": (1150, 32, 1),
        "magic": (800, 16, 3),
        "mammographic": (150, 16, 7),
        "miniboone": (650, 0.0625, 15),
        "molec-biol-promoter": (1250, 32, 1),
        "molec-biol-splice": (1000, 8, 15),
        "monks-1": (50, 4, 3),
        "monks-2": (400, 32, 1),
        "monks-3": (50, 4, 15),
        "mushroom": (150, 0.25, 3),
        "musk-1": (1300, 0.001953125, 7),
        "musk-2": (1150, 0.007812500, 7),
        "nursery": (1000, 32, 3),
        "oocytes_merluccius_nucleus_4d": (1500, 1, 7),
        "oocytes_merluccius_states_2f": (1500, 0.0625, 7),
        "oocytes_trisopterus_nucleus_2f": (1450, 0.003906250, 3),
        "oocytes_trisopterus_states_5b": (1450, 2, 7),
        "optical": (1100, 32, 7),
        "ozone": (50, 0.003906250, 1),
        "page-blocks": (800, 0.001953125, 1),
        "parkinsons": (1200, 0.5, 1),
        "pendigits": (1500, 0.1250000, 1),
        "pima": (50, 32, 1),
        "pittsburg-bridges-MATERIAL": (100, 8, 1),
        "pittsburg-bridges-REL-L": (1200, 0.5, 1),
        "pittsburg-bridges-SPAN": (450, 4, 7),
        "pittsburg-bridges-T-OR-D": (1000, 16, 1),
        "pittsburg-bridges-TYPE": (50, 32, 7),
        "planning": (50, 32, 1),
        "plant-margin": (1350, 2, 7),
        "plant-shape": (1450, 0.25, 3),
        "plant-texture": (1500, 4, 7),
        "post-operative": (50, 32, 15),
        "primary-tumor": (950, 32, 3),
        "ringnorm": (1500, 0.125, 3),
        "seeds": (550, 32, 1),
        "semeion": (1400, 32, 15),
        "soybean": (850, 1, 3),
        "spambase": (1350, 0.0078125, 15),
        "spect": (50, 32, 1),
        "spectf": (1100, 0.25, 15),
        "statlog-australian-credit": (200, 32, 15),
        "statlog-german-credit": (500, 32, 15),
        "statlog-heart": (50, 32, 7),
        "statlog-image": (950, 0.125, 1),
        "statlog-landsat": (1500, 16, 3),
        "statlog-shuttle": (100, 0.125, 7),
        "statlog-vehicle": (1450, 0.125, 7),
        "steel-plates": (1500, 0.0078125, 3),
        "synthetic-control": (1350, 16, 3),
        "teaching": (400, 32, 3),
        "thyroid": (300, 0.001953125, 7),
        "tic-tac-toe": (750, 8, 1),
        "titanic": (50, 0.0009765625, 1),
        "trains": (100, 16, 1),
        "twonorm": (1100, 0.0078125, 15),
        "vertebral-column-2clases": (250, 32, 3),
        "vertebral-column-3clases": (200, 32, 15),
        "wall-following": (1200, 0.00390625, 3),
        "waveform": (1400, 8, 7),
        "waveform-noise": (1300, 0.0009765625, 15),
        "wine": (850, 32, 1),
        "wine-quality-red": (1100, 32, 1),
        "wine-quality-white": (950, 8, 3),
        "yeast": (1350, 4, 1),
        "zoo": (400, 8, 7),
    }

    def __init__(
        self,
        dataset: datasets.CollectionDataset,
        num_feat: int,
        device: torch.device = None,
    ):
        super(IntRVFLRidge, self).__init__()
        self.device = device
        self.num_feat = num_feat
        # Fetch the hyperparameters for the corresponding dataset
        hyper_param = self.INT_RVFL_HYPER[dataset.name]
        # Dimensionality of vectors used when transforming input data
        self.dimensions = hyper_param[0]
        # Regularization parameter used for ridge regression classifier
        self.lamb = hyper_param[1]
        # Parameter of the clipping function used as the part of transforming input data
        self.kappa = hyper_param[2]
        # Number of classes in the dataset
        self.num_classes = len(dataset.classes)
        # Initialize the classifier
        self.classify = nn.Linear(self.dimensions, self.num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)
        # Set up the encoding for the model as specified in "Density"
        self.hypervector_encoding = embeddings.Density(
            self.num_feat, self.dimensions
        )

    # Specify encoding function for data samples
    def encode(self, x):
        return self.hypervector_encoding(x).clipping(self.kappa)

    # Specify how to make an inference step and issue a prediction
    def forward(self, x):
        # Make encodings for all data samples in the batch
        encodings = self.encode(x)
        # Get similarity values for each class assuming implicitly that there is only one prototype per class. This does not have to be the case in general.
        logit = self.classify(encodings)
        # Form predictions
        predictions = torch.argmax(logit, dim=-1)
        return predictions

    # Train the classfier
    def fit(
        self,
        train_ld: data.DataLoader,
    ):
        # Gets classifier (readout matrix) via the ridge regression
        Wout = classifier_ridge_regression(
            train_ld,
            self.dimensions,
            self.num_classes,
            self.lamb,
            self.encode,
            self.hypervector_encoding.key.weight.dtype,
            self.device,
        )
        # Assign the obtained classifier to the output
        with torch.no_grad():
            self.classify.weight.copy_(Wout)
              
   
# 
def classifier_ridge_regression(        
    train_ld: data.DataLoader,
    dimensions: int,
    num_classes: int,
    lamb: float,
    encoding_function,
    data_type: torch.dtype,
    device: torch.device,
):
    """Function that forms the classifier (readout matrix) with the ridge regression.

    It is a common way to form classifiers wihtin randomized neural networks see, e.g., `Randomness in Neural Networks: An Overview  <https://doi.org/10.1002/widm.1200>`_.
    """
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
