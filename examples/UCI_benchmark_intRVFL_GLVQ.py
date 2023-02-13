import copy
import warnings
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
import torchhd
from torchhd.datasets import UCIClassificationBenchmark

# Note: this example requires the prototorch library: https://github.com/si-cim/prototorch
import prototorch as pt
# Note: this example requires the prototorch-models library: https://github.com/si-cim/prototorch_models
from prototorch.models import GLVQ
# Note: this example requires the pytorch-lightning library: https://www.pytorchlightning.ai
import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import TQDMProgressBar

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)


# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input - min) / (max - min))

    return normalize


# Specify a model to be evaluated
class IntRVFLGLVQ(nn.Module):
    """Class implementing integer random vector functional link network (intRVFL) model with Generalized Learning Vector Quantization (GLVQ) classifier as described in `Generalized Learning Vector Quantization for Classification in Randomized Neural Networks and Hyperdimensional Computing <https://doi.org/10.1109/IJCNN52387.2021.9533316>`_.

    Args:
        dataset (torchhd.datasets.CollectionDataset): Specifies a dataset to be evaluted by the model.
        num_feat (int): Number of features in the dataset.
        max_epochs (int, optional): Number of epochs to train the classifier.
        device (torch.device, optional): Specifies device to be used for Torch.
    """

    # These values of hyperparameters were found via the grid search for intRVFL model with GLVQ classifier as described in the paper.
    INT_RVFL_GLVQ_HYPER = {
        "abalone": (1450, 15, 15, 1),
        "acute-inflammation": (50, 1, 5, 1),
        "acute-nephritis": (50, 1, 2, 1),
        "adult": (1150, 3, 15, 1),
        "annealing": (1150, 7, 15, 1),
        "arrhythmia": (1400, 7, 12, 1),
        "audiology-std": (950, 3, 13, 1),
        "balance-scale": (50, 7, 15, 1),
        "balloons": (50, 1, 7, 1),
        "bank": (200, 7, 14, 1),
        "blood": (50, 7, 10, 1),
        "breast-cancer": (50, 1, 13, 1),
        "breast-cancer-wisc": (650, 3, 3, 1),
        "breast-cancer-wisc-diag": (1500, 3, 15, 1),
        "breast-cancer-wisc-prog": (1450, 3, 7, 1),
        "breast-tissue": (1300, 1, 2, 1),
        "car": (250, 3, 15, 1),
        "cardiotocography-10clases": (1350, 3, 15, 1),
        "cardiotocography-3clases": (900, 15, 15, 1),
        "chess-krvk": (800, 1, 15, 1),
        "chess-krvkp": (1350, 3, 14, 1),
        "congressional-voting": (100, 15, 1, 1),
        "conn-bench-sonar-mines-rocks": (1100, 3, 13, 1),
        "conn-bench-vowel-deterding": (1350, 3, 15, 1),
        "connect-4": (1100, 3, 1, 1),
        "contrac": (50, 7, 15, 1),
        "credit-approval": (200, 7, 13, 1),
        "cylinder-bands": (1100, 7, 15, 1),
        "dermatology": (900, 3, 15, 1),
        "echocardiogram": (250, 15, 15, 1),
        "ecoli": (350, 3, 9, 1),
        "energy-y1": (650, 3, 15, 1),
        "energy-y2": (1000, 7, 13, 1),
        "fertility": (150, 7, 12, 1),
        "flags": (900, 15, 10, 1),
        "glass": (1400, 3, 13, 1),
        "haberman-survival": (100, 3, 1, 1),
        "hayes-roth": (50, 1, 8, 1),
        "heart-cleveland": (50, 15, 4, 1),
        "heart-hungarian": (50, 15, 3, 1),
        "heart-switzerland": (50, 15, 4, 1),
        "heart-va": (1350, 15, 6, 1),
        "hepatitis": (1300, 1, 6, 1),
        "hill-valley": (150, 1, 11, 1),
        "horse-colic": (850, 1, 12, 1),
        "ilpd-indian-liver": (1200, 7, 14, 1),
        "image-segmentation": (650, 1, 15, 1),
        "ionosphere": (1150, 1, 14, 1),
        "iris": (50, 3, 1, 1),
        "led-display": (50, 7, 1, 1),
        "lenses": (50, 1, 5, 1),
        "letter": (1500, 1, 15, 1),
        "libras": (1250, 3, 13, 1),
        "low-res-spect": (1400, 7, 14, 1),
        "lung-cancer": (450, 1, 4, 1),
        "lymphography": (1150, 1, 9, 1),
        "magic": (800, 3, 15, 1),
        "mammographic": (150, 7, 14, 1),
        "miniboone": (650, 15, 15, 1),
        "molec-biol-promoter": (1250, 1, 11, 1),
        "molec-biol-splice": (1000, 15, 15, 1),
        "monks-1": (50, 3, 8, 1),
        "monks-2": (400, 1, 14, 1),
        "monks-3": (50, 15, 12, 1),
        "mushroom": (150, 3, 12, 1),
        "musk-1": (1300, 7, 15, 1),
        "musk-2": (1150, 7, 15, 1),
        "nursery": (1000, 3, 15, 1),
        "oocytes_merluccius_nucleus_4d": (1500, 7, 15, 1),
        "oocytes_merluccius_states_2f": (1500, 7, 15, 1),
        "oocytes_trisopterus_nucleus_2f": (1450, 3, 15, 1),
        "oocytes_trisopterus_states_5b": (1450, 7, 15, 1),
        "optical": (1100, 7, 15, 1),
        "ozone": (50, 1, 1, 1),
        "page-blocks": (800, 1, 15, 1),
        "parkinsons": (1200, 1, 13, 1),
        "pendigits": (1500, 1, 15, 1),
        "pima": (50, 1, 13, 1),
        "pittsburg-bridges-MATERIAL": (100, 1, 9, 1),
        "pittsburg-bridges-REL-L": (1200, 1, 7, 1),
        "pittsburg-bridges-SPAN": (450, 7, 7, 1),
        "pittsburg-bridges-T-OR-D": (1000, 1, 8, 1),
        "pittsburg-bridges-TYPE": (50, 7, 5, 1),
        "planning": (50, 1, 1, 1),
        "plant-margin": (1350, 7, 8, 1),
        "plant-shape": (1450, 3, 14, 1),
        "plant-texture": (1500, 7, 10, 1),
        "post-operative": (50, 15, 1, 1),
        "primary-tumor": (950, 3, 1, 1),
        "ringnorm": (1500, 3, 14, 1),
        "seeds": (550, 1, 12, 1),
        "semeion": (1400, 15, 14, 1),
        "soybean": (850, 3, 5, 1),
        "spambase": (1350, 15, 15, 1),
        "spect": (50, 1, 8, 1),
        "spectf": (1100, 15, 15, 1),
        "statlog-australian-credit": (200, 15, 1, 1),
        "statlog-german-credit": (500, 15, 12, 1),
        "statlog-heart": (50, 7, 14, 1),
        "statlog-image": (950, 1, 15, 1),
        "statlog-landsat": (1500, 3, 15, 1),
        "statlog-shuttle": (100, 7, 15, 1),
        "statlog-vehicle": (1450, 7, 14, 1),
        "steel-plates": (1500, 3, 14, 1),
        "synthetic-control": (1350, 3, 14, 1),
        "teaching": (400, 3, 14, 1),
        "thyroid": (300, 7, 15, 1),
        "tic-tac-toe": (750, 1, 15, 1),
        "titanic": (50, 1, 13, 1),
        "trains": (100, 1, 7, 1),
        "twonorm": (1100, 15, 14, 1),
        "vertebral-column-2clases": (250, 3, 14, 1),
        "vertebral-column-3clases": (200, 15, 14, 1),
        "wall-following": (1200, 3, 15, 1),
        "waveform": (1400, 7, 15, 1),
        "waveform-noise": (1300, 15, 14, 1),
        "wine": (850, 1, 15, 1),
        "wine-quality-red": (1100, 1, 13, 1),
        "wine-quality-white": (950, 3, 14, 1),
        "yeast": (1350, 1, 5, 1),
        "zoo": (400, 7, 1, 1),
    }

    def __init__(
        self,
        dataset: torchhd.datasets.CollectionDataset,
        num_feat: int,
        max_epochs: int = 100,
        device: torch.device = None,
    ):
        super(IntRVFLGLVQ, self).__init__()
        self.device = device
        self.num_feat = num_feat
        self.max_epochs = max_epochs
        # Fetch the hyperparameters for the corresponding dataset
        hyper_param = self.INT_RVFL_GLVQ_HYPER[dataset.name]
        # Dimensionality of vectors used when transforming input data
        self.dimensions = hyper_param[0]
        # Parameter of the clipping function used as the part of transforming input data
        self.kappa = hyper_param[1]
        self.transfer_beta = hyper_param[2]
        self.prototypes_per_class = hyper_param[3]
        # Number of classes in the dataset
        self.num_classes = len(dataset.classes)
        # Set up the encoding for the model as specified in "Density"
        self.hypervector_encoding = torchhd.embeddings.Density(
            self.num_feat, self.dimensions
        )

    # Specify encoding function for data samples
    def encode(self, x):
        return self.hypervector_encoding(x).clipping(self.kappa)

    # Specify how to make an inference step and issue a prediction
    def forward(self, x):
        # Make encodings for all data samples in the batch
        encodings = self.encode(x)
        # Classify with GLVQ classifier
        predictions = self.classifier.predict(encodings)
        return predictions

    # Train the classfier
    def fit(
        self,
        train_ld: torch.utils.data.dataloader.DataLoader,
    ):
        # Hyperparameters for GLVQ
        hparams = dict(
            distribution={
                "num_classes": self.num_classes,
                "per_class": self.prototypes_per_class,
            },
            transfer_beta=self.transfer_beta,
            lr=0.1,
        )
        # Initialize the GLVQ classifier
        self.classifier = GLVQ(
            hparams,
            optimizer=torch.optim.Adam,
            prototypes_initializer=pt.initializers.SMCI(train_loader.dataset),
            lr_scheduler=ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
        )

        # Setup trainer
        trainer = pl.Trainer(
            callbacks=[
                TQDMProgressBar(refresh_rate=3),
            ],
            max_epochs=self.max_epochs,
            detect_anomaly=True,
        )

        # Training loop
        trainer.fit(self.classifier, train_ld)


# Specify device to be used for Torch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
# Specifies batch size to be used for the model.
batch_size = 200
# Specifies how many random initializations of the model to evaluate for each dataset in the collection.
repeats = 5


# Get an instance of the UCI benchmark
benchmark = UCIClassificationBenchmark("../data", download=True)
# Perform evaluation
for dataset in benchmark.datasets():
    print(dataset.name)

    # Number of features in the dataset.
    num_feat = dataset.train[0][0].size(-1)
    # Number of classes in the dataset.
    num_classes = len(dataset.train.classes)

    # Get values for min-max normalization and add the transformation
    min_val = torch.min(dataset.train.data, 0).values.to(device)
    max_val = torch.max(dataset.train.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    dataset.test.transform = transform

    # Set up data loaders
    test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

    # Run for the requested number of simulations
    for r in range(repeats):
        # Creates a model to be evaluated. The model should specify both transformation of input data as weel as the algortihm for forming the classifier.
        model = IntRVFLGLVQ(
            getattr(torchhd.datasets, dataset.name), num_feat, device=device
        ).to(device)

        # Replace raw data with hypervector encodings
        train_ds = copy.deepcopy(dataset.train)
        train_ds.data = torch.tensor(model.encode(transform(train_ds.data)))

        # Set up train loader
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )

        # Obtain the classifier for the model
        model.fit(train_loader)
        accuracy = torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=num_classes
        )

        with torch.no_grad():
            for samples, targets in tqdm(test_loader, desc="Testing"):
                samples = samples.to(device)
                # Make prediction
                predictions = model(samples)
                accuracy.update(predictions.cpu(), targets)
        
        print(f"Accuracy: {(accuracy.compute().item() * 100):.3f}%")
        benchmark.report(dataset, accuracy.compute().item())

# Returns a dictionary with names of the datasets and their respective accuracy that is averaged over folds (if applicable) and repeats
benchmark_accuracy = benchmark.score()
print(benchmark_accuracy)
