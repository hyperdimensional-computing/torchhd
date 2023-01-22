import torch
import torch.nn as nn
from tqdm import tqdm

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.map import MAP
from torch import Tensor



# DK: For classes and functions below we likely want to create a separate file as they might be more useful then just for this example
class EncodingDensityClipped():
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
        # DK: this will likely have to be double-checked once API for embeddigns is revised
        self.density_encoding = embeddings.Thermometer(
            dimensions + 1, dimensions, low=0, high=1
        )
        self.kappa = kappa

    def encode(self, x):
        # Perform binding of key and value vectors
        sample_hv = MAP.bind(self.key, self.density_encoding(x))
        # Perform the superposition operation on the bound key-value pairs
        sample_hv = MAP.multibundle(sample_hv)        
        # Perform clipping function on the result of the superposition operation and return
        return torchhd.clipping(sample_hv, self.kappa)

# Function that forms the classifier (readout matrix) with the ridge regression
def classifier_ridge(
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
            labels_one_hot[torch.arange(count,count+samples.size(0)), labels] = 1
            
            # Make transformation into high-dimensional space
            samples_hv = encoding_function(samples)
            total_samples_hv[count:count+samples.size(0), :] = samples_hv

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


class intRVFLRidge(nn.Module):
    """Class implementing integer random vector functional link network (intRVFL) model as described in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        dataset (torchhd.datasets.CollectionDataset): Specifies a dataset to be evaluted by the model.
        num_feat (int): Number of features in the dataset.
        num_classes (int): Number of classes in the dataset.
        dimensions (int): Dimensionality of vectors used when transforming input data.
        kappa (int): Parameter of the clipping function used as the part of transforming input data.
        lamb (float): Regularization parameter used for ridge regression.
        device (torch.device): Specifies device to be used for Torch.
    """

    # These values of hyperparameters were found via the grid search for intRVFL model as described in the article.
    int_rvfl_hyper = {
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
        dataset: torchhd.datasets.CollectionDataset,
        num_feat: int,
        device: torch.device,
    ):
        super(intRVFLRidge, self).__init__()

        # Fetch the hyperparameters for the corresponding dataset
        hyper_param = self.int_rvfl_hyper[dataset.name]
        self.num_feat = num_feat
        self.device = device
        self.dimensions = hyper_param[0]
        self.lamb = hyper_param[2]
        self.kappa = hyper_param[1]
        # Get number of classes
        self.num_classes = len(dataset.classes)
        self.classify = nn.Linear(self.dimensions, self.num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)
        self.hypervector_encoding = EncodingDensityClipped (self.dimensions, self.num_feat, self.kappa)

    def encode(self, x):
        return self.hypervector_encoding.encode(x)


    def forward(self, x):
        enc = self.encode(x)
        # Get similarity values for each class assuming implicitly that there is only one prototype per class. This does not have to be the case in general.
        logit = self.classify(enc)
        # Form predictions
        predictions = torch.argmax(logit, dim=-1)
        return predictions

    # Train the classfier
    def fit(
        self,
        train_ld: torch.utils.data.dataloader.DataLoader,
    ):
        # Gets classifier (readout matrix) via the ridge regression
        Wout = classifier_ridge(
            train_ld,
            self.dimensions,
            self.num_classes,
            self.lamb,
            self.encode,
            self.hypervector_encoding.key.dtype,
            self.device,
        )
        # Assign the obtained classifier to the output
        with torch.no_grad():
            self.classify.weight[:] = Wout


class BenchmarkCollectionDataset:
    """Helper class for measuring the accuracy of a model on the collection of 121 datasets from `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        model_class (nn.Module): Specifies a class for a model to be evaluated. The model should specify both transformation of input data as weel as the algortihm for forming the classifier.
        repeats (int, optional): Specifies how many random initializations of the model to evaluate for each dataset in the collection.
        batch_size (int, optional): Specifies batch size to be used for the model.
        device (torch.device, optional): Specifies device to be used for Torch.
    """

    # All datasets included in the collection
    dataset_collection = [
        "Abalone",
        "AcuteInflammation",
        "AcuteNephritis",
        "Adult",
        "Annealing",
        "Arrhythmia",
        "AudiologyStd",
        "BalanceScale",
        "Balloons",
        "Bank",
        "Blood",
        "BreastCancer",
        "BreastCancerWisc",
        "BreastCancerWiscDiag",
        "BreastCancerWiscProg",
        "BreastTissue",
        "Car",
        "Cardiotocography10Clases",
        "Cardiotocography3Clases",
        "ChessKrvk",
        "ChessKrvkp",
        "CongressionalVoting",
        "ConnBenchSonarMinesRocks",
        "ConnBenchVowelDeterding",
        "Connect4",
        "Contrac",
        "CreditApproval",
        "CylinderBands",
        "Dermatology",
        "Echocardiogram",
        "Ecoli",
        "EnergyY1",
        "EnergyY2",
        "Fertility",
        "Flags",
        "Glass",
        "HabermanSurvival",
        "HayesRoth",
        "HeartCleveland",
        "HeartHungarian",
        "HeartSwitzerland",
        "HeartVa",
        "Hepatitis",
        "HillValley",
        "HorseColic",
        "IlpdIndianLiver",
        "ImageSegmentation",
        "Ionosphere",
        "Iris",
        "LedDisplay",
        "Lenses",
        "Letter",
        "Libras",
        "LowResSpect",
        "LungCancer",
        "Lymphography",
        "Magic",
        "Mammographic",
        "Miniboone",
        "MolecBiolPromoter",
        "MolecBiolSplice",
        "Monks1",
        "Monks2",
        "Monks3",
        "Mushroom",
        "Musk1",
        "Musk2",
        "Nursery",
        "OocytesMerlucciusNucleus4d",
        "OocytesMerlucciusStates2f",
        "OocytesTrisopterusNucleus2f",
        "OocytesTrisopterusStates5b",
        "Optical",
        "Ozone",
        "PageBlocks",
        "Parkinsons",
        "Pendigits",
        "Pima",
        "PittsburgBridgesMaterial",
        "PittsburgBridgesRelL",
        "PittsburgBridgesSpan",
        "PittsburgBridgesTOrD",
        "PittsburgBridgesType",
        "Planning",
        "PlantMargin",
        "PlantShape",
        "PlantTexture",
        "PostOperative",
        "PrimaryTumor",
        "Ringnorm",
        "Seeds",
        "Semeion",
        "Soybean",
        "Spambase",
        "Spect",
        "Spectf",
        "StatlogAustralianCredit",
        "StatlogGermanCredit",
        "StatlogHeart",
        "StatlogImage",
        "StatlogLandsat",
        "StatlogShuttle",
        "StatlogVehicle",
        "SteelPlates",
        "SyntheticControl",
        "Teaching",
        "Thyroid",
        "TicTacToe",
        "Titanic",
        "Trains",
        "Twonorm",
        "VertebralColumn2Clases",
        "VertebralColumn3Clases",
        "WallFollowing",
        "Waveform",
        "WaveformNoise",
        "Wine",
        "WineQualityRed",
        "WineQualityWhite",
        "Yeast",
        "Zoo",
    ]

    def __init__(
        self,
        model_class: nn.Module,
        repeats: int = 1,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_class = model_class
        self.repeats = repeats
        self.batch_size = batch_size
        self.device = device

        # Collects average accuracy for each simulation of each dataset
        self.accuracy_collection = torch.zeros(
            len(self.dataset_collection),
            self.repeats,
            device=self.device,
        )

    # Function that performs min-max normalization of the input data samples 
    def normalize(self, input: Tensor) -> Tensor:
        return (input - self.min_val) / (self.max_val - self.min_val)

    ## Function compute the accuracy for a given model and data
    def compute_accuracy(self, model, train_ds, test_ds):
        # Make data loaders
        train_ld = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size
        )


        # Obtain the classifier for the model
        model.fit(train_ld)

        accuracy = torchmetrics.Accuracy()
        with torch.no_grad():
            for samples, labels in tqdm(test_ld, desc="Testing"):
                samples = samples.to(self.device)                
                # Make prediction
                predictions = model(samples)
                accuracy.update(predictions.cpu(), labels)

        return accuracy.compute().item()

    def evaluate(self):
        # For all datasets in the collection
        for i, [train_ds, test_ds] in enumerate(torchhd.datasets.UCIDatasetCollection(self.dataset_collection, "../data", True)):
            num_feat = train_ds[0][0][0].size(-1)
            # Run for the requested number of simulations
            for repeat in range(self.repeats):
                accuracy_dataset = 0.0
                for fold_id in range(len(train_ds)):
                    # Set test and train datasets for the current fold
                    # Get values for min-max normalization and add the transformation
                    # DK: doing this inside FOR loop for self.repeats because uncertain if otherwise each fold will get its own min_val and max_val in self.normalize 
                    self.min_val = torch.min(train_ds[fold_id].data, 0).values.to(self.device)
                    self.max_val = torch.max(train_ds[fold_id].data, 0).values.to(self.device)                        
                    train_ds[fold_id].transform = self.normalize
                    test_ds[fold_id].transform = self.normalize   
                    
                    # Initialize model
                    model = self.model_class(
                        getattr(torchhd.datasets, self.dataset_collection[i]), num_feat, self.device
                    ).to(self.device)
                    # Train model and computer the validation accuracy
                    accuracy_dataset += self.compute_accuracy(
                        model, train_ds[fold_id], test_ds[fold_id]
                    )

                # Average over folds
                accuracy_dataset = accuracy_dataset / len(train_ds)

                # Update the statistics for the current dataset and simulation
                self.accuracy_collection[i, repeat] = accuracy_dataset

            print(
                f"Dataset {i} - {self.dataset_collection[i]}: average testing accuracy of {(torch.mean(self.accuracy_collection[i, :])* 100):.2f}%"
            )

        return self.accuracy_collection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

benchmark = BenchmarkCollectionDataset(intRVFLRidge, repeats=3, batch_size = 10, device=device)
accuracy_collection = benchmark.evaluate()
