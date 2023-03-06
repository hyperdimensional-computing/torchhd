import os
import shutil
import pytest
import torch
import torch.utils.data as data

import torchhd.datasets
from torchhd.datasets import UCIClassificationBenchmark


dataset_metadata = {
    "Abalone": (8, 3, 3133),
    "AcuteInflammation": (6, 2, 90),
    "AcuteNephritis": (6, 2, 90),
    "Adult": (14, 2, 32561),
    "Annealing": (31, 5, 798),
    "Arrhythmia": (262, 13, 339),
    "AudiologyStd": (59, 18, 171),
    "BalanceScale": (4, 3, 469),
    "Balloons": (4, 2, 12),
    "Bank": (16, 2, 3391),
    "Blood": (4, 2, 561),
    "BreastCancer": (9, 2, 215),
    "BreastCancerWisc": (9, 2, 524),
    "BreastCancerWiscDiag": (30, 2, 427),
    "BreastCancerWiscProg": (33, 2, 149),
    "BreastTissue": (9, 6, 80),
    "Car": (6, 4, 1296),
    "Cardiotocography10Clases": (21, 10, 1595),
    "Cardiotocography3Clases": (21, 3, 1595),
    "ChessKrvk": (6, 18, 21042),
    "ChessKrvkp": (36, 2, 2397),
    "CongressionalVoting": (16, 2, 326),
    "ConnBenchSonarMinesRocks": (60, 2, 156),
    "ConnBenchVowelDeterding": (11, 11, 528),
    "Connect4": (42, 3, 50668),
    "Contrac": (9, 3, 1105),
    "CreditApproval": (15, 2, 518),
    "CylinderBands": (35, 2, 384),
    "Dermatology": (34, 6, 275),
    "Echocardiogram": (10, 2, 98),
    "Ecoli": (7, 8, 252),
    "EnergyY1": (8, 3, 576),
    "EnergyY2": (8, 3, 576),
    "Fertility": (9, 2, 75),
    "Flags": (28, 8, 146),
    "Glass": (9, 6, 161),
    "HabermanSurvival": (3, 2, 230),
    "HayesRoth": (3, 3, 132),
    "HeartCleveland": (13, 5, 227),
    "HeartHungarian": (12, 5, 221),
    "HeartSwitzerland": (12, 5, 92),
    "HeartVa": (12, 5, 150),
    "Hepatitis": (19, 2, 116),
    "HillValley": (100, 2, 606),
    "HorseColic": (25, 2, 300),
    "IlpdIndianLiver": (9, 2, 437),
    "ImageSegmentation": (18, 7, 210),
    "Ionosphere": (33, 2, 263),
    "Iris": (4, 3, 113),
    "LedDisplay": (7, 10, 750),
    "Lenses": (4, 3, 18),
    "Letter": (16, 26, 15000),
    "Libras": (90, 15, 270),
    "LowResSpect": (100, 9, 398),
    "LungCancer": (56, 3, 24),
    "Lymphography": (18, 4, 111),
    "Magic": (10, 2, 14265),
    "Mammographic": (5, 2, 721),
    "Miniboone": (50, 2, 97548),
    "MolecBiolPromoter": (57, 2, 80),
    "MolecBiolSplice": (60, 3, 2393),
    "Monks1": (6, 2, 124),
    "Monks2": (6, 2, 169),
    "Monks3": (6, 2, 122),
    "Mushroom": (21, 2, 6093),
    "Musk1": (166, 2, 357),
    "Musk2": (166, 2, 4949),
    "Nursery": (8, 5, 9720),
    "OocytesMerlucciusNucleus4d": (41, 2, 767),
    "OocytesMerlucciusStates2f": (25, 3, 767),
    "OocytesTrisopterusNucleus2f": (25, 2, 684),
    "OocytesTrisopterusStates5b": (32, 3, 684),
    "Optical": (62, 10, 3823),
    "Ozone": (72, 2, 1902),
    "PageBlocks": (10, 5, 4105),
    "Parkinsons": (22, 2, 146),
    "Pendigits": (16, 10, 7494),
    "Pima": (8, 2, 576),
    "PittsburgBridgesMaterial": (7, 3, 80),
    "PittsburgBridgesRelL": (7, 3, 77),
    "PittsburgBridgesSpan": (7, 3, 69),
    "PittsburgBridgesTOrD": (7, 2, 77),
    "PittsburgBridgesType": (7, 6, 79),
    "Planning": (12, 2, 137),
    "PlantMargin": (64, 100, 1200),
    "PlantShape": (64, 100, 1200),
    "PlantTexture": (64, 100, 1199),
    "PostOperative": (8, 3, 68),
    "PrimaryTumor": (17, 15, 248),
    "Ringnorm": (20, 2, 5550),
    "Seeds": (7, 3, 158),
    "Semeion": (256, 10, 1195),
    "Soybean": (35, 18, 307),
    "Spambase": (57, 2, 3451),
    "Spect": (22, 2, 79),
    "Spectf": (44, 2, 80),
    "StatlogAustralianCredit": (14, 2, 518),
    "StatlogGermanCredit": (24, 2, 750),
    "StatlogHeart": (13, 2, 203),
    "StatlogImage": (18, 7, 1733),
    "StatlogLandsat": (36, 6, 4435),
    "StatlogShuttle": (9, 7, 43500),
    "StatlogVehicle": (18, 4, 635),
    "SteelPlates": (27, 7, 1456),
    "SyntheticControl": (60, 6, 450),
    "Teaching": (5, 3, 113),
    "Thyroid": (21, 3, 3772),
    "TicTacToe": (9, 2, 719),
    "Titanic": (3, 2, 1651),
    "Trains": (29, 2, 8),
    "Twonorm": (20, 2, 5550),
    "VertebralColumn2Clases": (6, 2, 233),
    "VertebralColumn3Clases": (6, 3, 233),
    "WallFollowing": (24, 4, 4092),
    "Waveform": (21, 3, 3750),
    "WaveformNoise": (40, 3, 3750),
    "Wine": (13, 3, 134),
    "WineQualityRed": (11, 6, 1199),
    "WineQualityWhite": (11, 7, 3674),
    "Yeast": (8, 10, 1113),
    "Zoo": (16, 7, 76),
}


@pytest.fixture()
def cleandir():
    if os.path.isdir("./data"):
        shutil.rmtree("./data")


def is_dataset_class(key_value_pair):
    ds_name, ds_class = key_value_pair

    if not isinstance(ds_class, type):
        return False

    if ds_name in {
        "CollectionDataset",
        "DatasetFourFold",
        "DatasetTrainTest",
    }:
        return False

    return issubclass(ds_class, data.Dataset)


dataset_names = filter(is_dataset_class, torchhd.datasets.__dict__.items())
dataset_names = [name for name, ds in dataset_names]


class TestDataset:
    def test_benchmark(self):
        seen_datasets = set()
        benchmark = UCIClassificationBenchmark("./data", download=True)
        for dataset in benchmark.datasets():
            num_feat = dataset.train[0][0].size(-1)
            num_classes = len(dataset.train.classes)
            num_instances = len(dataset.train)
            assert dataset_metadata[dataset.name][0] == num_feat
            assert dataset_metadata[dataset.name][1] == num_classes
            assert dataset_metadata[dataset.name][2] == num_instances

            seen_datasets.add(dataset.name)
            benchmark.report(dataset, 0.5)

        assert len(benchmark.dataset_names) == len(seen_datasets)

        all_metrics = benchmark.score()
        for dataset in benchmark.datasets():
            assert all_metrics[dataset.name][0] == 0.5

    @pytest.mark.parametrize("dataset_name", dataset_names)
    def test_datasets_dowload(self, cleandir, dataset_name):
        dataset_class = getattr(torchhd.datasets, dataset_name)

        dataset = dataset_class("./data", download=True)
        assert len(dataset) > 0

        # Test if downloaded ds can be opened with download=False
        dataset = dataset_class("./data", download=False)
        assert len(dataset) > 0
