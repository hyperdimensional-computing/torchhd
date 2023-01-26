from typing import List
from torchhd.datasets import DatasetTrainTest


class Soybean(DatasetTrainTest):
    """`Soybean (Large) <https://archive.ics.uci.edu/ml/datasets/Soybean+(Large)>`_ dataset.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by hyper_search variable.
            Otherwise returns a subset of train dataset if hyperparameter search is performed (``hyper_search = True``) if not (``hyper_search = False``) returns test set.
        hyper_search (bool, optional): If True, creates dataset using indices in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if ``train = True``)
            while the second row corresponds to test indices (used if ``train = False``).
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    name = "soybean"
    classes: List[str] = [
        "diaporthe-stem-canker",
        "charcoal-rot",
        "rhizoctonia-root-rot",
        "phytophthora-rot",
        "brown-stem-rot",
        "powdery-mildew",
        "downy-mildew",
        "brown-spot",
        "bacterial-blight",
        "bacterial-pustule",
        "purple-seed-stain",
        "anthracnose",
        "phyllosticta-leaf-spot",
        "alternarialeaf-spot",
        "frog-eye-leaf-spot",
        "diaporthe-pod-&-stem-blight",
        "cyst-nematode",
        "herbicide-injury",
    ]
