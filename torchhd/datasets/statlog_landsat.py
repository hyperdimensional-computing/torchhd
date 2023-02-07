from typing import List
from torchhd.datasets import DatasetTrainTest


class StatlogLandsat(DatasetTrainTest):
    """`Statlog (Landsat Satellite) <https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)>`_ dataset.

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

    **STATS**

    .. list-table::
       :widths: 10 10 10 10
       :header-rows: 1

       * - Instances
         - Attributes
         - Task
         - Area
       * - 6435
         - 36
         - Classification
         - Physical

    """

    name = "statlog-landsat"
    classes: List[str] = [
        "red soil",
        "cotton crop",
        "grey soil",
        "damp grey soil",
        "soil with vegetation stubble",
        "mixture class (all types present)",
    ]
