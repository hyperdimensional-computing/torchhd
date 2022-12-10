from typing import Callable, Optional, Tuple, List
from torchhd.datasets import DatasetTrainTest

class Adult(DatasetTrainTest):
    """`Adult <https://archive.ics.uci.edu/ml/datasets/adult>`_ dataset.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by hyper_search variable.
            Otherwise returns a subset of train dataset if hypersearch is performed (hyper_search = True) if not (hyper_search = False) returns test set.
        hyper_search (bool, optional): If True, creates dataset from the training data using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if train = True)
            while the second row corresponds to test indices (used if train = False).
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes: List[str] = [
        ">50K",
        "<=50K",
    ]
   
    # Useful to specify name once to avoid changing it in several places
    DATASET = "adult"

    def __init__(
        self,
        root: str,
        train: bool = True,
        fold: int = -1,
        hyper_search: bool = False,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(self.DATASET,root,train,hyper_search,download,transform,target_transform)