from typing import Callable, Optional, Tuple, List
from torchhd.datasets import DatasetFourFold


class Abalone(DatasetFourFold):
    """`Abalone <https://archive.ics.uci.edu/ml/datasets/abalone>`_ dataset.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by fold and hyper_search variables.
            Otherwise returns a subset of train dataset if hypersearch is performed (hyper_search = True) if not (hyper_search = False) returns a subset of training dataset
            as specified in ``conxuntos_kfold.dat`` if fold number is correct. Otherwise issues an error.
        fold (int, optional): Specifies which fold number to use. The default value of -1 returns the whole train data from the corresponding file.
            Values between 0 and 3 specify, which fold in ``conxuntos_kfold.dat`` to use. Relevant only if hyper_search is set to False and  0<=fold<=3.
            Indices in even rows (zero indexing) of ``conxuntos_kfold.dat`` correspond to train subsets while indices in odd rows correspond to test subsets.
        hyper_search (bool, optional): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if train = True)
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
        "0",
        "1",
        "2",
    ]

    # Useful to specify name once to avoid changing it in several places
    DATASET = "abalone"

    def __init__(
        self,
        root: str,
        train: bool = True,
        fold: int = -1,
        hyper_search: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            self.DATASET,
            root,
            train,
            fold,
            hyper_search,
            download,
            transform,
            target_transform,
        )
