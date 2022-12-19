import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
from torch.utils import data
import pandas as pd
import tarfile
import numpy as np

from .utils import download_file_from_google_drive


class CollectionDataset(data.Dataset):
    """Generic class for loading datasets used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        root (string): Root directory containing the files of the dataset.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    # Name of the dataset. Used when extracting the corresponding files from archive; needs to be specified by subclasses
    name: str
    data: Tensor
    targets: Tensor

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, self.name)
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can try to specify download=True to download it"
            )

        self._load_data()

    def _check_integrity(self) -> bool:
        return NotImplemented

    def _load_data(self):
        return NotImplemented

    def __len__(self) -> int:
        return self.data.size(0)

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: (sample, target) where target is the index of the target class
        """
        sample = self.data[index]
        label = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files are already downloaded and verified")
            return

        # original data url:
        # http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz

        data_dir = os.path.join(self.root, os.pardir)
        archive_path = os.path.join(data_dir, "data_hundreds_classifiers.tar.gz")

        if os.path.isfile(archive_path):
            print("Archive file is already downloaded")
        else:
            download_file_from_google_drive(
                "1Z3tEzCmR-yTvn1ZlAXaeAuVB5a9oCAkk", archive_path
            )

        # Extract only the requested dataset from the archive
        with tarfile.open(archive_path) as file:
            for member in file.getmembers():
                if member.name.startswith(self.name):
                    file.extract(member, data_dir)


class DatasetFourFold(CollectionDataset):
    """Generic class for loading datasets without separate test data that were used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by fold and hyper_search variables.
            Otherwise returns a subset of train dataset if hypersearch is performed (``hyper_search = True``) if not (``hyper_search = False``) returns a subset of training dataset
            as specified in ``conxuntos_kfold.dat`` if fold number is correct. Otherwise issues an error.
        fold (int, optional): Specifies which fold number to use. The default value of -1 returns all the training data from the corresponding file.
            Values between 0 and 3 specify, which fold in ``conxuntos_kfold.dat`` to use. Relevant only if hyper_search is set to False and ``0 <= fold <= 3``.
            Indices in even rows (zero indexing) of ``conxuntos_kfold.dat`` correspond to train subsets while indices in odd rows correspond to test subsets.
        hyper_search (bool, optional): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if ``train = True``)
            while the second row corresponds to test indices (used if ``train = False``).
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    # Number of folds for cross validation
    num_folds = 4

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
        if (fold < -1) or (fold >= self.num_folds):
            raise ValueError(
                f"Fold number {fold} is not available. Fold should be between -1 and {self.num_folds - 1}"
            )

        if not train and fold == -1 and not hyper_search:
            raise ValueError(
                "This dataset does not have a separate file for test data. Please check that fold is specified correctly."
            )

        self.train = train
        self.fold = fold
        self.hyper_search = hyper_search
        super().__init__(root, transform, target_transform, download)

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the required files
        has_train_file = os.path.isfile(os.path.join(self.root, self.name + "_R.dat"))

        has_k_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos_kfold.dat"))
        has_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos.dat"))

        if has_train_file and has_k_fold_file and has_fold_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        data_path = os.path.join(self.root, self.name + "_R.dat")
        data = np.loadtxt(data_path, skiprows=1, dtype=np.float32)
        # Separate the targets from the data
        targets = torch.from_numpy(data[:, -1].astype(np.int64))
        data = torch.from_numpy(data[:, 1:-1])

        # Load fold used in hyperparameter search if necessary
        if self.hyper_search:
            # Files with pre-generated folds with indices
            hyper_split_path = os.path.join(self.root, "conxuntos.dat")
            line_idx = 0 if self.train else 1

            with open(hyper_split_path, "r") as file:
                lines = file.readlines()
                indices = np.fromstring(lines[line_idx], sep=" ", dtype=np.int64)
                indices = torch.from_numpy(indices)

                data = data[indices]
                targets = targets[indices]

        elif self.fold != -1:
            # Files with pre-generated folds with indices
            k_fold_path = os.path.join(self.root, "conxuntos_kfold.dat")
            fold_idx = self.fold * 2
            if not self.train:
                fold_idx += 1

            with open(k_fold_path, "r") as file:
                lines = file.readlines()
                indices = np.fromstring(lines[fold_idx], sep=" ", dtype=np.int64)
                indices = torch.from_numpy(indices)

                data = data[indices]
                targets = targets[indices]

        self.data = data
        self.targets = targets


class DatasetTrainTest(CollectionDataset):
    """Generic class for loading datasets with separate files for train and test data that were used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

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

    def __init__(
        self,
        root: str,
        train: bool = True,
        hyper_search: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.train = train
        self.hyper_search = hyper_search
        super().__init__(root, transform, target_transform, download)

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the required files
        has_train_file = os.path.isfile(
            os.path.join(self.root, self.name + "_train_R.dat")
        )

        has_k_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos_kfold.dat"))
        has_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos.dat"))

        if has_train_file and has_k_fold_file and has_fold_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        if self.train or self.hyper_search:
            data_name = self.name + "_train_R.dat"
        else:
            data_name = self.name + "_test_R.dat"

        data_path = os.path.join(self.root, data_name)

        data = np.loadtxt(data_path, skiprows=1, dtype=np.float32)
        # Separate the targets from the data
        targets = torch.from_numpy(data[:, -1].astype(np.int64))
        data = torch.from_numpy(data[:, 1:-1])

        # Load fold used in hyperparameter search if necessary
        if self.hyper_search:

            # Files with pre-generated folds with indices
            hyper_split_path = os.path.join(self.root, "conxuntos.dat")
            line_idx = 0 if self.train else 1

            with open(hyper_split_path, "r") as file:
                lines = file.readlines()
                indices = np.fromstring(lines[line_idx], sep=" ", dtype=np.int64)
                indices = torch.from_numpy(indices)

                data = data[indices]
                targets = targets[indices]

        self.data = data
        self.targets = targets
