import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd
import tarfile
import numpy as np

from .utils import download_file_from_google_drive


class CollectionDataset(data.Dataset):
    """`Generic class for loading datasets used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        dataset (string): Name of the dataset. Used when extracting the corresponding files from archive.
        root (string): Root directory containing the files of the dataset.
        has_test (bool): Indicates whether the dataset has a separate test file or not.
        download (bool): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        dataset: str,
        root: str,
        has_test: bool,
        download: bool,
    ):
        self.__dataset = dataset
        self.__has_test = has_test
        root = os.path.join(root, self.__dataset)
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can try to specify download=True to download it"
            )


    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the required files
        if self.__has_test:
            has_train_file = os.path.isfile(
                os.path.join(self.root, self.__dataset + "_train_R.dat")
            )            
        else:
            has_train_file = os.path.isfile(
                os.path.join(self.root, self.__dataset + "_R.dat")
            )
        
        has_k_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos_kfold.dat"))
        has_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos.dat"))

        if has_train_file and has_k_fold_file and has_fold_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files are already downloaded and verified")
            return

        # original data url
        # http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz

        data_dir = os.path.join(self.root, os.pardir)
        archive_path = os.path.join(data_dir, "data_hundreds_classifiers.tar.gz")

        if os.path.isfile(archive_path):
            print("Archive file is already downloaded")
        else:
            download_file_from_google_drive(
                "1Z3tEzCmR-yTvn1ZlAXaeAuVB5a9oCAkk", archive_path
            )

        # Extract archive
        with tarfile.open(archive_path) as file:
            for member in file.getmembers():
                if member.name.startswith(self.__dataset):
                    file.extract(member, data_dir)



class DatasetFourFold(CollectionDataset):
    """`Generic class for loading datasets without separate test data that were used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        dataset (string): Name of the dataset. Used when downloading
        root (string): Root directory containing the files of the dataset.
        train (bool): If True, returns training (sub)set from the file storing training data as further determined by fold and hyper_search variables.
            Otherwise returns a subset of train dataset if hypersearch is performed (hyper_search = True) if not (hyper_search = False) returns a subset of training dataset 
            as specified in ``conxuntos_kfold.dat`` if fold number is correct. Otherwise issues an error.
        fold (int): Specifies which fold number to use. The default value of -1 returns the whole train data from the corresponding file.
            Values between 0 and 3 specify, which fold in ``conxuntos_kfold.dat`` to use. Relevant only if hyper_search is set to False and  0<=fold<=3.
            Indices in even rows (zero indexing) of ``conxuntos_kfold.dat`` correspond to train subsets while indices in odd rows correspond to test subsets.
        hyper_search (bool): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if train = True)
            while the second row corresponds to test indices (used if train = False).
        download (bool): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable): A function/transform that takes in the
            target and transforms it.
    """

    # Number of folds for cross validation
    NUM_FOLDS = 4
    # Indicates whether the dataset has a separate test file
    HAS_TEST = False

    def __init__(
        self,
        dataset: str,
        root: str,
        train: bool,
        fold: int,
        hyper_search: bool,
        download: bool,
        transform: Optional[Callable],
        target_transform: Optional[Callable],
    ):
        super().__init__(dataset,root,self.HAS_TEST,download)
        self.__dataset = dataset
        self.train = train
        self.fold = fold
        self.hyper_search = hyper_search
        self.transform = transform
        self.target_transform = target_transform

        self._load_data()

    def __len__(self) -> int:
        return self.data.size(0)

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


    def _load_data(self):
        # Files with pregenerated folds with indices
        k_fold_file = "conxuntos_kfold.dat"
        hyper_search_file = "conxuntos.dat"

        # Names of data file
        if not self.HAS_TEST:
            train_data_file = self.__dataset + "_R.dat"

        # Load train data
        train_data = pd.read_csv(
            os.path.join(self.root, train_data_file),
            sep="\t",
            header=None,
            skiprows=1,
        )
        # Skip number & target columns
        train_targets_all = train_data.values[:, -1].astype(int)
        train_data_all = train_data.values[:, 1:-1]

        # Load fold used in hyperparameter search if necessary
        if self.hyper_search:
            hyper_search_ind = pd.read_csv(
                os.path.join(self.root, hyper_search_file), sep=" ", header=None
            ).values
        # Load k-fold if necessary
        if self.fold >= 0:
            cross_val_ind = pd.read_csv(
                os.path.join(self.root, k_fold_file), sep=" ", header=None
            ).values

        # train = True  - some form of train data is requested
        if self.train:
            # If hyperparameter search is being performed
            if self.hyper_search:
                indices = hyper_search_ind[
                    0, np.invert(np.isnan(hyper_search_ind[0, :]))
                ].astype(int)
                self.data = torch.tensor(train_data_all[indices, :], dtype=torch.float)
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )
            # If regular training is going on
            else:
                # If no fold is specified
                if self.fold == -1:
                    self.data = torch.tensor(train_data_all, dtype=torch.float)
                    self.targets = torch.tensor(train_targets_all, dtype=torch.long)
                elif (self.fold >= 0) and (self.fold < self.NUM_FOLDS):
                    # Identify which row of cross_val_ind to index - even number
                    fold_index = self.fold * 2
                    indices = cross_val_ind[
                        fold_index, np.invert(np.isnan(cross_val_ind[fold_index, :]))
                    ].astype(int)
                    self.data = torch.tensor(
                        train_data_all[indices, :], dtype=torch.float
                    )
                    self.targets = torch.tensor(
                        train_targets_all[indices], dtype=torch.long
                    )
                # Wrong fold number
                else:
                    raise ValueError(
                        f"No such fold number. Please check that it is specified correctly"
                    )
        # train = False  - some form of test data is requested
        else:
            # If hyperparameter search is being performed
            if self.hyper_search:
                indices = hyper_search_ind[
                    1, np.invert(np.isnan(hyper_search_ind[1, :]))
                ].astype(int)
                self.data = torch.tensor(train_data_all[indices, :], dtype=torch.float)
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )
            # If regular testing is going on
            else:
                # Check that fold number is within limits otherwise issue error
                if (self.fold >= 0) and (self.fold < self.NUM_FOLDS):
                    # Identify which row of cross_val_ind to index - odd number
                    fold_index = self.fold * 2 + 1
                    indices = cross_val_ind[
                        fold_index,
                        np.invert(np.isnan(cross_val_ind[fold_index, :])),
                    ].astype(int)
                    self.data = torch.tensor(
                        train_data_all[indices, :], dtype=torch.float
                    )
                    self.targets = torch.tensor(
                        train_targets_all[indices], dtype=torch.long
                    )
                else:
                    raise ValueError(
                        f"This dataset does not have a separate file for test data. Please check that fold is specified correctly"
                    )


class DatasetTrainTest(CollectionDataset):
    """`Generic class for loading datasets with separate files for train and test data that were used in `Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? <https://jmlr.org/papers/v15/delgado14a.html>`_.

    Args:
        dataset (string): Name of the dataset. Used when downloading
        root (string): Root directory containing the files of the dataset.
        train (bool): If True, returns training (sub)set from the file storing training data as further determined by hyper_search variable.
            Otherwise returns a subset of train dataset if hypersearch is performed (hyper_search = True) if not (hyper_search = False) returns test set.
        hyper_search (bool): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if train = True)
            while the second row corresponds to test indices (used if train = False).
        download (bool): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable): A function/transform that takes in the
            target and transforms it.
    """

    # Indicates whether the dataset has a separate test file
    HAS_TEST = True

    def __init__(
        self,
        dataset: str,
        root: str,
        train: bool,
        hyper_search: bool,
        download: bool,
        transform: Optional[Callable],
        target_transform: Optional[Callable],
    ):
        super().__init__(dataset,root,self.HAS_TEST,download)
        self.__dataset = dataset
        self.train = train
        self.hyper_search = hyper_search
        self.transform = transform
        self.target_transform = target_transform

        self._load_data()

    def __len__(self) -> int:
        return self.data.size(0)

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


    def _load_data(self):
        # Files with pregenerated folds with indices
        hyper_search_file = "conxuntos.dat"

        # Names of data file
        if self.HAS_TEST:
            train_data_file = self.__dataset + "_train_R.dat"
            test_data_file = self.__dataset + "_test_R.dat"

        # Load train data
        train_data = pd.read_csv(
            os.path.join(self.root, train_data_file),
            sep="\t",
            header=None,
            skiprows=1,
        )
        # Skip number & target columns
        train_targets_all = train_data.values[:, -1].astype(int)
        train_data_all = train_data.values[:, 1:-1]

        # Load test data
        test_data = pd.read_csv(
            os.path.join(self.root, test_data_file),
            sep="\t",
            header=None,
            skiprows=1,
        )
        # Skip number & target columns
        test_targets = test_data.values[:, -1].astype(int)
        test_data = test_data.values[:, 1:-1]

        # Load fold used in hyperparameter search if necessary
        if self.hyper_search:
            hyper_search_ind = pd.read_csv(
                os.path.join(self.root, hyper_search_file), sep=" ", header=None
            ).values

        # train = True  - some form of train data is requested
        if self.train:
            # If hyperparameter search is being performed
            if self.hyper_search:
                indices = hyper_search_ind[
                    0, np.invert(np.isnan(hyper_search_ind[0, :]))
                ].astype(int)
                self.data = torch.tensor(train_data_all[indices, :], dtype=torch.float)
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )
            # If regular training is going on
            else:
                self.data = torch.tensor(train_data_all, dtype=torch.float)
                self.targets = torch.tensor(train_targets_all, dtype=torch.long)

        # train = False  - some form of test data is requested
        else:
            # If hyperparameter search is being performed
            if self.hyper_search:
                indices = hyper_search_ind[
                    1, np.invert(np.isnan(hyper_search_ind[1, :]))
                ].astype(int)
                self.data = torch.tensor(train_data_all[indices, :], dtype=torch.float)
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )
            # If regular testing is going on
            else:
                # If there is a dedicted file
                if self.HAS_TEST:
                    self.data = torch.tensor(test_data, dtype=torch.float)
                    self.targets = torch.tensor(test_targets, dtype=torch.long)
                else:
                    raise ValueError(
                        f"This dataset does not have a separate file for test data. Please check if the correct class is used."
                    )




