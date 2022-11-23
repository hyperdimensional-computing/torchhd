import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd
import tarfile
import numpy as np

from .utils import download_file_from_google_drive


class Abalone(data.Dataset):
    """`Abalone <https://archive.ics.uci.edu/ml/datasets/abalone>`_ dataset.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set stored in ``train.data`` as further determined by fold_id and fold_train variables.
            Otherwise tries to return test set if ``test.data`` exists, issues error in case it is not available.
        fold_id (int, optional): Specifies which fold number to use. Relevant only if train is set to True. The default value of 0 returns the whole data in ``train.data``.
            Values between 1 and 4 specify, which fold in ``k_fold_cross_val.data`` to use.
        fold_val (bool, optional): If True, creates dataset using indeces in ``k_fold_cross_val.data`` for for validation part (some odd row) specified by fold_id.
            Otherwise uses indices the training part (some even row) of the fold. Relevant only if train is set to True and fold_id>0.
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

    def __init__(
        self,
        root: str,
        train: bool = True,
        fold_id: int = 0,
        fold_val: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, "abalone")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.train = train
        self.fold_id = fold_id
        self.fold_val = fold_val
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

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

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the required files
        has_train_file = os.path.isfile(os.path.join(self.root, "abalone_R.dat"))
        has_k_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos_kfold.dat"))
        if has_train_file and has_k_fold_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        train_data_file = "abalone_R.dat"
        val_file = "conxuntos_kfold.dat"

        if self.train:
            train_data = pd.read_csv(
                os.path.join(self.root, train_data_file),
                sep="\t",
                header=None,
                skiprows=1,
            )
            train_data_all = train_data.values[:, 1:-1]
            train_targets_all = train_data.values[:, -1].astype(int)

            if self.fold_id == 0:
                self.data = torch.tensor(train_data_all, dtype=torch.float)
                self.targets = torch.tensor(train_targets_all, dtype=torch.long)
            else:
                cross_val_ind = pd.read_csv(
                    os.path.join(self.root, val_file), sep=" ", header=None
                )
                cross_val_ind = cross_val_ind.values[:, 0:-1]
                fold_index = (self.fold_id - 1) * 2 + int(self.fold_val)
                k_fold_indices = cross_val_ind[
                    fold_index, np.invert(np.isnan(cross_val_ind[fold_index, :]))
                ].astype(int)
                self.data = torch.tensor(
                    train_data_all[k_fold_indices, :], dtype=torch.float
                )
                self.targets = torch.tensor(
                    train_targets_all[k_fold_indices], dtype=torch.long
                )

        else:
            raise ValueError(
                f"This dataset does not have a separate file for test data."
            )

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
                if member.name.startswith("abalone"):
                    file.extract(member, data_dir)
