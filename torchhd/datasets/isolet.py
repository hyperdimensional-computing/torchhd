import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd

from .utils import download_file_from_google_drive, unzip_file


class ISOLET(data.Dataset):
    """`ISOLET <https://archive.ics.uci.edu/ml/datasets/isolet>`_ dataset.

    Args:
        root (string): Root directory of dataset where ``isolet1+2+3+4.data``
            and  ``isolet5.data`` exist.
        train (bool, optional): If True, creates dataset from ``isolet1+2+3+4.data``,
            otherwise from ``isolet5.data``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes: List[str] = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, "isolet")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.train = train
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
        has_train_file = os.path.isfile(os.path.join(self.root, "isolet1+2+3+4.data"))
        has_test_file = os.path.isfile(os.path.join(self.root, "isolet5.data"))
        if has_train_file and has_test_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        data_file = "isolet1+2+3+4.data" if self.train else "isolet5.data"
        data = pd.read_csv(os.path.join(self.root, data_file), header=None)
        self.data = torch.tensor(data.values[:, :-1], dtype=torch.float)
        self.targets = torch.tensor(data.values[:, -1], dtype=torch.long) - 1

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file_from_google_drive(
            "1IMC6xzs2kBnf5_kaiBUzSWiTMR_dFbIX",  # Google Drive shared file ID
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)
