import os
import os.path as path
from typing import Callable, Optional, Tuple
import torch
from torch.utils import data
import pandas as pd

from .utils import download_file, unzip_file


class CyclePowerPlant(data.Dataset):
    """`Combined cycle power planet <https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant>`_ dataset.
        Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.

    Args:
        root (string): Root directory of dataset where downloaded dataset exists
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        root = path.join(root, "ccpp")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

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

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: (sample, target) where target is the index of the target class
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

        # Check if root directory contains the required data file
        has_data_file = os.path.isfile(os.path.join(self.root, "Folds5x2_pp.xlsx"))
        if has_data_file:
            return True

        return False

    def _load_data(self):
        file_name = "Folds5x2_pp.xlsx"
        data = pd.read_excel(os.path.join(self.root, file_name))
        self.data = torch.tensor(data.values[:, :-1], dtype=torch.float)
        self.targets = torch.tensor(data.values[:, -1], dtype=torch.float)

    def download(self):
        """Downloads the dataset if not already present"""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)

        source_dir = os.path.join(self.root, "CCPP")
        data_files = os.listdir(source_dir)
        for filename in data_files:
            os.rename(
                os.path.join(source_dir, filename), os.path.join(self.root, filename)
            )

        os.rmdir(source_dir)
