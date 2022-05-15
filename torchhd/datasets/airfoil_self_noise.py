import os
import os.path
from typing import Callable, Optional, Tuple
import torch
from torch.utils import data
import pandas as pd

from .utils import download_file


class AirfoilSelfNoise(data.Dataset):
    """`NASA Airfoil Self-Noise <https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise>`_ dataset.
    Dataset is obtained from a series of aerodynamic and acoustic tests of two and three-dimensional
    airfoil blade sections conducted in an anechoic wind tunnel.


    Args:
        root (string): Root directory of dataset where ``airfoil_self_noise.dat`` exists
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
        root = os.path.join(root, "airfoil_self_noise")
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
        has_data_file = os.path.isfile(
            os.path.join(self.root, "airfoil_self_noise.dat")
        )
        if has_data_file:
            return True

        return False

    def _load_data(self):
        file_name = "airfoil_self_noise.dat"
        data = pd.read_csv(
            os.path.join(self.root, file_name), delim_whitespace=True, header=None
        )
        self.data = torch.tensor(data.values[:, :-1], dtype=torch.float)
        self.targets = torch.tensor(data.values[:, -1], dtype=torch.float)

    def download(self):
        """Download dataset if does not already exist"""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "airfoil_self_noise.dat")

        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
            zip_file_path,
        )
