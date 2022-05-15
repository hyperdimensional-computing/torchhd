import os
import os.path
import copy
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd

from .utils import download_file, unzip_file


class PAMAP(data.Dataset):
    """`PAMAP <https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring>`_ dataset.

    Args:
        root (string): Root directory of dataset.
        subjects (list): List of subjects to be loaded in dataset
        optional (bool): If true optional data of some subjectes will be loaded.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes: List[str] = [
        "lying",
        "sitting",
        "standing",
        "walking",
        "running",
        "cycling",
        "nordic walking",
        "watching TV",
        "computer work",
        "car driving",
        "ascending stairs",
        "descending stairs",
        "vacuum cleaning",
        "ironing",
        "folding laundry",
        "house cleaning",
        "playing soccer",
        "rope jumping",
    ]

    columns: List[str] = [
        "timestamp",
        "activity",
        "heartRate",
        "handTemp",
        "handAcc11",
        "handAcc12",
        "handAcc13",
        "handAcc21",
        "handAcc22",
        "handAcc23",
        "handGyro1",
        "handGyro2",
        "handGyro3",
        "handMagnetometer1",
        "handMagnetometer2",
        "handMagnetometer3",
        "handOrientation1",
        "handOrientation2",
        "handOrientation3",
        "handOrientation4",
        "chestTemp",
        "chestAcc11",
        "chestAcc12",
        "chestAcc13",
        "chestAcc21",
        "chestAcc22",
        "chestAcc23",
        "chestGyro1",
        "chestGyro2",
        "chestGyro3",
        "chestMagnetometer1",
        "chestMagnetometer2",
        "chestMagnetometer3",
        "chestOrientation1",
        "chestOrientation2",
        "chestOrientation3",
        "chestOrientation4",
        "ankleTemp",
        "ankleAcc11",
        "ankleAcc12",
        "ankleAcc13",
        "ankleAcc21",
        "ankleAcc22",
        "ankleAcc23",
        "ankleGyro1",
        "ankleGyro2",
        "ankleGyro3",
        "ankleMagnetometer1",
        "ankleMagnetometer2",
        "ankleMagnetometer3",
        "ankleOrientation1",
        "ankleOrientation2",
        "ankleOrientation3",
        "ankleOrientation4",
    ]

    subjects_with_optional_data: List[int] = [0, 4, 5, 7, 8]

    def __init__(
        self,
        root: str,
        subjects: list = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        optional: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, "pamap")
        root = os.path.expanduser(root)
        self.root = root
        self.subjects = subjects
        self.optional = optional
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
        has_all_files = []
        for i in [1, 5, 6, 8, 9]:
            has_all_files.append(
                os.path.isfile(
                    os.path.join(
                        self.root, "PAMAP2_Dataset/Optional/subject10" + str(i) + ".dat"
                    )
                )
            )
        for i in range(1, 10):
            has_all_files.append(
                os.path.isfile(
                    os.path.join(
                        self.root, "PAMAP2_Dataset/Protocol/subject10" + str(i) + ".dat"
                    )
                )
            )

        if all(has_all_files):
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        clean_labels = torch.empty(0, dtype=torch.long)
        clean_features = torch.empty(0, dtype=torch.long)
        for i in self.subjects:
            data = pd.read_csv(
                os.path.join(
                    self.root, "PAMAP2_Dataset/Protocol/subject10" + str(i + 1) + ".dat"
                ),
                delimiter=" ",
                header=None,
            )
            # Adding optional data if requested and exists
            if self.optional and i in self.subjects_with_optional_data:
                optional_data = pd.read_csv(
                    os.path.join(
                        self.root,
                        "PAMAP2_Dataset/Optional/subject10" + str(i + 1) + ".dat",
                    ),
                    delimiter=" ",
                    header=None,
                )
                data = pd.concat([data, optional_data])
            # Activity with value 0 should be discarded in any kind of analysis
            data = data[data[1] != 0]
            cols = copy.copy(self.columns)
            data.columns = cols
            cols.remove("heartRate")
            # Drop Nan values that are not heartRate
            data = data.dropna(subset=cols)
            # Replace Nan values of heart rate for value before
            data.ffill(inplace=True)
            data = data.dropna()
            data = data.reset_index(drop=True)
            activities = data["activity"]
            # Replace data activity lables
            activities.replace(
                [9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24],
                [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                inplace=True,
            )
            labels = torch.tensor(activities.values, dtype=torch.long)
            del data["activity"]
            features = torch.tensor(data.values, dtype=torch.long)
            clean_labels = torch.cat((clean_labels, labels))
            clean_features = torch.cat((clean_features, features))

        self.data = clean_features
        self.targets = clean_labels

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)
