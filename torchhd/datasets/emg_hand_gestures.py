import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd
import math

from .utils import download_file_from_google_drive, unzip_file


class EMGHandGestures(data.Dataset):
    """EMG-based hand gestures dataset.

    Dataset from the paper `"Hyperdimensional Biosignal Processing: A Case Study for EMG-based Hand Gesture Recognition" <https://iis-people.ee.ethz.ch/~arahimi/papers/ICRC16.pdf>`_.

    Args:
        root (string): Root directory of dataset where files are stored.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        subjects (list[int], optional): The subject numbers from 0 til 4 to include. Defaults to [0, 1, 2, 3, 4].
        window (int, optional): The number of measurements to include in each sample. Defaults to 256.
    """

    classes: List[str] = [
        "Closed hand",
        "Open hand",
        "Two-finger pinch",
        "Point index",
        "Rest position",
    ]

    features_files = [
        "COMPLETE_1.csv",
        "COMPLETE_2.csv",
        "COMPLETE_3.csv",
        "COMPLETE_4.csv",
        "COMPLETE_5.csv",
    ]

    labels_files = [
        "LABEL_1.csv",
        "LABEL_2.csv",
        "LABEL_3.csv",
        "LABEL_4.csv",
        "LABEL_5.csv",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        subjects: list = [0, 1, 2, 3, 4],
        window: int = 256,
    ):
        root = os.path.join(root, "EMG_based_hand_gesture")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.transform = transform
        self.target_transform = target_transform
        self.subjects = subjects
        self.window = window
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

        has_not_feature_files = sum(
            list(
                map(
                    lambda x: not os.path.isfile(os.path.join(self.root, x)),
                    self.features_files,
                )
            )
        )
        has_not_label_files = sum(
            list(
                map(
                    lambda x: not os.path.isfile(os.path.join(self.root, x)),
                    self.labels_files,
                )
            )
        )
        # Check if the root directory contains the required files
        if not has_not_feature_files and not has_not_label_files:
            return True
        return False

    def _load_data(self):
        features = torch.empty(0, dtype=torch.long)
        labels = torch.empty(0, dtype=torch.long)
        for i in self.subjects:
            complete = pd.read_csv(
                os.path.join(self.root, self.features_files[i]), header=None
            )
            label = pd.read_csv(
                os.path.join(self.root, self.labels_files[i]), header=None
            )
            # List of indices where the gesture changes
            indexes = [
                index
                for index, _ in enumerate(label.values)
                if label.values[index] != label.values[index - 1]
            ]
            prev = 0
            labels_clean = torch.empty(0, dtype=torch.long)
            features_clean = torch.empty(0, self.window, 4, dtype=torch.long)
            # Every change of gesture we group it values
            for j in indexes:
                span = j - prev
                # If we have that the amount of data of the gesture fits in the window we have a new sample of the gesture of size window
                if span > self.window:
                    for k in range(math.floor(span / self.window)):
                        # Clean the label data
                        label_clean = (
                            torch.tensor(
                                label.values[prev + (self.window * k)], dtype=torch.long
                            )
                            - 1
                        )
                        # Clean the feature data
                        feature_clean = torch.tensor(
                            complete.values[
                                prev
                                + (self.window * k) : prev
                                + (self.window * (k + 1))
                            ],
                            dtype=torch.long,
                        )[None, :, :]
                        labels_clean = torch.cat((labels_clean, label_clean))
                        features_clean = torch.cat((features_clean, feature_clean))
                prev = j
            features = torch.cat((features, features_clean))
            labels = torch.cat((labels, labels_clean))
        self.data = features
        self.targets = labels

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file_from_google_drive(
            "1_9DYP1MwICtnMRm_F34xVvv1S2OhVnHH",  # Google Drive shared file ID
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)
