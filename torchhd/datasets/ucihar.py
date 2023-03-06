#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import os.path as path
import shutil
from typing import Callable, Optional, List
import torch
from torch.utils import data
import numpy as np
import pandas as pd

from .utils import download_file, unzip_file


class UCIHAR(data.Dataset):
    """`UCI Human Activity Recognition <https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>`_ dataset.
    As found in the paper `"Human Activity Recognition Using Smartphones" <https://ieeexplore.ieee.org/document/8567275>`_.

    .. list-table::
       :widths: 10 10 10 10
       :align: center
       :header-rows: 1

       * - Instances
         - Attributes
         - Task
         - Area
       * - 10299
         - 561
         - Classification
         - N/A

    Args:
        root (string): Root directory of dataset where the training and testing samples are located.
        train (bool, optional): If True, creates dataset from UCIHAR-training data,
            otherwise from UCIHAR-testing data
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.LongTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.


    """

    classes: List[str] = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = path.join(root, "ucihar")
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
        return self.targets.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: (sample, target) where target is the index of the target class
        """
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        train_dir = os.path.join(self.root, "train")
        has_train_dir = os.path.isdir(train_dir)
        test_dir = os.path.join(self.root, "test")
        has_test_dir = os.path.isdir(test_dir)

        if (not has_train_dir) or (not has_test_dir):
            return False

        has_train_x = os.path.isfile(os.path.join(train_dir, "X_train.txt"))
        has_train_y = os.path.isfile(os.path.join(train_dir, "y_train.txt"))

        if (not has_train_x) or (not has_train_y):
            return False

        has_test_x = os.path.isfile(os.path.join(test_dir, "X_test.txt"))
        has_test_y = os.path.isfile(os.path.join(test_dir, "y_test.txt"))

        if (not has_test_x) or (not has_test_y):
            return False

        return True

    def _load_data(self):
        data_dir = os.path.join(self.root, "train" if self.train else "test")
        data_file = "X_train.txt" if self.train else "X_test.txt"
        target_file = "y_train.txt" if self.train else "y_test.txt"

        data = np.loadtxt(os.path.join(data_dir, data_file), dtype="float32")
        targets = np.loadtxt(path.join(data_dir, target_file), dtype="int64")

        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets) - 1

    def download(self):
        """Downloads the dataset if it doesn't exist already"""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)

        source_dir = os.path.join(self.root, "UCI HAR Dataset")
        data_files = os.listdir(source_dir)
        for filename in data_files:
            src = os.path.join(source_dir, filename)
            dest = os.path.join(self.root, filename)
            os.rename(src, dest)

        os.rmdir(source_dir)
