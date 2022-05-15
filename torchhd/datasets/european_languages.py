import os
from typing import Callable, Optional, Tuple, List
import torch
import torch.utils.data as data

from .utils import download_file_from_google_drive, unzip_file


class EuropeanLanguages(data.Dataset):
    """European Languages dataset.

    As used in the paper `"A Robust and Energy-Efficient Classifier Using
    Brain-Inspired Hyperdimensional Computing" <https://iis-people.ee.ethz.ch/~arahimi/papers/ISLPED16.pdf>`_.
    The dataset contains sentences in 21 European languages,
    the training data was taken from `Wortschatz Corpora <https://wortschatz.uni-leipzig.de/en/download>`_
    and the testing data from `Europarl Parallel Corpus <https://www.statmt.org/europarl/>`_.

    Args:
        root (string): Root directory of dataset where the training and testing samples are located.
        train (bool, optional): If True, creates dataset from Wortschatz Corpora,
            otherwise from Europarl Parallel Corpus.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.LongTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes: List[str] = [
        "Bulgarian",
        "Czech",
        "Danish",
        "Dutch",
        "German",
        "English",
        "Estonian",
        "Finnish",
        "French",
        "Greek",
        "Hungarian",
        "Italian",
        "Latvian",
        "Lithuanian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Slovak",
        "Slovenian",
        "Spanish",
        "Swedish",
    ]

    files: List[str] = [
        "bul.txt",
        "ces.txt",
        "dan.txt",
        "nld.txt",
        "deu.txt",
        "eng.txt",
        "est.txt",
        "fin.txt",
        "fra.txt",
        "ell.txt",
        "hun.txt",
        "ita.txt",
        "lav.txt",
        "lit.txt",
        "pol.txt",
        "por.txt",
        "ron.txt",
        "slk.txt",
        "slv.txt",
        "spa.txt",
        "swe.txt",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, "language-recognition")
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

    def __getitem__(self, index) -> Tuple[str, torch.LongTensor]:
        """
        Args:
            index (int): Index

        Returns:
            Tuple[str, torch.LongTensor]: (sample, target) where target is the index of the target class
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

        train_dir = os.path.join(self.root, "training")
        has_train_dir = os.path.isdir(train_dir)
        test_dir = os.path.join(self.root, "testing")
        has_test_dir = os.path.isdir(test_dir)
        if not has_train_dir or not has_test_dir:
            return False

        for file in self.files:
            has_train_file = os.path.isfile(os.path.join(train_dir, file))
            if not has_train_file:
                return False

            has_test_file = os.path.isfile(os.path.join(test_dir, file))
            if not has_test_file:
                return False

        return True

    def _load_data(self):
        data_dir = os.path.join(self.root, "training" if self.train else "testing")

        data = []
        targets = []

        for class_label, filename in enumerate(self.files):
            with open(os.path.join(data_dir, filename), "r") as file:
                lines = file.readlines()
                lines = map(self._clean_line, lines)
                lines = filter(self._filter_line, lines)
                lines = list(lines)

                data += lines
                targets += [class_label] * len(lines)

        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.long)

    def _clean_line(self, line):
        line = line.strip()  # remove space at start and end
        line = " ".join(line.split())  # compact any whitespace to a single space
        return line

    def _filter_line(self, line):
        return line != ""

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file_from_google_drive(
            "1zCvjPf0R5pOR46CNBNMM60b_LwQKvltI", zip_file_path
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)
