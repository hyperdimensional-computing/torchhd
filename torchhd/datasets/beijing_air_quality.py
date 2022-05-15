import os
import os.path
from typing import Callable, Optional, NamedTuple, List
import torch
from torch.utils import data
import pandas as pd

from .utils import download_file, unzip_file


class BeijingAirQualityDataSample(NamedTuple):
    categorical: torch.LongTensor
    continuous: torch.FloatTensor


class BeijingAirQuality(data.Dataset):
    """`Beijing Multi-Site Air-Quality <https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data>`_ dataset.

    .. warning::
        The data contains NaN values that need to be taken into account.

    Args:
        root (string): Root directory of dataset where directory
            ``beijing-air-quality`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in a feature Tensor
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    wind_directions: List[str]
    stations: List[str]
    categorical_columns: List[str] = [
        "year",
        "month",
        "day",
        "hour",
        "wind_direction",
        "station",
    ]
    continuous_columns: List[str] = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "temperature",
        "pressure",
        "dew_point_temperature",
        "precipitation",
        "wind_speed",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, "beijing-air-quality")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.transform = transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self._load_data()

    def __len__(self) -> int:
        return self.categorical_data.size(0)

    def __getitem__(self, index: int) -> BeijingAirQualityDataSample:
        """
        Args:
            index (int): Index

        Returns:
            BeijingAirQualityDataSample: Indexed Sample
        """
        sample = BeijingAirQualityDataSample(
            self.categorical_data[index], self.continuous_data[index]
        )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the expected number of csv files
        files = os.listdir(self.root)
        files = [file for file in files if file.endswith(".csv")]
        if len(files) == 12:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):

        files = os.listdir(self.root)
        csv_files = [file for file in files if file.endswith(".csv")]

        data_tables = []
        for filename in csv_files:
            data = pd.read_csv(os.path.join(self.root, filename))
            # No column does not provide any meaningful information
            data = data.drop(columns=["No"])
            data_tables.append(data)

        data = pd.concat(data_tables, ignore_index=True)
        # Rename columns for accessability
        data = data.rename(
            columns={
                "wd": "wind_direction",
                "TEMP": "temperature",
                "PRES": "pressure",
                "DEWP": "dew_point_temperature",
                "RAIN": "precipitation",
                "WSPM": "wind_speed",
            }
        )

        # Change null values to Not Available string "N/A"
        data.loc[data.wind_direction.isnull(), "wind_direction"] = "N/A"
        # Map the wind directions to a category identifier (int)
        # # Save the mapping for user referencing
        self.wind_directions = tuple(sorted(list(set(data.wind_direction))))
        data.loc[:, "wind_direction"] = data.wind_direction.apply(
            lambda x: self.wind_directions.index(x)
        )

        # Map the stations to a category identifier (int)
        # Save the mapping for user referencing
        self.stations = tuple(sorted(list(set(data.station))))
        data.loc[:, "station"] = data.station.apply(lambda x: self.stations.index(x))

        categorical = data[self.categorical_columns]
        self.categorical_data = torch.tensor(categorical.values, dtype=torch.long)

        continuous = data[self.continuous_columns]
        self.continuous_data = torch.tensor(continuous.values, dtype=torch.float)

    def download(self):
        """Download the data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        zip_file_path = os.path.join(self.root, "data.zip")
        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip",
            zip_file_path,
        )

        unzip_file(zip_file_path, self.root)
        os.remove(zip_file_path)

        source_dir = os.path.join(self.root, "PRSA_Data_20130301-20170228")
        data_files = os.listdir(source_dir)
        for filename in data_files:
            os.rename(
                os.path.join(source_dir, filename), os.path.join(self.root, filename)
            )

        os.rmdir(source_dir)
