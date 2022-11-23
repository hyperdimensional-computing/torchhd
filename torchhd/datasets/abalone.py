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
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by fold and hyper_search variables.
            Otherwise returns a subset of train dataset if hypersearch is performed (hyper_search = True) if not (hyper_search = False) returns test set if it exists.
            If separate test set is not aviable returns a subset of training dataset as specified in ``conxuntos_kfold.dat`` if fold number is correc. 
            Otherwise issues an error.
        fold (int, optional): Specifies which fold number to use. The default value of -1 returns the whole train data from the corresponding file.
            Values between 0 and 3 specify, which fold in ``conxuntos_kfold.dat`` to use. Relevant only if train is set to True and fold>=0.
            Indices in even rows (zero indexing) of ``conxuntos_kfold.dat`` correspond to train subsets while indices in odd rows correspond to test subsets.
        hyper_search (bool, optional): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if train = True) 
            while the second row corresponds to test indices (used if train = False).             
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
    #Number of folds for cross validation
    num_folds = 4
    #Indicates whether the dataset has a separate training file
    has_test = False
    #Useful to specify name once to avoid changing it in several places 
    dataset = "abalone"

    def __init__(
        self,
        root: str,
        train: bool = True,
        fold: int = -1,
        hyper_search: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = os.path.join(root, self.dataset)
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.train = train
        self.fold = fold
        self.hyper_search = hyper_search
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
        has_train_file = os.path.isfile(os.path.join(self.root, self.dataset+"_R.dat"))
        has_k_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos_kfold.dat"))
        has_fold_file = os.path.isfile(os.path.join(self.root, "conxuntos.dat"))

        if has_train_file and has_k_fold_file and has_fold_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        # Files with pregenerated folds with indices
        k_fold_file = "conxuntos_kfold.dat"
        hyper_search_file = "conxuntos.dat"
        
        # Names of data files depending on whether test data comes separately
        if self.has_test: 
            train_data_file = self.dataset + "_train_R.dat"
            test_data_file = self.dataset + "_test_R.dat"
            # Load test data
            test_data = pd.read_csv(os.path.join(self.root, test_data_file), sep="\t", header=None, skiprows=1,)
            # Skip number & target columns 
            test_targets = test_data.values[:, -1].astype(int)
            test_data = test_data.values[:, 1:-1]
            
        else:
            train_data_file = self.dataset + "_R.dat"       

        # Load train data
        train_data = pd.read_csv(os.path.join(self.root, train_data_file), sep="\t", header=None, skiprows=1,)
        # Skip number & target columns 
        train_targets_all = train_data.values[:, -1].astype(int)
        train_data_all = train_data.values[:, 1:-1]
        
        # Load fold used in hyperparameter search if necessary
        if self.hyper_search:
                hyper_search_ind = pd.read_csv(os.path.join(self.root, hyper_search_file), sep=" ", header=None).values
        # Load k-fold if necessary
        if self.fold>=0:
                cross_val_ind = pd.read_csv(os.path.join(self.root, k_fold_file), sep=" ", header=None).values
        
        #train = True  - some form of train data is requested
        if self.train:
            # If hyperparameter search is being performed
            if self.hyper_search: 
                indices = hyper_search_ind[
                    0, np.invert(np.isnan(hyper_search_ind[0, :]))
                ].astype(int)
                self.data = torch.tensor(
                    train_data_all[indices, :], dtype=torch.float
                )
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )   
            # If regular training is going on    
            else:  
                # If not fold is specified
                if self.fold == -1:
                    self.data = torch.tensor(train_data_all, dtype=torch.float)
                    self.targets = torch.tensor(train_targets_all, dtype=torch.long)
                elif (self.fold>=0) and (self.fold<self.num_folds):
                    #Identify which row of cross_val_ind to index - even number
                    fold_index = self.fold  * 2 
                    indices = cross_val_ind[
                        fold_index, np.invert(np.isnan(cross_val_ind[fold_index, :]))
                    ].astype(int)
                    self.data = torch.tensor(
                        train_data_all[indices, :], dtype=torch.float
                    )
                    self.targets = torch.tensor(
                        train_targets_all[indices], dtype=torch.long
                    )                   
                #Wrong fold number
                else:
                    raise ValueError(
                        f"No such fold number. Please check that it is specified correctly"
                    )                       
        #train = False  - some form of test data is requested
        else:
            # If hyperparameter search is being performed
            if self.hyper_search: 
                indices = hyper_search_ind[
                    1, np.invert(np.isnan(hyper_search_ind[1, :]))
                ].astype(int)
                self.data = torch.tensor(
                    train_data_all[indices, :], dtype=torch.float
                )
                self.targets = torch.tensor(
                    train_targets_all[indices], dtype=torch.long
                )   
            # If regular testing is going on    
            else:   
                #If there is a dedicted file 
                if self.has_test:
                    self.data = torch.tensor(
                        test_data, dtype=torch.float
                    )
                    self.targets = torch.tensor(
                        test_targets, dtype=torch.long
                    )   
                #In case there is no dedicated use crossfold
                else:
                    #Check that fold number is within limits otherwise issue error
                    if (self.fold>=0) and (self.fold<self.num_folds):
                        #Identify which row of cross_val_ind to index - odd number
                        fold_index = self.fold  * 2 + 1
                        indices = cross_val_ind[
                            fold_index, np.invert(np.isnan(cross_val_ind[fold_index, :]))
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
                if member.name.startswith(self.dataset):
                    file.extract(member, data_dir)

