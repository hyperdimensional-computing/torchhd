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
from typing import List
from torchhd.datasets import DatasetFourFold


class IlpdIndianLiver(DatasetFourFold):
    """`ILPD (Indian Liver Patient Dataset) <https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)>`_ dataset.

    .. list-table::
       :widths: 10 10 10 10
       :align: center
       :header-rows: 1

       * - Instances
         - Attributes
         - Task
         - Area
       * - 583
         - 10
         - Classification
         - Life

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by fold and hyper_search variables.
            Otherwise returns a subset of train dataset if hypersearch is performed (``hyper_search = True``) if not (``hyper_search = False``) returns a subset of training dataset
            as specified in ``conxuntos_kfold.dat`` if fold number is correct. Otherwise issues an error.
        fold (int, optional): Specifies which fold number to use. The default value of -1 returns all the training data from the corresponding file.
            Values between 0 and 3 specify, which fold in ``conxuntos_kfold.dat`` to use. Relevant only if hyper_search is set to False and ``0 <= fold <= 3``.
            Indices in even rows (zero indexing) of ``conxuntos_kfold.dat`` correspond to train subsets while indices in odd rows correspond to test subsets.
        hyper_search (bool, optional): If True, creates dataset using indeces in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if ``train = True``)
            while the second row corresponds to test indices (used if ``train = False``).
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.


    """

    name = "ilpd-indian-liver"
    classes: List[str] = [
        "liver patient",
        "not liver patient",
    ]
