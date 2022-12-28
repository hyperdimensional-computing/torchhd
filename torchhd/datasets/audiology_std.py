from typing import List
from torchhd.datasets import DatasetTrainTest


class AudiologyStd(DatasetTrainTest):
    """`Audiology (Standardized) <https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29>`_ dataset.

    Args:
        root (string): Root directory containing the files of the dataset.
        train (bool, optional): If True, returns training (sub)set from the file storing training data as further determined by hyper_search variable.
            Otherwise returns a subset of train dataset if hyperparameter search is performed (``hyper_search = True``) if not (``hyper_search = False``) returns test set.
        hyper_search (bool, optional): If True, creates dataset using indices in ``conxuntos.dat``. This split is used for hyperparameter search. The first row corresponds to train indices (used if ``train = True``)
            while the second row corresponds to test indices (used if ``train = False``).
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    name = "audiology-std"
    classes: List[str] = [
        "cochlear_age",
        "cochlear_age_and_noise",
        "cochlear_noise_and_heredity",
        "cochlear_poss_noise",
        "cochlear_unknown",
        "conductive_discontinuity",
        "conductive_fixation",
        "mixed_cochlear_age_otitis_media",
        "mixed_cochlear_age_s_om",
        "mixed_cochlear_unk_discontinuity",
        "mixed_cochlear_unk_fixation",
        "mixed_cochlear_unk_ser_om",
        "mixed_poss_noise_om",
        "normal_ear",
        "otitis_media",
        "possible_brainstem_disorder",
        "possible_menieres",
        "retrocochlear_unknown",
    ]
