import torch
from torchhd.datasets import HDCArena, UCIClassificationBenchmark
from torchhd.models import Centroid
import sys

sys.path.append("/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena")
from encoder import Encoder
from preprocess import preprocess
import methods_selection
from torchhd.models import BHDC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exec_arena(
    config=None,
    encoding="density",
    dimensions=10,
    repeats=1,
    batch_size=1,
    iterations=1,
    partial_data=False,
    robustness=[],
):
    for dataset in benchmark.datasets():
        for r in range(repeats):
            (
                train_loader,
                test_loader,
                num_classes,
                num_feat,
                train_ds,
                test_ds,
            ) = preprocess(dataset, batch_size, device, partial_data, config["method"])
            print(encoding)
            encode = Encoder(num_feat, dimensions, encoding, dataset.name)
            encode = encode.to(device)

            if encoding == "fractional":
                t = torch.complex64
            else:
                t = None

            if config["method"] == "lehdc":
                dropout_rate = 0.5
                model = BHDC(
                    in_features=dimensions,
                    out_features=num_classes,
                    dropout_prob=config["dropout_rate"],
                ).to(device)
            else:
                model = Centroid(
                    dimensions, num_classes, method=config["method"], dtype=t
                )
            model = model.to(device)

            methods_selection.select_model(
                train_ds,
                test_ds,
                train_loader,
                test_loader,
                num_classes,
                num_feat,
                encode,
                model,
                device,
                dataset,
                encoding=encoding,
                method=config["method"],
                dimensions=dimensions,
                iterations=iterations,
                lr=config["lr"],
                chunks=config["chunks"],
                threshold=config["threshold"],
                reduce_subclasses=config["multi_reduce_subclass"],
                model_quantize=config["model_quantize"],
                epsilon=config["epsilon"],
                alpha=config["alpha"],
                beta=config["beta"],
                theta=config["theta"],
                r=config["r"],
                s=config["s"],
                model_sparse=config["model_sparse"],
                lazy_regeneration=5,
                model_neural="reset",
                partial_data=partial_data,
                robustness=robustness,
                weight_decay=config["weight_decay"],
                learning_rate=config["learning_rate"],
                dropout_rate=config["dropout_rate"],
            )


# ENCODINGS = ["bundle", "sequence", "ngram", "hashmap", "flocet", "density", "random", "sinusoid","generic","fractional"]
ENCODINGS = ["flocet"]

configurations = methods_selection.configs

REPEATS = 1
DIMENSIONS = [10000]
ITERATIONS = 30
# PARTIAL_DATA = [0.2, 0.4, 0.6, 0.8, 1]
# , 0.6, 0.7, 0.8, 0.9, 1]
PARTIAL_DATA = [0.4, 0.5,0.6]
# PARTIAL_DATA = [1]
ROBUSTNESS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
# ROBUSTNESS = [0]
arena = False

if arena:
    benchmark = HDCArena("../data", download=True)

else:
    benchmark = UCIClassificationBenchmark("../data", download=True)

for i in DIMENSIONS:
    for j in ENCODINGS:
        for partial in PARTIAL_DATA:
            for k in configurations:
                exec_arena(
                    encoding=j,
                    dimensions=i,
                    repeats=REPEATS,
                    iterations=ITERATIONS,
                    config=k,
                    partial_data=partial,
                    robustness=ROBUSTNESS,
                )
