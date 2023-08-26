import time
import csv
import sys
sys.path.append('/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena')
from methods import vanillaHD
from methods import adaptHD
from methods import onlineHD
from methods import adjustHD
from methods import compHD
from methods import multiCentroidHD
from methods import adaptHDIterative
from methods import onlineHDIterative
from methods import adjustHDIterative
from methods import quantHDIterative
from methods import sparseHDIterative
from methods import neuralHDIterative
from methods import distHDIterative


results_file = "results/results" + str(time.time()) + ".csv"

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "name",
            "accuracy",
            "train_time",
            "test_time",
            "dimensions",
            "method",
            "encoding",
            "iterations",
            "lr",
            "chunks",
            "threshold",
            "reduce_subclass",
            "model_quantize",
            "epsilon",
            "alpha",
            "beta",
            "theta",
            "r",
            "lazy_regeneration",
            "model_neural",
            "lazy_regeneration",
            "model_neural",
            "partial_data",
            "robustness_failed_dimensions",
        ]
    )

def select_model(train_loader,
                 test_loader,
                 num_classes,
                 num_feat,
                 encode,
                 model,
                 device,
                 dataset,
                 method="add",
                 encoding="density",
                 iterations=10,
                 dimensions=10000,
                 lr=1,
                 chunks=10,
                 threshold=0.03,
                 reduce_subclasses="drop",
                 model_quantize="binary",
                 lazy_regeneration=5,
                 model_neural="reset",
                 epsilon = 0.01,
                 alpha=4,
                 beta = 2,
                 theta = 1,
                 r = 0.05,
                 model_sparse="class",
                 s = 0.5,
                 partial_data=False,
                 robustness=[]
        ):
    if method == "add":
        vanillaHD.train_vanillaHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "adjust":
        adjustHD.train_adjustHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "adapt":
        adaptHD.train_adaptHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "online":
        onlineHD.train_onlineHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "comp":
        compHD.train_compHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "multicentroid":
        multiCentroidHD.train_multicentroidHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "adapt_iterative":
        adaptHDIterative.train_adaptHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "online_iterative":
        onlineHDIterative.train_onlineHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "adjust_iterative":
        adjustHDIterative.train_adjustHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "quant_iterative":
        quantHDIterative.train_quantHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "sparse_iterative":
        sparseHDIterative.train_sparseHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "neural_iterative":
        neuralHDIterative.train_neuralHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)
    elif method == "dist_iterative":
        distHDIterative.train_distHD(train_loader, test_loader, num_classes, encode, model, device, dataset.name, method, encoding,
                       iterations, dimensions, lr, chunks, threshold, reduce_subclasses, model_quantize, epsilon,
                       model_sparse, s, alpha, beta, theta, r, partial_data, robustness, lazy_regeneration, model_neural, results_file)

configs = [
{
    "method": "add",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "adapt",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "online",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "adjust",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "comp",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": 10,
    "s":None,
},
{
    "method": "multicentroid",
    "multi_reduce_subclass": "drop",
    "threshold": 0.03,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "adapt_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 5,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s": None,
},
{
    "method": "online_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 5,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "adjust_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 5,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "quant_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1.5,
    "epsilon": 0.01,
    "model_quantize": "binary",
    "model_sparse": None,
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s":None,
},
{
    "method": "sparse_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": 0.01,
    "model_quantize": None,
    "model_sparse": "class",
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": None,
    "alpha": None,
    "beta": None,
    "theta": None,
    "chunks": None,
    "s": 0.5,
},
{
    "method": "dist_iterative",
    "multi_reduce_subclass": None,
    "threshold": None,
    "lr": 1,
    "epsilon": None,
    "model_quantize": None,
    "model_sparse": "class",
    "sparsity": None,
    "lazy_regeneration": None,
    "model_neural": None,
    "r": 0.05,
    "alpha": 4,
    "beta": 2,
    "theta": 1,
    "chunks": None,
    "s": None,
},
]

# METHODS = ["add","adapt","online","adjust","comp","adapt_iterative","online_iterative","adjust_iterative",
# "quant_iterative","sparse_iterative","neural_iterative","dist_iterative","multicentroid","rvfl"]