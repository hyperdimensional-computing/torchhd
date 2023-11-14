import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Note: this example requires the torch_geometric library: https://pytorch-geometric.readthedocs.io
from torch_geometric.datasets import TUDataset

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid


def experiment(randomness=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    DIMENSIONS = 10000  # hypervectors dimension

    # for other available datasets see: https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html?highlight=tudatasets
    dataset = "MUTAG"

    graphs = TUDataset("../data", dataset)
    train_size = int(0.7 * len(graphs))
    test_size = len(graphs) - train_size
    train_ld, test_ld = torch.utils.data.random_split(graphs, [train_size, test_size])

    def sparse_stochastic_graph(G):
        """
        Returns a sparse adjacency matrix of the graph G.
        The values indicate the probability of leaving a vertex.
        This means that each column sums up to one.
        """
        rows, columns = G.edge_index
        # Calculate the probability for each column
        values_per_column = 1.0 / torch.bincount(columns, minlength=G.num_nodes)
        values_per_node = values_per_column[columns]
        size = (G.num_nodes, G.num_nodes)
        return torch.sparse_coo_tensor(G.edge_index, values_per_node, size)

    def pagerank(G, alpha=0.85, max_iter=100, tol=1e-06):
        N = G.num_nodes
        M = sparse_stochastic_graph(G) * alpha
        v = torch.zeros(N, device=G.edge_index.device) + 1 / N
        p = torch.zeros(N, device=G.edge_index.device) + 1 / N
        for _ in range(max_iter):
            v_prev = v
            v = M @ v + p * (1 - alpha)

            err = (v - v_prev).abs().sum()
            if tol != None and err < N * tol:
                return v
        return v

    def to_undirected(edge_index):
        """
        Returns the undirected edge_index
        [[0, 1], [1, 0]] will result in [[0], [1]]
        """
        edge_index = edge_index.sort(dim=0)[0]
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index

    def min_max_graph_size(graph_dataset):
        if len(graph_dataset) == 0:
            return None, None

        max_num_nodes = float("-inf")
        min_num_nodes = float("inf")

        for G in graph_dataset:
            num_nodes = G.num_nodes
            max_num_nodes = max(max_num_nodes, num_nodes)
            min_num_nodes = min(min_num_nodes, num_nodes)

        return min_num_nodes, max_num_nodes

    class Encoder(nn.Module):
        def __init__(self, out_features, size):
            super(Encoder, self).__init__()
            self.out_features = out_features
            self.node_ids = embeddings.Level(size, out_features, randomness=0.05)

        def forward(self, x):
            pr = pagerank(x)
            pr_sort, pr_argsort = pr.sort()

            node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)
            node_id_hvs[pr_argsort] = self.node_ids.weight[: x.num_nodes]

            row, col = to_undirected(x.edge_index)

            hvs = torchhd.bind(node_id_hvs[row], node_id_hvs[col])
            return torchhd.multiset(hvs)

    min_graph_size, max_graph_size = min_max_graph_size(graphs)
    encode = Encoder(DIMENSIONS, max_graph_size)
    encode = encode.to(device)

    model = Centroid(DIMENSIONS, graphs.num_classes)
    model = model.to(device)

    with torch.no_grad():
        for samples in tqdm(train_ld, desc="Training"):
            samples.edge_index = samples.edge_index.to(device)
            samples.y = samples.y.to(device)

            samples_hv = encode(samples).unsqueeze(0)
            model.add(samples_hv, samples.y)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=graphs.num_classes)
    f1 = torchmetrics.F1Score("multiclass", num_classes=graphs.num_classes)
    auc = torchmetrics.AUROC("multiclass", num_classes=graphs.num_classes)

    with torch.no_grad():
        model.normalize()

        for samples in tqdm(test_ld, desc="Testing"):
            samples.edge_index = samples.edge_index.to(device)

            samples_hv = encode(samples).unsqueeze(0)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), samples.y)
            f1.update(outputs.cpu(), samples.y)
            auc.update(outputs.cpu(), samples.y)

    acc = accuracy.compute().item() * 100
    f = f1.compute().item() * 100
    au = auc.compute().item() * 100
    print(f"Testing accuracy of {acc:.3f}%")
    print(f"Testing f1 of {f:.3f}%")
    print(f"Testing AUC of {au:.3f}%")
    return acc, f, au


REPETITIONS = 10
RANDOMNESS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]


acc_final = []
f1_final = []
auc_final = []

for i in RANDOMNESS:
    acc_aux = []
    f1_aux = []
    auc_aux = []
    for j in range(REPETITIONS):
        acc, f1, auc = experiment(i)
        acc_aux.append(acc)
        f1_aux.append(f1)
        auc_aux.append(auc)
    acc_final.append(round(sum(acc_aux) / REPETITIONS, 2))
    f1_final.append(round(sum(f1_aux) / REPETITIONS, 2))
    auc_final.append(round(sum(auc_aux) / REPETITIONS, 2))

print(acc_final)
print(f1_final)
print(auc_final)
