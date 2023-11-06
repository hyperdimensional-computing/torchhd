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
import csv

import time
csv_file = 'experiment_4/result'+str(time.time())+'.csv'

def experiment(randomness=0, dataset="MUTAG"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    DIMENSIONS = 10000  # hypervectors dimension

    # for other available datasets see: https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html?highlight=tudatasets
    # dataset = "MUTAG"

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
            self.node_ids = embeddings.Random(size, out_features)
            self.levels = embeddings.Circular(size, out_features)

        def local_centrality2(self, x):
            nodes, _ = x.edge_index
            nodes = list(set(nodes))
            node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)
            for i in nodes:
                adjacent_nodes = x.edge_index[1][x.edge_index[0] == i]
                node_id_hvs[i] = self.node_ids.weight[i]
                for j in adjacent_nodes:
                    node_id_hvs[i] += torchhd.permute(self.node_ids.weight[j])

            row, col = to_undirected(x.edge_index)
            hvs = torchhd.bind(node_id_hvs[row], node_id_hvs[col])
            return torchhd.multiset(hvs)

        def local_centrality(self, x):
            nodes, _ = x.edge_index
            nodes = list(set(nodes))
            node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)

            for i in nodes:
                adjacent_nodes = x.edge_index[1][x.edge_index[0] == i]
                node_id_hvs[i] = torchhd.bind(self.node_ids.weight[i], self.levels.weight[len(adjacent_nodes)])

            row, col = to_undirected(x.edge_index)
            hvs = torchhd.bind(node_id_hvs[row], node_id_hvs[col])
            return torchhd.multiset(hvs)

        def semi_local_centrality(self, x):
            nodes, _ = x.edge_index
            nodes = list(set(nodes))
            node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)

            for i in nodes:
                adjacent_nodes = x.edge_index[1][x.edge_index[0] == i]
                for j in adjacent_nodes:
                    node_id_hvs[i] = torchhd.bundle(self.levels.weight[len(x.edge_index[1][x.edge_index[0] == j])], node_id_hvs[i])
                node_id_hvs[i] = torchhd.bind(node_id_hvs[i], (self.node_ids.weight[i]))

            row, col = to_undirected(x.edge_index)
            hvs = torchhd.bundle(node_id_hvs[row], node_id_hvs[col])
            return torchhd.multiset(hvs)

        def forward(self, x):
            return self.local_centrality(x)
            '''
            nodes, _ = x.edge_index
            nodes = list(set(nodes))
            node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)
            for i in nodes:
                adjacent_nodes = x.edge_index[1][x.edge_index[0] == i]
                node_id_hvs[i] = self.node_ids.weight[i]
                for j in adjacent_nodes:
                    node_id_hvs[i] += torchhd.permute(self.node_ids.weight[j])

            node_id_hvs_2 = torch.zeros((x.num_nodes, self.out_features), device=device)

            for i in nodes:
                adjacent_nodes = x.edge_index[1][x.edge_index[0] == i]
                node_id_hvs_2[i] = node_id_hvs[i]
                for j in adjacent_nodes:
                    node_id_hvs_2[i] += torchhd.permute(node_id_hvs[j])

            row, col = to_undirected(x.edge_index)
            hvs = torchhd.bind(node_id_hvs_2[row], node_id_hvs_2[col])
            return torchhd.multiset(hvs)
            '''

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
    #f1 = torchmetrics.F1Score(num_classes=graphs.num_classes, average='macro', multiclass=True)
    f1 = torchmetrics.F1Score("multiclass", num_classes=graphs.num_classes)

    with torch.no_grad():
        model.normalize()

        for samples in tqdm(test_ld, desc="Testing"):
            samples.edge_index = samples.edge_index.to(device)

            samples_hv = encode(samples).unsqueeze(0)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), samples.y)
            f1.update(outputs.cpu(), samples.y)

    acc = (accuracy.compute().item() * 100)
    f = (f1.compute().item() * 100)
    return acc, f



REPETITIONS = 1
DATASET = ['MUTAG']

for d in DATASET:
    acc_final = []
    f1_final = []
    acc_aux = []
    f1_aux = []
    for j in range(REPETITIONS):
        acc, f1 = experiment(100, d)
        acc_aux.append(acc)
        f1_aux.append(f1)
    acc_final.append(round(sum(acc_aux)/REPETITIONS, 2))
    f1_final.append(round(sum(f1_aux)/REPETITIONS,2))

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([d] +['RANDOM'])
        writer.writerows([acc_final])
        writer.writerows([f1_final])


