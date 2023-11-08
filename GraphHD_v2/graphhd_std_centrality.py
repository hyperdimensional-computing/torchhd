import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader
from torch_geometric.utils.degree import degree
import networkx as nx

# Note: this example requires the torch_geometric library: https://pytorch-geometric.readthedocs.io
from torch_geometric.datasets import TUDataset

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import csv
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import time

csv_file = "metrics/result" + str(time.time()) + ".csv"
DIM = 10000


def experiment(randomness=0, embed="random", dataset="MUTAG", metric="page_rank"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    DIMENSIONS = DIM  # hypervectors dimension

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

    def centrality(data):
        degree_centrality = data.edge_index[0].bincount(minlength=data.num_nodes)
        degree_ranked_nodes = sorted(range(data.num_nodes), key=lambda node: degree_centrality[node])

    def semi_local_centrality(data):
        G = nx.Graph()

        for i in range(data.edge_index.size(1)):
            edge = tuple(data.edge_index[:, i].tolist())
            G.add_edge(*edge)

        # Calculate semi-local centrality using a custom approach
        semi_local_centrality = []

        for node in G.nodes():
            ego_graph = nx.ego_graph(G, node, radius=2)  # Adjust the radius (2 in this case)
            semi_local_centrality.append(len(ego_graph))

        # Store the semi-local centrality scores in the PyTorch Geometric Data object
        data.semi_local_centrality = torch.tensor(semi_local_centrality)

        # Rank nodes based on semi-local centrality
        semi_local_ranked_nodes = sorted(G.nodes(), key=lambda node: semi_local_centrality[node])
        return semi_local_ranked_nodes

    def degree_centrality(data):
        G = to_networkx(data)

        scores = nx.degree_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def eigen_centrality(data):
        G = to_networkx(data)

        scores = nx.eigenvector_centrality(G, max_iter=1000)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def katz_centrality(data):
        G = to_networkx(data)

        beta = 0.1
        scores = nx.katz_centrality(G, beta=beta)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def closeness_centrality(data):
        G = to_networkx(data)

        scores = nx.closeness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def incremental_closeness_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()
        G.add_edges_from(data.edge_index.t().tolist())

        scores = nx.incremental_closeness_centrality(G, G.edges())
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def current_flow_closeness_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()
        scores = nx.current_flow_closeness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def information_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()
        scores = nx.information_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def betweenness_centrality(data):
        G = to_networkx(data)

        scores = nx.betweenness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def edge_betweenness_centrality(data):
        G = to_networkx(data)

        scores = nx.edge_betweenness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def current_flow_betweeness_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.current_flow_closeness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes


    def edge_current_flow_betweeness_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.edge_current_flow_betweenness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def communicability_betweeness_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.communicability_betweenness_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def load_centrality(data):
        G = to_networkx(data)

        scores = nx.load_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def edge_load_centrality(data):
        G = to_networkx(data)

        scores = nx.edge_load_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def subgraph_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.subgraph_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def subgraph_centrality_exp(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.subgraph_centrality_exp(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def estrada_index(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.estrada_index(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def harmonic_centrality(data):
        G = to_networkx(data)

        scores = nx.harmonic_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def dispersion(data):
        G = to_networkx(data)

        scores = nx.dispersion(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def global_reaching_centrality(data):
        G = to_networkx(data)

        scores = nx.global_reaching_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def percolation_centrality(data):
        G = to_networkx(data)

        scores = nx.percolation_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def second_order_centrality(data):
        G = to_networkx(data)
        G = G.to_undirected()

        scores = nx.second_order_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def trophic_levels(data):
        G = to_networkx(data)

        scores = nx.trophic_levels(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def trophic_differences(data):
        G = to_networkx(data)

        scores = nx.trophic_differences(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def trophic_incoherence_parameter(data):
        G = to_networkx(data)

        scores = nx.trophic_incoherence_parameter(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def voterank(data):
        G = to_networkx(data)

        scores = nx.voterank(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

    def laplacian_centrality(data):
        G = to_networkx(data)

        scores = nx.laplacian_centrality(G)
        scores_nodes = sorted(G.nodes(), key=lambda node: scores[node])

        return scores_nodes

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
        def __init__(self, out_features, size, metric):
            super(Encoder, self).__init__()
            self.out_features = out_features
            self.metric = metric
            if embed == "thermometer":
                self.node_ids = embeddings.Thermometer(size, out_features, vsa=VSA)
            elif embed == "circular":
                self.node_ids = embeddings.Circular(size, out_features, vsa=VSA)
            elif embed == "projection":
                self.node_ids = embeddings.Projection(size, out_features, vsa=VSA)
            elif embed == "sinusoid":
                self.node_ids = embeddings.Sinusoid(size, out_features, vsa=VSA)
            elif embed == "density":
                self.node_ids = embeddings.Density(size, out_features, vsa=VSA)
            else:
                self.node_ids = embeddings.Level(size, out_features, vsa=VSA)

        def forward(self, x):
            if metric == "degree_centrality":
                order = degree_centrality(x)
            elif metric == "eigen_centrality":
                order = eigen_centrality(x)
            elif metric == "katz_centrality":
                order = katz_centrality(x)
            elif metric == "closeness_centrality":
                order = closeness_centrality(x)
            elif metric == "current_flow_closeness_centrality":
                order = current_flow_closeness_centrality(x)
            elif metric == "information_centrality":
                order = information_centrality(x)
            elif metric == "betweenness_centrality":
                order = betweenness_centrality(x)
            elif metric == "current_flow_betweeness_centrality":
                order = current_flow_betweeness_centrality(x)
            elif metric == "communicability_betweeness_centrality":
                order = communicability_betweeness_centrality(x)
            elif metric == "load_centrality":
                order = load_centrality(x)
            elif metric == "subgraph_centrality":
                order = subgraph_centrality(x)
            elif metric == "subgraph_centrality_exp":
                order = subgraph_centrality_exp(x)
            elif metric == "harmonic_centrality":
                order = harmonic_centrality(x)
            elif metric == "second_order_centrality":
                order = second_order_centrality(x)
            elif metric == "trophic_levels":
                order = trophic_levels(x)
            elif metric == "laplacian_centrality":
                order = laplacian_centrality(x)
            elif metric == "none":
                order = list(range(x.num_nodes))
            else:
                pr = pagerank(x)
                pr_sort, order = pr.sort()


            node_id_hvs = torchhd.empty(x.num_nodes, self.out_features, VSA)
            node_id_hvs[order] = self.node_ids.weight[: x.num_nodes]

            row, col = to_undirected(x.edge_index)

            hvs = torchhd.bind(node_id_hvs[row], node_id_hvs[col])
            return torchhd.multiset(hvs)

    min_graph_size, max_graph_size = min_max_graph_size(graphs)
    encode = Encoder(DIMENSIONS, max_graph_size, metric)
    encode = encode.to(device)

    model = Centroid(DIMENSIONS, graphs.num_classes, VSA)
    model = model.to(device)

    train_t = time.time()
    with torch.no_grad():
        for samples in tqdm(train_ld, desc="Training"):
            samples.edge_index = samples.edge_index.to(device)
            samples.y = samples.y.to(device)

            samples_hv = encode(samples).unsqueeze(0)
            model.add(samples_hv, samples.y)
    train_t = time.time() - train_t
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=graphs.num_classes)
    # f1 = torchmetrics.F1Score(num_classes=graphs.num_classes, average='macro', multiclass=True)
    f1 = torchmetrics.F1Score("multiclass", num_classes=graphs.num_classes)

    test_t = time.time()
    with torch.no_grad():
        if VSA != "BSC":
            model.normalize()

        for samples in tqdm(test_ld, desc="Testing"):
            samples.edge_index = samples.edge_index.to(device)

            samples_hv = encode(samples).unsqueeze(0)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), samples.y)
            f1.update(outputs.cpu(), samples.y)
    test_t = time.time() - test_t
    acc = accuracy.compute().item() * 100
    f = f1.compute().item() * 100
    return acc, f, train_t, test_t

REPETITIONS = 50
RANDOMNESS = ["random"]
# DATASET = ["PTC_FM", "MUTAG", "NCI1", "ENZYMES", "PROTEINS", "DD"]
METRICS = ["none","page_rank","degree_centrality",
          "closeness_centrality",
          "betweenness_centrality", "load_centrality", "subgraph_centrality",
          "subgraph_centrality_exp", "harmonic_centrality"]
DATASET = ["PTC_FM", "MUTAG", "NCI1", "ENZYMES", "PROTEINS", "DD"]
# VSAS = ["BSC", "MAP", "HRR", "FHRR"]
VSAS = ["FHRR"]


for VSA in VSAS:
    for d in DATASET:
        for METRIC in METRICS:
            acc_final = []
            f1_final = []
            train_final = []
            test_final = []
            for i in RANDOMNESS:
                acc_aux = []
                f1_aux = []
                train_aux = []
                test_aux = []
                for j in range(REPETITIONS):
                    acc, f1, train_t, test_t = experiment(1, i, d, METRIC)
                    acc_aux.append(acc)
                    f1_aux.append(f1)
                    train_aux.append(train_t)
                    test_aux.append(test_t)
                acc_final.append(round(sum(acc_aux) / REPETITIONS, 2))
                f1_final.append(round(sum(f1_aux) / REPETITIONS, 2))
                train_final.append(round(sum(train_aux) / REPETITIONS, 2))
                test_final.append(round(sum(test_aux) / REPETITIONS, 2))

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "dataset",
                        "dimensions",
                        "train_time",
                        "test_time",
                        "accuracy",
                        "f1",
                        "VSA",
                        "metric"
                    ]
                )
                writer.writerows(
                    [
                        [
                            d,
                            DIM,
                            train_final[0],
                            test_final[0],
                            acc_final[0],
                            f1_final[0],
                            VSA,
                            METRIC
                        ]
                    ]
                )
