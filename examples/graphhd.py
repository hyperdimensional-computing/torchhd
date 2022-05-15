import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Note: this example requires the torch_geometric library: https://pytorch-geometric.readthedocs.io
import torch_geometric

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

from torchhd import functional
from torchhd import embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # hypervectors dimension

# for other available datasets see: https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html?highlight=tudatasets
dataset = "MUTAG"

graphs = torch_geometric.datasets.TUDataset("data", dataset)
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


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.node_ids = embeddings.Random(size, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        pr = pagerank(x)
        pr_sort, pr_argsort = pr.sort()

        node_id_hvs = torch.zeros((x.num_nodes, DIMENSIONS), device=device)
        node_id_hvs[pr_argsort] = self.node_ids.weight[: x.num_nodes]

        row, col = to_undirected(x.edge_index)

        hvs = functional.bind(node_id_hvs[row], node_id_hvs[col])
        return functional.multiset(hvs)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


min_graph_size, max_graph_size = min_max_graph_size(graphs)
model = Model(graphs.num_classes, max_graph_size)
model = model.to(device)

with torch.no_grad():
    for samples in tqdm(train_ld, desc="Training"):
        samples.edge_index = samples.edge_index.to(device)
        samples.y = samples.y.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[samples.y] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

accuracy = torchmetrics.Accuracy()

with torch.no_grad():
    for samples in tqdm(test_ld, desc="Testing"):
        samples.edge_index = samples.edge_index.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1).unsqueeze(0)
        accuracy.update(predictions.cpu(), samples.y)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
