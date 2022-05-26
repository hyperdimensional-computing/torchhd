import pytest
import torch
import string

from torchhd import structures, functional

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestGraph:
    def test_creation_dim(self):
        G = structures.Graph(10000, directed=True)
        assert torch.equal(G.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(len(letters), 10000, generator=generator)
        g = functional.bind(hv[0], hv[1])
        G = structures.Graph(g)
        assert torch.equal(G.value, g)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_add_edge(self):
        G = structures.Graph(8)
        hv = torch.tensor(
            [
                [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
            ]
        )

        G.add_edge(hv[0], hv[1])
        assert torch.equal(
            G.value, torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        )
        G.add_edge(hv[2], hv[3])
        assert torch.equal(
            G.value, torch.tensor([-2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0])
        )

        GD = structures.Graph(8, directed=True)

        GD.add_edge(hv[0], hv[1])
        assert torch.equal(
            GD.value, torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        )
        GD.add_edge(hv[2], hv[3])
        assert torch.equal(
            GD.value, torch.tensor([0.0, 0.0, 0.0, -2.0, 0.0, -2.0, 2.0, -2.0])
        )

    def test_encode_edge(self):
        G = structures.Graph(8)
        hv = torch.tensor(
            [
                [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
            ]
        )

        e1 = G.encode_edge(hv[0], hv[1])
        assert torch.equal(
            e1, torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        )
        e2 = G.encode_edge(hv[2], hv[3])
        assert torch.equal(
            e2, torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
        )

        GD = structures.Graph(8, directed=True)

        e1 = GD.encode_edge(hv[0], hv[1])
        assert torch.equal(
            e1, torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        )
        e2 = GD.encode_edge(hv[2], hv[3])
        print(e2)
        assert torch.equal(
            e2, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        )

    def test_node_neighbors(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(10, 10000, generator=generator)
        G = structures.Graph(10000, directed=True)

        G.add_edge(hv[0], hv[1])
        G.add_edge(hv[0], hv[2])
        G.add_edge(hv[1], hv[2])

        assert (
            torch.argmax(
                functional.cosine_similarity(G.node_neighbors(hv[1]), hv)
            ).item()
            == 2
        )
        assert functional.cosine_similarity(G.node_neighbors(hv[1]), hv)[2] > 0.5
        assert functional.cosine_similarity(G.node_neighbors(hv[0]), hv)[2] > 0.5
        assert functional.cosine_similarity(G.node_neighbors(hv[0]), hv)[1] > 0.5
        assert functional.cosine_similarity(G.node_neighbors(hv[2]), hv)[1] < 0.5
        assert functional.cosine_similarity(G.node_neighbors(hv[2]), hv)[0] < 0.5
        assert functional.cosine_similarity(G.node_neighbors(hv[1]), hv)[0] < 0.5

        G1 = structures.Graph(10000, directed=False)

        G1.add_edge(hv[0], hv[1])
        G1.add_edge(hv[0], hv[2])
        G1.add_edge(hv[1], hv[2])
        assert functional.cosine_similarity(G1.node_neighbors(hv[1]), hv)[0] > 0.5
        assert functional.cosine_similarity(G1.node_neighbors(hv[0]), hv)[1] > 0.5
        assert functional.cosine_similarity(G1.node_neighbors(hv[0]), hv)[2] > 0.5
        assert functional.cosine_similarity(G1.node_neighbors(hv[2]), hv)[0] > 0.5
        assert functional.cosine_similarity(G1.node_neighbors(hv[1]), hv)[2] > 0.5
        assert functional.cosine_similarity(G1.node_neighbors(hv[2]), hv)[1] > 0.5

    def test_contains(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(4, 8, generator=generator)
        G = structures.Graph(8)

        e1 = G.encode_edge(hv[0], hv[1])
        e2 = G.encode_edge(hv[0], hv[2])
        e3 = G.encode_edge(hv[2], hv[3])

        G.add_edge(hv[0], hv[1])
        G.add_edge(hv[0], hv[2])
        G.add_edge(hv[1], hv[2])

        assert G.contains(e1) > torch.tensor(0.6)
        assert G.contains(e2) > torch.tensor([0.6])
        assert G.contains(e3) < torch.tensor(0.6)

        GD = structures.Graph(8, directed=True)

        ee1 = GD.encode_edge(hv[0], hv[1])
        ee2 = GD.encode_edge(hv[0], hv[2])
        ee3 = GD.encode_edge(hv[2], hv[3])
        ee4 = GD.encode_edge(hv[1], hv[0])

        GD.add_edge(hv[0], hv[1])
        GD.add_edge(hv[0], hv[2])
        GD.add_edge(hv[3], hv[1])

        assert GD.contains(ee1) > torch.tensor(0.6)
        assert GD.contains(ee2) > torch.tensor(0.6)
        assert GD.contains(ee3) < torch.tensor(0.6)
        assert GD.contains(ee4) < torch.tensor(0.6)

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random_hv(4, 8, generator=generator)
        G = structures.Graph(8)

        G.add_edge(hv[0], hv[1])
        G.add_edge(hv[0], hv[2])
        G.add_edge(hv[1], hv[2])

        G.clear()

        assert torch.equal(
            G.value, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
