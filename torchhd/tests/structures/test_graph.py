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
import pytest
import torch
import string

from torchhd import structures, functional
from torchhd.tensors.map import MAPTensor

seed = 2147483645
letters = list(string.ascii_lowercase)


class TestGraph:
    def test_creation_dim(self):
        G = structures.Graph(10000, directed=True)
        assert torch.allclose(G.value, torch.zeros(10000))

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        g = functional.bind(hv[0], hv[1])
        G = structures.Graph(g)
        assert torch.allclose(G.value, g)

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv1 = functional.random(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(seed)
        hv2 = functional.random(60, 10000, generator=generator)

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
        ).as_subclass(MAPTensor)

        G.add_edge(hv[0], hv[1])
        assert torch.allclose(
            G.value, torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        )
        G.add_edge(hv[2], hv[3])
        assert torch.allclose(
            G.value, torch.tensor([-2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0])
        )

        GD = structures.Graph(8, directed=True)

        GD.add_edge(hv[0], hv[1])
        assert torch.allclose(
            GD.value, torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        )
        GD.add_edge(hv[2], hv[3])
        assert torch.allclose(
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
        ).as_subclass(MAPTensor)

        e1 = G.encode_edge(hv[0], hv[1])
        assert torch.allclose(
            e1, torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        )
        e2 = G.encode_edge(hv[2], hv[3])
        assert torch.allclose(
            e2, torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
        )

        GD = structures.Graph(8, directed=True)

        e1 = GD.encode_edge(hv[0], hv[1])
        assert torch.allclose(
            e1, torch.tensor([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        )
        e2 = GD.encode_edge(hv[2], hv[3])
        assert torch.allclose(
            e2, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        )

    def test_node_neighbors(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(10, 10000, generator=generator)
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
        hv = functional.random(4, 1000, generator=generator)
        G = structures.Graph(1000)

        e1 = G.encode_edge(hv[0], hv[1])
        e2 = G.encode_edge(hv[0], hv[2])
        e3 = G.encode_edge(hv[2], hv[3])

        G.add_edge(hv[0], hv[1])
        G.add_edge(hv[0], hv[2])
        G.add_edge(hv[1], hv[2])

        assert G.contains(e1) > torch.tensor(0.6)
        assert G.contains(e2) > torch.tensor([0.6])
        assert G.contains(e3) < torch.tensor(0.6)

        GD = structures.Graph(1000, directed=True)

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
        hv = functional.random(4, 8, generator=generator)
        G = structures.Graph(8)

        G.add_edge(hv[0], hv[1])
        G.add_edge(hv[0], hv[2])
        G.add_edge(hv[1], hv[2])

        G.clear()

        assert torch.allclose(
            G.value, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    def test_from_edges(self):
        generator = torch.Generator()
        generator.manual_seed(seed)

        hv = functional.random(4, 1000, generator=generator)
        edges = torch.empty(2, 3, 1000).as_subclass(MAPTensor)
        edges[0, 0] = hv[0]
        edges[1, 0] = hv[1]
        edges[0, 1] = hv[0]
        edges[1, 1] = hv[2]
        edges[0, 2] = hv[1]
        edges[1, 2] = hv[2]

        G = structures.Graph.from_edges(edges)
        neighbors = G.node_neighbors(hv[0])
        neighbor_similarity = functional.cosine_similarity(neighbors, hv)

        assert neighbor_similarity[0] < torch.tensor(0.5)
        assert neighbor_similarity[1] > torch.tensor(0.5)
        assert neighbor_similarity[2] > torch.tensor(0.5)
        assert neighbor_similarity[3] < torch.tensor(0.5)
