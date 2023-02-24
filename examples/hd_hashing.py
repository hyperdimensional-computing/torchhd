import torch

# Note: this example requires the mmh3 library: https://github.com/hajimes/mmh3
import mmh3

import torchhd


class HDHashing:
    """Hyperdimensional Computing dynamic hash table"""

    def __init__(self, levels: int, dimensions: int, device=None):
        self.levels = levels
        self.dimensions = dimensions
        self.device = device

        self.hvs = torchhd.circular(levels, dimensions, device=device)
        self.servers = []
        self.server_hvs = []
        self.weight_by_server = {}

    def hash(self, value: str, seed: int = 0):
        """Hash function return uniform pseudo-random numbers
        Returns a value between 0 and self.levels
        """
        return mmh3.hash(value, seed, signed=False) % self.levels

    def request(self, value: str):
        """Assigns a request to an available server"""
        h = self.hash(value)
        hv = self.hvs[h]

        server_hvs = torch.stack(self.server_hvs, dim=0)

        # The next three lines simulate associative memory in HDC
        # It returns the value at the memory location (server)
        # that is most similar to the requested location (request).
        similarity = torchhd.dot_similarity(hv, server_hvs)
        server_idx = torch.argmax(similarity).item()
        return self.servers[server_idx]

    def join(self, value: str, weight: int = 1):
        """Adds a new server to the set of available servers in the cluster.

        The weight of the server indicates the number of times the normal load
        that the server will receive.
        """
        self.weight_by_server[value] = weight

        for i in range(weight):
            # Using the index as the seed ensures that each virtual copy
            # of the server is assigned a random location.
            h = self.hash(value, i)
            hv = self.hvs[h]

            self.servers.append(value)
            self.server_hvs.append(hv)

    def leave(self, value: str):
        """Removes a server from the set of available servers in the cluster"""
        weight = self.weight_by_server.get(value, None)
        if weight == None:
            return

        del self.weight_by_server[value]

        server_idx = self.servers.index(value)
        if server_idx == -1:
            return

        # Since all servers are added consecutively we can delete
        # all occurrences if we know the start index and the length.
        del self.servers[server_idx : server_idx + weight]
        del self.server_hvs[server_idx : server_idx + weight]


if __name__ == "__main__":
    hash_table = HDHashing(512, 4096)

    # Three servers join the cluster by their IP address
    hash_table.join("7.225.242.236")
    hash_table.join("244.144.238.83")
    hash_table.join("44.41.24.132")

    # Request is assigned to a server
    server_ip = hash_table.request("34.152.205.192")
    print(server_ip)

    # New server joins the cluster with a weight of 3
    # Meaning it takes 3 times the normal load
    hash_table.join("206.88.93.11", weight=3)

    # Requests are assigned to a server
    server_ip = hash_table.request("34.152.205.192")
    print(server_ip)

    server_ip = hash_table.request("63.54.53.26")
    print(server_ip)

    server_ip = hash_table.request("99.34.0.207")
    print(server_ip)

    # Servers leaves the cluster
    hash_table.leave("244.144.238.83")
    hash_table.leave("206.88.93.11")

    # Requests are assigned to a server
    server_ip = hash_table.request("150.167.170.96")
    print(server_ip)

    server_ip = hash_table.request("37.229.115.130")
    print(server_ip)

    server_ip = hash_table.request("22.163.83.231")
    print(server_ip)
