import torch
from torchhd import functional

d = 10000  # dimensions
F = functional.random_hv(3, d)  # fruits
W = functional.level_hv(10, d)  # weights
S = functional.circular_hv(4, d)  # seasons
V = functional.random_hv(3, d)  # variables

# explicit mapping of the fruit weight to an index
weight = torch.tensor([149.0])
w_i = functional.value_to_index(weight, 0, 200.0, 10)
W[w_i]  # select representation of 149

f = functional.bind(V[0], F[0])  # fruit = apple
w = functional.bind(V[1], W[w_i])  # weight = 149
s = functional.bind(V[2], S[3])  # season = fall
r1 = functional.bundle(functional.bundle(f, w), s)
# equivalent short-hand encoding of record r1:
# r1 = V[0] * F[0] + V[1] * W[w_i] + V[2] * S[3]

season = functional.bind(r1, V[2])

memory = torch.cat([F, W, S])
similarity = functional.cosine_similarity(season, memory)

import matplotlib.pyplot as plt

plt.style.use(["science", "nature"])

fig, ax = plt.subplots(1, 1, figsize=(3.7, 2.2))

weights = functional.index_to_value(torch.arange(0, 10), 10, 0, 200).tolist()
weights = [f"{w:.0f}" for w in weights]
concepts = (
    ["Apple", "Lemon", "Mango"] + weights + ["Winter", "Summer", "Spring", "Fall"]
)

markerline, stemlines, baseline = ax.stem(
    concepts,
    similarity.tolist(),
    "midnightblue",
    use_line_collection=True,
    markerfmt="o",
)
plt.setp(markerline, "color", "midnightblue")
plt.setp(stemlines, "color", "midnightblue")
plt.setp(baseline, "color", "midnightblue", "alpha", 0.3)

ax.set_xticks([i for i in range(len(concepts))], concepts)
ax.set_xticklabels(concepts, rotation=65)
ax.set_xlabel("Hypervector", labelpad=-5)
ax.set_ylabel("Cosine similarity")
plt.savefig("similarity.pgf")
plt.savefig("similarity.png", dpi=300)
