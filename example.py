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

fruit = functional.bind(r1, V[0])
weight = functional.bind(r1, V[1])
season = functional.bind(r1, V[2])

memory = torch.cat([F, W, S])
fruit_similarity = functional.cosine_similarity(fruit, memory)
weight_similarity = functional.cosine_similarity(weight, memory)
season_similarity = functional.cosine_similarity(season, memory)

import matplotlib.pyplot as plt

plt.style.use(["science", "nature"])

fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.2))

weights = functional.index_to_value(torch.arange(0, 10), 10, 0, 200).tolist()
weights = [f"{w:.0f}" for w in weights]
concepts = (
    ["Apple", "Lemon", "Mango"] + weights + ["Winter", "Spring", "Summer", "Fall"]
)

markerline, stemlines, baseline = ax.stem(
    concepts,
    fruit_similarity.tolist(),
    "#F7934C",
    use_line_collection=True,
    markerfmt="o",
    label="Fruit",
)
plt.setp(markerline, "color", "#F7934C")
plt.setp(stemlines, "color", "#F7934C", "alpha", 0.7)
plt.setp(baseline, "color", "#F7934C", "alpha", 0.3)

markerline, stemlines, baseline = ax.stem(
    concepts,
    weight_similarity.tolist(),
    "#04724D",
    use_line_collection=True,
    markerfmt="o",
    label="Weight",
)
plt.setp(markerline, "color", "#04724D")
plt.setp(stemlines, "color", "#04724D", "alpha", 0.7)
plt.setp(baseline, "color", "#04724D", "alpha", 0.3)

markerline, stemlines, baseline = ax.stem(
    concepts,
    season_similarity.tolist(),
    "#2659A6",
    use_line_collection=True,
    markerfmt="o",
    label="Season",
)
plt.setp(markerline, "color", "#2659A6")
plt.setp(stemlines, "color", "#2659A6", "alpha", 0.7)
plt.setp(baseline, "color", "#2659A6", "alpha", 0.3)

ax.set_xticks([i for i in range(len(concepts))], concepts)
ax.set_xticklabels(concepts, rotation=65)
ax.set_xlabel("Hypervector", labelpad=-5)
ax.set_ylabel("Cosine similarity")
ax.margins(y=0.1)
plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", title="Variable")
plt.savefig("record-similarity.pgf")
plt.savefig("record-similarity.png", dpi=300)
