import pandas as pd

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/results"
)
print(df.head())

mean_accuracy_by_dimension_and_method = df.groupby(["name","method"])[
    "accuracy"
].mean()

variance_accuracy_by_dimension_and_method = df.groupby(["name","method"])[
    "accuracy"
].var()

print(mean_accuracy_by_dimension_and_method.mean())
print(variance_accuracy_by_dimension_and_method.mean())
