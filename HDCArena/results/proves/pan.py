import pandas as pd

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/results1684767254.650597.csv"
)
print(df.head())
df = df[df['partial_data'] == 0.25]
df = df[df['robustness_failed_dimensions'] == 0]
df = df[df['method'] == 'adjust']

mean_accuracy_by_dimension_and_method = df.groupby(["name", "method"])[
    "accuracy"
].mean()

variance_accuracy_by_dimension_and_method = df.groupby(["name", "method"])[
    "accuracy"
].var()

print(mean_accuracy_by_dimension_and_method.mean())
