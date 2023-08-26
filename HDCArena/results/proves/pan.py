import pandas as pd

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/results1684710856.177537.csv"
)
print(df.head())

mean_accuracy_by_dimension_and_method = df.groupby(["partial_data", "method"])[
    "accuracy"
].mean()


print(mean_accuracy_by_dimension_and_method)
