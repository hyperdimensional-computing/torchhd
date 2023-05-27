import pandas as pd

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/multicentroid_method"
)
print(df.head())

for i in df["method"].unique():
    print(i)
    mean_accuracy_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["accuracy"].mean()
    )

    variance_accuracy_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["accuracy"].var()
    )

    print(
        i,
        mean_accuracy_by_dimension_and_method.mean(),
        variance_accuracy_by_dimension_and_method.mean(),
    )
