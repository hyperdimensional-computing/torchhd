import pandas as pd


# colors: ['333436', '005CAB', 'E31B23', '78468E', '1BA13D', 'EA8810', '898b8c', '69A7DD', 'F87272', 'B186D1', '7EC095', 'F4BF70']

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/results1692628926.944422.csv"
)
print(df.head())

for i in df["method"].unique():
    print(i)
    mean_accuracy_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["accuracy"].mean()
    )

    variance_accuracy_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["accuracy"].std()
    )

    mean_train_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["train_time"].mean()
    )

    variance_train_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["train_time"].std()
    )

    mean_test_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["test_time"].mean()
    )

    variance_test_by_dimension_and_method = (
        df[df["method"] == i].groupby(["name", "method"])["test_time"].std()
    )




    print(
        i,
        mean_accuracy_by_dimension_and_method.mean(),
        variance_accuracy_by_dimension_and_method.mean(),
    )



    print(
        i,
        mean_train_by_dimension_and_method.mean(),
        variance_train_by_dimension_and_method.mean(),
    )



    print(
        i,
        mean_test_by_dimension_and_method.mean(),
        variance_test_by_dimension_and_method.mean(),
    )
