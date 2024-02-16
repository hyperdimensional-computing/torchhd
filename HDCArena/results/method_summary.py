import pandas as pd


# colors: ['333436', '005CAB', 'E31B23', '78468E', '1BA13D', 'EA8810', '898b8c', '69A7DD', 'F87272', 'B186D1', '7EC095', 'F4BF70']

df = pd.read_csv("survey_results/semi_uci")

for i in df["method"].unique():
    print("Method name: " + i)
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
        "Accuracy",
        i,
        mean_accuracy_by_dimension_and_method.mean(),
        variance_accuracy_by_dimension_and_method.mean(),
    )

    print(
        "Train time",
        i,
        mean_train_by_dimension_and_method.mean(),
        variance_train_by_dimension_and_method.mean(),
    )

    print(
        "Test time",
        i,
        mean_test_by_dimension_and_method.mean(),
        variance_test_by_dimension_and_method.mean(),
    )
