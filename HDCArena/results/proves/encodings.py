import pandas as pd
import warnings

warnings.filterwarnings("ignore")

latex = 1
pand = 1
train_time = 1
test_time = 1

embeddings_order = [
    "bundle",
    "sequence",
    "ngram",
    "hashmap",
    "density",
    "flocet",
    "generic",
    "random",
    "sinusoid",
]
embeddings_order = [0, 7, 5, 4, 1, 2, 3, 6, 8]

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/encodings"
)

for i in df["encoding"].unique():
    variance_accuracy_by_dimension_and_method = (
        df[df["encoding"] == i].groupby(["name", "method"])["accuracy"].var()
    )
    # print(variance_accuracy_by_dimension_and_method.mean())


mean_of_encoding = (
    df.groupby(["encoding"])["accuracy"].mean().round(3).reset_index().T
)[embeddings_order]

var_of_encoding = (
    df.groupby(["encoding", "name", "method"])["accuracy"]
    .var()
    .groupby("encoding")
    .mean()
    .reset_index()
    .T
)[embeddings_order]

mean_of_encoding_train_time = (
    df.groupby(["encoding"])["train_time"].mean().round(3).reset_index().T
)[embeddings_order]

mean_of_encoding_test_time = (
    df.groupby(["encoding"])["test_time"].mean().round(3).reset_index().T
)[embeddings_order]

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)
    if train_time:
        print(mean_of_encoding_train_time)
    if test_time:
        print(mean_of_encoding_test_time)

if latex:
    latex_table = mean_of_encoding.to_latex(
        index=False, caption="Encodings accuracy mean"
    )
    print(latex_table)
    pd.options.display.float_format = "{:.2e}".format
    latex_table = var_of_encoding.to_latex(
        index=False, caption="Encodings accuracy variance"
    )
    print(latex_table)
    pd.options.display.float_format = None

    if train_time:
        latex_table = mean_of_encoding_train_time.to_latex(
            index=False, caption="Encodings train time mean"
        )
        print(latex_table)
    if test_time:
        latex_table = mean_of_encoding_test_time.to_latex(
            index=False, caption="Encodings test time mean"
        )
        print(latex_table)
