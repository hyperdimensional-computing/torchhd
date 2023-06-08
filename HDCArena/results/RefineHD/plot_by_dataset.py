import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings("ignore")

file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/add_dimensions_uci"
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/adjust"
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/dimensions_arenaaa.csv"

df = pd.read_csv(file)

df = df[df['dimensions'] == 10000]

print(df.groupby(["method", "name"])["accuracy"].mean())
latex = 0
pand = 0

mean_of_encoding = (
    df.groupby(["method", "name"])["accuracy"].mean().round(3).reset_index().T
)
var_of_encoding = (
    df.groupby(["method", "name"])["accuracy"].std().round(3).reset_index().T
)


# time_of_encoding = df.groupby(["method"])["train_time"].mean().round(3).reset_index().T
# var_time_of_encoding = df.groupby(["method","name"])["train_time"].var().round(3).reset_index().T
mean_of_encoding_train_time = (
    df.groupby(["method", "name"])["train_time"].mean().round(3).reset_index().T
)
var_of_encoding_train_time = (
    df.groupby(["method", "name"])["train_time"].std().round(3).reset_index().T
)

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)
    print(var_of_encoding_train_time)

if latex:
    latex_table = mean_of_encoding.to_latex(
        index=False, caption="Encodings accuracy mean"
    )
    print(latex_table)
    # pd.options.display.float_format = "{:.2e}".format
    latex_table = var_of_encoding.to_latex(
        index=False, caption="Encodings accuracy variance"
    )
    print(latex_table)
    pd.options.display.float_format = None

    latex_table = mean_of_encoding_train_time.to_latex(
        index=False, caption="Encodings accuracy time mean"
    )
    print(latex_table)

    latex_table = var_of_encoding_train_time.to_latex(
        index=False, caption="Encodings accuracy time var"
    )
    print(latex_table)

mean_of_encoding = (
    df.groupby(["method", "dimensions", "name"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
    .T
)
print(mean_of_encoding)
print(mean_of_encoding.T['accuracy'].mean())


def std(x):
    return np.std(x)


var_of_encoding = df.groupby(["method", "dimensions", "name"])["accuracy"].std().round(3).reset_index().T

#print(df.groupby(["dimensions", "name"])["accuracy"].agg(["min"]))
#print(df.groupby(["dimensions", "name"])["accuracy"].agg(["max"]))
#result = df.groupby(["dimensions", "name"])["accuracy"].agg(["std", "mean"])
#print(var_of_encoding)
print(var_of_encoding.T['accuracy'].mean())
