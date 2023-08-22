import pandas as pd
import warnings
import torch

warnings.filterwarnings("ignore")

# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/refine_dimensions_arena"
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/adjust"
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/dimensions_arenaaa.csv"
file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/online_dimensions_uci"

file2 = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/refine_dimensions_uci"

df = pd.read_csv(file)
df2 = pd.read_csv(file2)

df = df[df["dimensions"] == 10000]
df2 = df2[df2["dimensions"] == 10000]

print(df.groupby(["method"])["accuracy"].mean())
latex = 1
pand = 1

mean_of_encoding = df.groupby(["method"])["accuracy"].mean().round(3).reset_index().T
var_of_encoding = df.groupby(["method"])["accuracy"].std().round(3).reset_index().T


# time_of_encoding = df.groupby(["method"])["train_time"].mean().round(3).reset_index().T
# var_time_of_encoding = df.groupby(["method","name"])["train_time"].var().round(3).reset_index().T
mean_of_encoding_train_time = df.groupby(["name"])["accuracy"].mean().round(3).T
# print(mean_of_encoding_train_time.T)
mean_of_encoding_train_time2 = df2.groupby(["name"])["accuracy"].mean().round(3).T

import numpy as np

print(
    "MAAX",
    max(mean_of_encoding_train_time2.T - mean_of_encoding_train_time.T),
    np.argmax(mean_of_encoding_train_time2.T - mean_of_encoding_train_time.T),
)
print(
    "MIIN",
    min(mean_of_encoding_train_time2.T - mean_of_encoding_train_time.T),
    np.argmin(mean_of_encoding_train_time2.T - mean_of_encoding_train_time.T),
)

print(mean_of_encoding_train_time2.T.iloc[79])
print(mean_of_encoding_train_time.T.iloc[79])
mean_of_encoding_train_time = (
    df.groupby(["method"])["train_time"].mean().round(3).reset_index().T
)
var_of_encoding_train_time = (
    df.groupby(["method"])["train_time"].std().round(3).reset_index().T
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
    df.groupby(["method", "dimensions"])["accuracy"].mean().round(3).reset_index().T
)
print(mean_of_encoding)
var_of_encoding = (
    df.groupby(["method", "dimensions"])["accuracy"].std().round(3).reset_index().T
)
print(var_of_encoding)
