import pandas as pd
import warnings

warnings.filterwarnings("ignore")

file = (
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/adjust_arena"
)
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/adjust"
# file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/final_results/dimensions_arenaaa.csv"

df = pd.read_csv(file)

# df = df[df['Dimensions'] == 10000]

print(df.groupby(["method"])["accuracy"].mean())
latex = 1
pand = 1

mean_of_encoding = df.groupby(["method"])["accuracy"].mean().round(3).reset_index().T
var_of_encoding = df.groupby(["method"])["accuracy"].std().round(3).reset_index().T


# time_of_encoding = df.groupby(["method"])["train_time"].mean().round(3).reset_index().T
# var_time_of_encoding = df.groupby(["method","name"])["train_time"].var().round(3).reset_index().T
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
    df.groupby(["method", "name"])["accuracy"].mean().round(3).reset_index().T
)
print(mean_of_encoding)
