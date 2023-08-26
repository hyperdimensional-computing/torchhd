import pandas as pd

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/multicentroid_robust_partial"
)

latex = 1
pand = 1
df = df[df["robustness_failed_dimensions"] == 0]
mean_of_encoding = (
    df.groupby(["encoding", "partial_data"])["accuracy"].mean().round(3).reset_index().T
)
var_of_encoding = (
    df.groupby(["encoding", "partial_data"])["accuracy"].std().round(3).reset_index().T
)

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)

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
