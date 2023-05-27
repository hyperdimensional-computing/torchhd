import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/encodings"
)

for i in df["encoding"].unique():

    variance_accuracy_by_dimension_and_method = (
        df[df["encoding"] == i].groupby(["name", "method"])["accuracy"].var()
    )
    print(variance_accuracy_by_dimension_and_method.mean())

mean_of_encoding = (
    df.groupby(["encoding"])["accuracy"].mean().round(3).reset_index().T
)

var_of_encoding = (
    df.groupby(["encoding", "name", "method"])["accuracy"].var().groupby("encoding").mean().reset_index().T
)

latex = 1
pand = 1

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)

if latex:
    latex_table = mean_of_encoding.to_latex(index=False, caption='Encodings accuracy mean')
    print(latex_table)
    pd.options.display.float_format = '{:.2e}'.format
    latex_table = var_of_encoding.to_latex(index=False, caption='Encodings accuracy variance')
    print(latex_table)
