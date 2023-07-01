import pandas as pd
'''
VANILLA
ADAPT
ONLINE
ADAPT_ITER
ONLINE_ITER
NERUAL
DIST
SPARES
QUANT
COMP
SEMI
MULTI
RVFL
LEHDC
'''



df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/lehdc_robust_partial_arena"
)
latex = 1
pand = 1
df = df[df["partial_data"] == 1]

print(
    df.sort_values(["robustness_failed_dimensions"], ascending=False)
    .groupby(["robustness_failed_dimensions", "name"])["accuracy"]
    .mean()
    .to_string()
)
mean_of_encoding = (
    df.sort_values(["robustness_failed_dimensions"], ascending=False)
    .groupby(["robustness_failed_dimensions", "name"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
)
mean_of_encoding = (
    mean_of_encoding.groupby(["robustness_failed_dimensions"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
    .T
)
var_of_encoding = (
    df.sort_values(["robustness_failed_dimensions"], ascending=False)
    .groupby(["robustness_failed_dimensions", "name"])["accuracy"]
    .std()
    .round(3)
    .reset_index()
)
var_of_encoding = (
    var_of_encoding.groupby(["robustness_failed_dimensions"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
    .T
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

    def join_with_commas(row):
        return ", ".join(str(value) for value in row)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # Apply the function to each row
    joined_values = mean_of_encoding.apply(join_with_commas, axis=1)

    # Print the joined values
    print(joined_values["robustness_failed_dimensions"])
    print(joined_values["accuracy"])
