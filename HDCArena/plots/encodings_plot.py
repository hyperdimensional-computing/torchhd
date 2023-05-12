import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv("../results/final_results/encodings_comparison_arena")
var = "Method"

methods_order = [
    "add",
    "adapt",
    "online",
    "adapt_iterative",
    "online_iterative",
    "quant_iterative",
    "sparse_iterative",
    "neural_iterative",
    "dist_iterative",
    "multicentroid",
    "comp",
    # "noise",
    "adjust",
]

df_mean = df.groupby([var, "Name"])["TrainTime"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="TrainTime")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)

print()
print("Train time")
print(df_pivot)
print()
print()

df_mean = df.groupby([var, "Name"])["TestTime"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="TestTime")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)

print()
print("Test time")
print(df_pivot)
print()
print()

df_mean = df.groupby([var, "Name"])["Accuracy"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="Accuracy")
df_pivot.loc["Mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)


column_names = df_pivot.columns.tolist()
column_dict = {column_name: 0 for column_name in column_names}


def count_max(row):
    max_val = row.max()
    maxx = list(row[row == max_val].index)
    for i in maxx:
        column_dict[i] += 1


max_counts = df_pivot.apply(count_max, axis=1)


print()
print("Accuracy")
print(df_pivot)

df_pivot = df_pivot.apply(
    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
)
latex_table = df_pivot.to_latex(index=True)

print(latex_table)
print(pd.DataFrame(column_dict, index=["Best"]))
