import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv("../results/survey_results/encodings_arena")
var = "encoding"

embeddings_order = ["sequence", "hashmap", "flocet", "sinusoid", "generic"]

df_mean = df.groupby([var, "name"])["train_time"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="name", columns=var, values="train_time")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)

print()
print("Train time")
print(df_pivot)
print()
print()

df_mean = df.groupby([var, "name"])["test_time"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="name", columns=var, values="test_time")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)

print()
print("Test time")
print(df_pivot)
print()
print()

df_mean = df.groupby([var, "name"])["accuracy"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="name", columns=var, values="accuracy")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)


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
