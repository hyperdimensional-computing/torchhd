import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv("../results/final_results/compare_embeddings_singlepass_arena")
var = "Encoding"
df = df[df["Method"] == "adjust"]
embeddings_order = ["add", "adapt", "online", "adjust"]
embeddings_order = ["hashmap", "sinusoid", "flocet"]
mean_accuracy_by_dimension_and_method = df.groupby(
    ["Dimensions", "Method", "Encoding"]
)["Accuracy"].mean()

print(mean_accuracy_by_dimension_and_method)

var_accuracy_by_dimension_and_method = df.groupby(["Dimensions", "Method"])[
    "Accuracy"
].var()
# print(mean_accuracy_by_dimension_and_method)
# print()
# print()
# print("-------------")
# print()
# print()
# print("varaicne",var_accuracy_by_dimension_and_method)
df_mean = df.groupby([var, "Name"])["TrainTime"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="TrainTime")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)

# print()
# print("Train time")
# print(df_pivot)
# print()
# print()

df_mean = df.groupby([var, "Name"])["TestTime"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="TestTime")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)

# print()
# print("Test time")
# print(df_pivot)
# print()
# print()

df_mean = df.groupby([var, "Name"])["Accuracy"].mean().to_frame()
df_var = df.groupby([var, "Name"])["Accuracy"].var().to_frame()

df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="Accuracy")
df_pivot.loc["Var"] = df_pivot.var(axis=0)

print("VARAINCE", df_pivot)

df_pivot.loc["Mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)


column_names = df_pivot.columns.tolist()
column_dict = {column_name: 0 for column_name in column_names}


def count_max(row):
    max_val = row.max()
    maxx = list(row[row == max_val].index)
    for i in maxx:
        column_dict[i] += 1


max_counts = df_pivot.apply(count_max, axis=1)


# print()
# print("Accuracy")
# print(df_pivot)

df_pivot = df_pivot.apply(
    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
)
latex_table = df_pivot.to_latex(index=True)

# print(latex_table)
# print(pd.DataFrame(column_dict, index=["Best"]))


# mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.apply(
#    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
# )
mean_accuracy_by_dimension_and_method = (
    df.groupby(["Dimensions", "Method"])["Accuracy"].mean().round(3)
)
# mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.transpose()
order = ["add", "adapt", "online", "adjust"]
mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)
mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.apply(
    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
)
latex_table = mean_accuracy_by_dimension_and_method.to_latex(index=True)
print(latex_table)
# print("varaicne",var_accuracy_by_dimension_and_method)

var_accuracy_by_dimension_and_method = (
    df.groupby(["Dimensions", "Method"])["Accuracy"].var().round(5)
)
print(var_accuracy_by_dimension_and_method)
order = ["add", "adapt", "online", "adjust"]
var_accuracy_by_dimension_and_method = var_accuracy_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)
var_accuracy_by_dimension_and_method = var_accuracy_by_dimension_and_method.apply(
    lambda row: row.replace(row.min(), "bof(" + str(row.min()) + ")"), axis=1
)
latex_table = var_accuracy_by_dimension_and_method.to_latex(index=True)
print(latex_table)


train_time_by_dimension_and_method = (
    df.groupby(["Dimensions", "Method"])["TrainTime"].var().round(3)
)

print(train_time_by_dimension_and_method)
order = ["add", "adapt", "online", "adjust"]
train_time_by_dimension_and_method = train_time_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)
train_time_by_dimension_and_method = train_time_by_dimension_and_method.apply(
    lambda row: row.replace(row.min(), "bof(" + str(row.min()) + ")"), axis=1
)
latex_table = train_time_by_dimension_and_method.to_latex(index=True)
print(latex_table)


test_time_by_dimension_and_method = (
    df.groupby(["Dimensions", "Method"])["TestTime"].var().round(3)
)

print(test_time_by_dimension_and_method)
order = ["add", "adapt", "online", "adjust"]
test_time_by_dimension_and_method = test_time_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)
test_time_by_dimension_and_method = test_time_by_dimension_and_method.apply(
    lambda row: row.replace(row.min(), "bof(" + str(row.min()) + ")"), axis=1
)
latex_table = test_time_by_dimension_and_method.to_latex(index=True)
print(latex_table)


# df = df[df["Dimensions"] == 10000]
# df = df[df["Encoding"] == 'adjust']
print(df)
df_mean = df.groupby([var, "Name"])["Accuracy"].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="Accuracy")
df_pivot.loc["Mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[embeddings_order].round(3)
print(df_pivot)

df_pivot = df_pivot.apply(
    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
)
latex_table = df_pivot.to_latex(index=True)

print(latex_table)
