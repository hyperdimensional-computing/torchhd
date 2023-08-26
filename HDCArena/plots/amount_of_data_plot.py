import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in the CSV file
df = pd.read_csv("../results/final_results/amount_of_data_arena")
var = "Method"
import seaborn as sns

embeddings_order = ["add", "adapt", "online", "adjust"]
mean_accuracy_by_dimension_and_method = df.groupby(["AmountData", "Method"])[
    "Accuracy"
].mean()
var_accuracy_by_dimension_and_method = df.groupby(["AmountData", "Method"])[
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
    df.groupby(["Method", "AmountData"])["Accuracy"].mean().round(3)
)
var_accuracy_by_dimension_and_method = df.groupby(["AmountData", "Method"])[
    "Accuracy"
].var()

order = ["add", "adapt", "online", "adjust"]


var_accuracy_by_dimension_and_method = var_accuracy_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)

print(mean_accuracy_by_dimension_and_method)
# mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.transpose()
mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.unstack(
    "Method"
).reindex(columns=order)
print(mean_accuracy_by_dimension_and_method)
print(mean_accuracy_by_dimension_and_method["add"])
for i in mean_accuracy_by_dimension_and_method:
    values_list = [
        value for index, value in mean_accuracy_by_dimension_and_method[i].items()
    ]
    index_list = [
        index for index, value in mean_accuracy_by_dimension_and_method[i].items()
    ]

    print(index_list, values_list)
# mean_accuracy_by_dimension_and_method = mean_accuracy_by_dimension_and_method.apply(
#    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
# )
# latex_table = mean_accuracy_by_dimension_and_method.to_latex(index=True)

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed


sns.set_palette("husl")


for i in mean_accuracy_by_dimension_and_method:
    print(i)
    values_list = [
        value for index, value in mean_accuracy_by_dimension_and_method[i].items()
    ]
    index_list = [
        str(int(index * 100)) + "%"
        for index, value in mean_accuracy_by_dimension_and_method[i].items()
    ]

    var_list = [
        value for index, value in var_accuracy_by_dimension_and_method[i].items()
    ]
    # print(var_list)
    l = "VanillaHD"
    if i == "adapt":
        l = "AdaptHD"
    elif i == "online":
        l = "OnlineHD"
    elif i == "adjust":
        l = "OUR"
    print(index_list)
    sns.lineplot(x=index_list, y=values_list, label=l)
    # if i == 'adjust' or i == 'online':
    #    plt.fill_between(index_list, np.array(values_list) - np.array(var_list), np.array(values_list) + np.array(var_list), alpha=0.3)

    # print(index_list, values_list)
    # plt.plot(index_list, values_list, label=i)

plt.ylabel("Accuracy")
plt.xlabel("Partial train data")
plt.title("Partial data accuracy evaluation over all methods")
plt.legend()
plt.show()
