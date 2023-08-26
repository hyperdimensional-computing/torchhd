import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv("../results/final_results/generic.csv")

methods_order = [
    "generic",
]
var = "Encoding"
# methods_order = ["add"]
# Create the first plot
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby(["Name", var])["Accuracy"].mean().unstack().reindex(
    methods_order, axis=1
).plot(kind="bar", ax=ax, width=0.8)

print(
    df.groupby(["Name", var])["Accuracy"]
    .mean()
    .unstack()
    .reindex(methods_order, axis=1)
)
ax.set_xlabel(var + " and Dataset Name")
ax.set_ylabel("Accuracy")
ax.legend(title="Dataset Name", loc="upper left")
plt.title("Accuracy by " + var + " and Dataset")

# Create the second plot
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby(["Name", var])["TrainTime"].mean().unstack().reindex(
    methods_order, axis=1
).plot(kind="bar", ax=ax, width=0.8)
ax.set_xlabel(var + " and Dataset Name")
ax.set_ylabel("TrainTime")
ax.legend(title="Dataset Name", loc="upper right")
plt.title("v by " + var + " and Dataset")

# Show the plots
plt.show()
df_mean = df.groupby([var, "Name"])["Accuracy"].mean().to_frame()
print(df_mean)
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="Accuracy")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
# Print the new DataFrame to the console
column_names = df_pivot.columns.tolist()
column_dict = {column_name: 0 for column_name in column_names}


def count_max(row):
    max_val = row.max()
    maxx = list(row[row == max_val].index)
    for i in maxx:
        column_dict[i] += 1


# apply the count_max function to each row of the DataFrame
max_counts = df_pivot.apply(count_max, axis=1)

# print the resulting counts
print(column_dict)

df_pivot = df_pivot.apply(
    lambda row: row.replace(row.max(), "bof(" + str(row.max()) + ")"), axis=1
)

latex_table = df_pivot.to_latex(index=True)

# Print the LaTeX table to the console
print(latex_table)


df_mean = df.groupby([var, "Name"])["TrainTime"].mean().to_frame()
print(df_mean)
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values="TrainTime")
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
# Print the new DataFrame to the console


# print the resulting counts
# print(max_counts)
# display the results

df_pivot = df_pivot.apply(
    lambda row: row.replace(row.min(), "bof(" + str(row.min()) + ")"), axis=1
)
latex_table = df_pivot.to_latex(index=True)

# Print the LaTeX table to the console
print(latex_table)
