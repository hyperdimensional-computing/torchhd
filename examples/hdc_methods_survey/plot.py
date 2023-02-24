import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file = "results/openlab.csv"
data = pd.read_csv(file)
accuracy = []
first = True
df = None
for i in data["Method"].unique():
    if first:
        df = pd.DataFrame({i: np.array(data[data["Method"] == i]["Accuracy"])})
        df.index = np.array(data[data["Method"] == i]["Name"])
        first = False
    else:
        df[i] = np.array(data[data["Method"] == i]["Accuracy"])
    accuracy.append(data[data["Method"] == i]["Accuracy"])


# print(df.shape[1])
df2 = pd.DataFrame(0, index=np.arange(df.shape[0]), columns=df.columns)
for index, (i, row) in enumerate(df.iterrows()):
    # print(row)
    maxes = np.flatnonzero(row == np.max(row))
    for j in maxes:
        df2.iloc[index][j] = 1

df2.index = df.index
configs_scores = []
configs = []
for i in df2:
    configs.append(i)
    configs_scores.append(sum(df2[i] == 1))
    # print(sum(df[i] == 1))
print(configs_scores)
print(np.argmax(configs_scores))
print(configs[np.argmax(configs_scores)], configs_scores[np.argmax(configs_scores)])

plt.figure(figsize=(15, 8))
sns.heatmap(df, cmap=sns.color_palette("blend:#FF0000,#06c258", as_cmap=True))
plt.show()

sns.heatmap(df2, cmap=sns.color_palette("blend:#FFF,#000", as_cmap=True))
plt.show()
