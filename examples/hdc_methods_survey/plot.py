import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv
file = "results/openlab.csv"
data = pd.read_csv(file)
accuracy = []
first = True
df = None

methods = ['MemoryModel', 'MemoryModelOnline']
df_new = {}
l = 0
for i in data["Method"].unique():
    if first:
        df = pd.DataFrame({i: np.array(data[data["Method"] == i]["Accuracy"])})
        df.index = np.array(data[data["Method"] == i]["Name"])
        first = False
        l = len(df.index)
        df_new['dataset'] = data[data["Method"] == i]['Name'].values
        df_new['classes'] = data[data["Method"] == i]['Classes'].values
    else:
        print(i)
        print(len(data[data["Method"] == i]["Accuracy"]))
        df[i] = np.array(data[data["Method"] == i]["Accuracy"])

    if i in methods:
        df_new[i] = data[data["Method"] == i]['Accuracy'].values
    accuracy.append(data[data["Method"] == i]["Accuracy"])

with open("comparison.csv", "w") as outfile:
    writer = csv.writer(outfile)
    key_list = list(df_new.keys())
    writer.writerow(df_new.keys())

    for i in range(l):
        writer.writerow([df_new[x][i] for x in key_list])


# print(df.shape[1])
df2 = pd.DataFrame(0, index=np.arange(df.shape[0]), columns=df.columns)
for index, (i, row) in enumerate(df.iterrows()):
    # print(row)
    maxes = np.flatnonzero(row == np.max(row))
    for j in maxes:
        df2.iloc[index][j] = 1
    if df.columns[np.argmax(row)] in ['MemoryModel','MemoryModelOnline']:
        print("MEMORY")
    if row.name == 'HayesRoth':
        print(row)
    print(row.name, np.max(row), np.argmax(row), df.columns[np.argmax(row)])

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
#plt.show()

sns.heatmap(df2, cmap=sns.color_palette("blend:#FFF,#000", as_cmap=True))
#plt.show()
