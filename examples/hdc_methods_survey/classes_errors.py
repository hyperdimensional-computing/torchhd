import json
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.simplefilter(action="ignore", category=FutureWarning)


with open("results/embeddings_comparison/missclassified_samples.json") as f:
    f = json.load(f)
    missclassifed_data = f
    total_missclassified = {}
    for i in f:
        total_missclassified[i] = {}
        for j in f[i]:
            total_missclassified[i][j] = 0
            for k in f[i][j]:
                total_missclassified[i][j] += f[i][j][k]

    normalized_misses = {}
    for i in f:
        normalized_misses[i] = {}
        for j in f[i]:
            normalized_misses[i][j] = {}
            for k in f[i][j]:
                normalized_misses[i][j][k] = 0
                if total_missclassified[i][j] == 0:
                    normalized_misses[i][j][k] = 0
                else:
                    normalized_misses[i][j][k] = f[i][j][k] / total_missclassified[i][j]


with open("results/embeddings_comparison/train_samples.json") as f:
    f = json.load(f)

    data = f
    total_value = {}
    dataset_names = []
    for i in f:
        total_value[i] = {}
        for j in f[i]:
            dataset_names.append(j)

            total_value[i][j] = 0
            for k in f[i][j]:
                total_value[i][j] += f[i][j][k]

    normalized_data = {}
    for i in f:
        normalized_data[i] = {}
        for j in f[i]:
            normalized_data[i][j] = {}
            for k in f[i][j]:
                if str(k) not in normalized_misses[i][j]:
                    normalized_misses[i][j][k] = 0
                    missclassifed_data[i][j][k] = 0
                normalized_data[i][j][k] = 0
                if total_value[i][j] == 0:
                    normalized_data[i][j][k] = 0
                else:
                    normalized_data[i][j][k] = f[i][j][k] / total_value[i][j]

df_misses = pd.DataFrame.from_dict(normalized_misses)
df_data = pd.DataFrame.from_dict(normalized_data)
df_amount_per_class = pd.DataFrame.from_dict(data)
df_missclassified = pd.DataFrame.from_dict(missclassifed_data)

df_misses = df_misses.sort_index(axis=1)
df_data = df_data.sort_index(axis=1)
df_amount_per_class = df_amount_per_class.sort_index(axis=1)
df_missclassified = df_missclassified.sort_index(axis=1)

for i in range(len(df_misses)):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(str(dataset_names[i]), fontsize=16)
    print(df_misses.iloc[i].apply(pd.Series).fillna(0).sort_index(axis=1))
    print(df_data.iloc[i].apply(pd.Series).fillna(0).sort_index(axis=1))
    sns.heatmap(
        df_misses.iloc[i].apply(pd.Series).fillna(0).sort_index(axis=1),
        fmt="g",
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("blend:#FFF,#000", as_cmap=True),
        ax=ax[0],
        annot=df_missclassified.iloc[i].apply(pd.Series).fillna(0).sort_index(axis=1),
    )
    sns.heatmap(
        df_data.iloc[i].apply(pd.Series).fillna(0).sort_index(axis=1),
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("blend:#FFF,#000", as_cmap=True),
        ax=ax[1],
        annot=True,
        yticklabels=False,
        xticklabels=df_amount_per_class.iloc[i].apply(pd.Series).fillna(0).iloc[0],
    )

    plt.show()
