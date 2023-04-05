import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv
#file = "results/results1678375052.90958.csv"
file = "results/hasmap_iterative.csv"
data = pd.read_csv(file)
accuracy = []
first = True
df = None

df_new = {}
l = 0

col = 'Method'

title = 'Online HD, vanilla vs new '

for i in data[col].unique():
    if first:
        df = pd.DataFrame({i: np.array(data[data[col] == i]["Accuracy"])})
        df.index = np.array(data[data[col] == i]["Name"])
        first = False
        l = len(df.index)
        df_new['dataset'] = data[data[col] == i]['Name'].values
        df_new['classes'] = data[data[col] == i]['Classes'].values
    else:
        print(i)
        print(len(data[data[col] == i]["Accuracy"]))
        df[i] = np.array(data[data[col] == i]["Accuracy"])

    print(data[data[col] == i]["Accuracy"].sum()/121)


    #if i in methods:
    df_new[i] = data[data[col] == i]['Accuracy'].values
    accuracy.append(data[data[col] == i]["Accuracy"])

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
    #if df.columns[np.argmax(row)] in ['MemoryModel','MemoryModelOnline']:
    #    print("MEMORY")
    #if row.name == 'HayesRoth':
    #    print(row)
    #print(row.name, np.max(row), np.argmax(row), df.columns[np.argmax(row)])

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

num_rows = 40
l = len(df.columns)

print(df.columns)
new_df = pd.DataFrame(data[data[col] == "HashmapProjectionOnline"]["Accuracy"])
print()
df22 = pd.DataFrame(data[data[col] == "HashmapProjectionOnlineV2"]["Accuracy"]).reset_index()['Accuracy']
new_df["V2"] = df22
#print(new_df)

def highlight_max(row):
    return ['background-color: yellow' if val == row.max() else '' for val in row]

# Apply the function to the DataFrame using the `style.apply` method
styled_df = new_df.style.apply(highlight_max, axis=1)

# Display the styled DataFrame
print(styled_df)

splits = [[0,40],[40,80],[80,122]]

for idx, i in enumerate(splits):
    rows = data["Name"].unique()[i[0]:i[1]]
    cols = df.columns


    heat = df.to_numpy()[i[0]:i[1]]
    heat = heat.T




    fig, ax = plt.subplots()
    im = ax.imshow(heat, aspect=3, cmap='viridis')

    ax.set_xticks(np.arange(len(rows)), labels=rows)
    ax.set_yticks(np.arange(len(cols)), labels=cols)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    for l in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(l, j, round(heat[j, l], 2),
                           ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax)
    fig.set_size_inches(22,8)
    ax.set_title(title + " part " + str(idx))
    fig.subplots_adjust(
        top=0.981,
        bottom=0.049,
        left=0.05,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )

    #plt.show()

    #fig.tight_layout()
    #plt.show()
    #fig.savefig(title + " part " + str(idx), dpi = 200)



    best = df2.to_numpy()[i[0]:i[1]]
    best = best.T
    print(best)


    fig, ax = plt.subplots()
    im = ax.imshow(best, aspect=3, cmap='binary')

    ax.set_xticks(np.arange(len(rows)), labels=rows)
    ax.set_yticks(np.arange(len(cols)), labels=cols)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    fig.set_size_inches(22, 8)
    ax.set_title(title + " part " + str(idx))
    fig.subplots_adjust(
        top=0.981,
        bottom=0.2,
        left=0.05,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )

    # plt.show()

    # fig.tight_layout()

    #fig.savefig("Best" + title + "part " + str(idx), dpi=200)

'''
plt.figure(figsize=(15, 8))
sns.heatmap(df, cmap=sns.color_palette("blend:#FF0000,#06c258", as_cmap=True))
plt.show()

sns.heatmap(df2, cmap=sns.color_palette("blend:#FFF,#000", as_cmap=True))
plt.show()
'''