import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv('results/results1681161541.596628.csv')

methods_order = ['bundle','sequence','ngram','hashmap','density','flocet','random','sinusoid']
methods_order = ['add', 'add_adapt', 'add_online','add_adjust']
# Create the first plot
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby(['Name', 'Method'])['Accuracy'].mean().unstack().reindex(methods_order, axis=1).plot(kind='bar', ax=ax, width=0.8)

print(df.groupby(['Name', 'Method'])['Accuracy'].mean().unstack().reindex(methods_order, axis=1))
ax.set_xlabel('Method and Dataset Name')
ax.set_ylabel('Accuracy')
ax.legend(title='Dataset Name', loc='upper left')
plt.title('Accuracy by Method and Dataset')

# Create the second plot
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby(['Name', 'Method'])['Time'].mean().unstack().reindex(methods_order, axis=1).plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Method and Dataset Name')
ax.set_ylabel('Time')
ax.legend(title='Dataset Name', loc='upper right')
plt.title('Time by Method and Dataset')

# Show the plots
plt.show()
df_mean = df.groupby(['Method', 'Name'])['Accuracy'].mean().to_frame()
print(df_mean)
df_pivot = df_mean.reset_index().pivot(index='Name', columns='Method', values='Accuracy')
df_pivot.loc['mean'] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
# Print the new DataFrame to the console

latex_table = df_pivot.to_latex(index=True)

# Print the LaTeX table to the console
print(latex_table)


df_mean = df.groupby(['Method', 'Name'])['Time'].mean().to_frame()
print(df_mean)
df_pivot = df_mean.reset_index().pivot(index='Name', columns='Method', values='Time')
df_pivot.loc['mean'] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
# Print the new DataFrame to the console

latex_table = df_pivot.to_latex(index=True)

# Print the LaTeX table to the console
print(latex_table)