import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn color palette to coolwarm_r
sns.set_palette("colorblind")

# Sample data
methods = ['VanillaHD', 'AdaptHD', 'OnlineHD', 'AdaptHDIter', 'OnlineHDIter', 'NeuralHD', 'DistHD', 'SparseHD', 'QuantHD', 'CompHD', 'LeHDC', 'SemiHD', 'Multicentroid']
accuracy = [0.702, 0.687, 0.730, 0.761, 0.760, 0.736, 0.770, 0.731, 0.685, 0.625, 0.755, 0.687, 0.498]  # Example accuracy values (0 to 1)
time = [3.243, 6.576, 5.482, 385.181, 401.009, 182.432, 273.944, 53.427, 37.89, 3.429, 154.879, 105.56, 97.575]  # Example time values (in seconds)

# Normalize time values for better comparison
max_time = max(time)
normalized_time = [t / max_time * 100 for t in time]

# Sort data by accuracy in descending order
sorted_indices = sorted(range(len(time)), key=lambda k: time[k], reverse=True)
methods = [methods[i] for i in sorted_indices]
accuracy = [accuracy[i] for i in sorted_indices]
normalized_time = [normalized_time[i] for i in sorted_indices]

# Create subplots
fig, ax1 = plt.subplots()

# Create bar plot for accuracy (using a specific color)
bar_color = sns.color_palette()[0]  # Use the first color from the palette
ax1.bar(normalized_time, normalized_time, color=bar_color, label='Time')
ax1.set_xlabel('Methods')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='y')

# Create secondary y-axis for time (using a different color)
line_color = sns.color_palette()[1]  # Use the third color from the palette
ax2 = ax1.twinx()
ax2.scatter(methods, accuracy, color=line_color, marker='o', label='Accuracy')

for idx, model in enumerate(methods):
    acc = accuracy[idx]
    if model == 'Multicentroid':
        ax2.annotate(model, (idx, acc+0.058), rotation=0, ha='center', va='bottom')
    elif model == 'OnlineHDIter':
        ax2.annotate(model, (idx, acc+0.010), rotation=0, ha='center', va='bottom')
    else:
        ax2.annotate(model, (idx, acc+0.005), rotation=0, ha='center', va='bottom')

# Title and layout adjustments
plt.title('Comparison of Accuracy and Training Time Classification models UCI Benchmark')
plt.xticks(rotation=45, ha='right')
ax1.set_xticklabels(methods, rotation=45, ha='right')

# Display legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left')

# Show the plot
plt.tight_layout()
plt.show()
