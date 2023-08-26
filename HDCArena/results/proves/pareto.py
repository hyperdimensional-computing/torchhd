import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn color palette to coolwarm_r
sns.set_palette("colorblind")

# Sample data
methods = [
    "VanillaHD",
    "AdaptHD",
    "OnlineHD",
    "AdaptHDIter",
    "OnlineHDIter",
    "NeuralHD",
    "DistHD",
    "SparseHD",
    "QuantHD",
    "CompHD",
    "LeHDC",
    "SemiHD",
]
accuracy = [
    0.677,
    0.693,
    0.703,
    0.724,
    0.725,
    0.718,
    0.725,
    0.714,
    0.594,
    0.346,
    0.704,
    0.676,
]  # Example accuracy values (0 to 1)
time = [
    90.334,
    145.7,
    158.32,
    8662.54,
    8618.98,
    10989.987,
    5511.652,
    933.503,
    958.36,
    95.03,
    4000.0,
    2868.40,
]  # Example time values (in seconds)

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
ax1.bar(methods, normalized_time, color=bar_color, label="Time")
ax1.set_xlabel("Methods")
ax1.set_ylabel("Accuracy")
ax1.tick_params(axis="y")

# Create secondary y-axis for time (using a different color)
line_color = sns.color_palette()[1]  # Use the third color from the palette
ax2 = ax1.twinx()
ax2.scatter(methods, accuracy, color=line_color, marker="o", label="Accuracy")

for idx, model in enumerate(methods):
    acc = accuracy[idx]
    ax2.annotate(model, (idx, acc + 0.005), rotation=0, ha="center", va="bottom")

# Title and layout adjustments
plt.title("Comparison of Accuracy and Training Time Classification models in HDC Arena")
plt.xticks(rotation=45, ha="right")
ax1.set_xticklabels(methods, rotation=45, ha="right")

# Display legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="lower left")

# Show the plot
plt.tight_layout()
plt.show()
