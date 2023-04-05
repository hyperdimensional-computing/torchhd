import matplotlib.pyplot as plt
import numpy as np

# Create some sample data
values1 = [0.855, 0.862, 0.830]
values2 = [0.858, 0.942, 0.914]
values3 = [0.914, 0.946, 0.935]
groups = ["ISOLET", "UCIHAR", "MNIST"]

# Set the width of each bar
bar_width = 0.25

# Create a figure and axis object
fig, ax = plt.subplots()

# Create the first set of bars
ax.bar(
    np.arange(len(groups)) - bar_width / 2,
    values1,
    width=bar_width,
    label="Base single pass",
)
ax.bar(
    np.arange(len(groups)) + bar_width / 2, values2, width=bar_width, label="OnlineHD"
)
ax.bar(
    np.arange(len(groups)) + bar_width * 3 / 2,
    values3,
    width=bar_width,
    label="Our version",
)

# Set the x-axis tick labels to the group names
ax.set_xticks(np.arange(len(groups)) + bar_width / 2)
ax.set_xticklabels(groups)
ax.set_xlabel("Encoding")
ax.set_ylabel("Accuracy")
# Add a legend
ax.legend()
plt.ylim([0.8, 1])

ax.set_title("OnlineHD vs Our version, on ISOLET, UCIHAR and MNIST")
# Display the plot
plt.show()
