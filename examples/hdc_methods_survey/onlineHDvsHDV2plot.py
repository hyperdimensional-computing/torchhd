import matplotlib.pyplot as plt
import numpy as np

# Create some sample data
values1 = [0.725, 0.714, 0.719, 0.734]
values2 = [0.741, 0.724, 0.742, 0.753]
groups = ['Hashmap', 'RandomProj', 'SinusoidProj', 'Density']

# Set the width of each bar
bar_width = 0.35

# Create a figure and axis object
fig, ax = plt.subplots()

# Create the first set of bars
ax.bar(np.arange(len(groups)), values1, width=bar_width, label='OnlineHD')

# Create the second set of bars
ax.bar(np.arange(len(groups))+bar_width, values2, width=bar_width, label='Our version')

# Set the x-axis tick labels to the group names
ax.set_xticks(np.arange(len(groups))+bar_width/2)
ax.set_xticklabels(groups)
ax.set_xlabel('Encoding')
ax.set_ylabel('Accuracy')
# Add a legend
ax.legend()
plt.ylim([0.7, 0.78])

ax.set_title('OnlineHD vs Our version, average on 120 datasets')
# Display the plot
plt.show()