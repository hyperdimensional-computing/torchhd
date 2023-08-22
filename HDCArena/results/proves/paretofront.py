import matplotlib.pyplot as plt


methods = ['VanillaHD', 'AdaptHD', 'OnlineHD', 'AdaptHDIter', 'OnlineHDIter', 'NeuralHD', 'DistHD', 'SparseHD', 'QuantHD', 'CompHD', 'LeHDC', 'SemiHD']
accuracy = [0.677, 0.693, 0.703, 0.724, 0.725, 0.718, 0.725, 0.714, 0.594, 0.346, 0.704, 0.676]  # Example accuracy values (0 to 1)
time = [90.334, 145.7, 158.32, 8662.54, 8618.98, 10989.987, 5511.652, 933.503, 958.36, 95.03, 4000.0, 2868.40]  # Example time values (in seconds)

y_position = [90.334, 745.7, 758.32, 8662.54, 8618.98, 10989.987, 5511.652, 933.503, 958.36, 95.03, 4000.0, 2868.40]  # Example time values (in seconds)

# Create a Pareto front plot
plt.figure(figsize=(10, 6))
max_time = max(time)
time = [t / max_time * 100 for t in time]
y_position = [t / max_time * 100 for t in y_position]
# Sort data based on the Pareto principle (higher accuracy and lower time preferred)
sorted_indices = sorted(range(len(time)), key=lambda k: -time[k], reverse=True)
#time = [time[i] for i in sorted_indices]

sorted_methods = [methods[i] for i in sorted_indices]
sorted_time = [time[i] for i in sorted_indices]
sorted_accuracy = [accuracy[i] for i in sorted_indices]
x_position = [0.645, 0.670, 0.685, 0.695, 0.718, 0.718, 0.718, 0.714, 0.594, 0.346, 0.704, 0.676]  # Example accuracy values (0 to 1)
sorted_x_position = [x_position[i] for i in sorted_indices]
sorted_y_position = [y_position[i] for i in sorted_indices]
# Scatter plot
plt.scatter(sorted_time, sorted_accuracy, marker='o')

# Add labels to points
for i, label in enumerate(sorted_methods):
    plt.annotate(label, (sorted_y_position[i], sorted_x_position[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.gca().invert_xaxis()

# Set plot labels and title
plt.xlabel('Normalized Time')
plt.ylabel('Accuracy')
plt.title('Pareto plot HDC Arena')

plt.grid(True)
plt.show()

