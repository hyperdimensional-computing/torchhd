import matplotlib.pyplot as plt


methods = ['VanillaHD', 'AdaptHD', 'OnlineHD', 'AdaptHDIter', 'OnlineHDIter', 'NeuralHD', 'DistHD', 'SparseHD', 'QuantHD', 'CompHD', 'LeHDC', 'SemiHD', 'Multicentroid']
accuracy = [0.702, 0.687, 0.730, 0.761, 0.760, 0.736, 0.770, 0.731, 0.685, 0.625, 0.755, 0.687, 0.498]  # Example accuracy values (0 to 1)
time = [3.243, 6.576, 5.482, 385.181, 401.00, 182.43, 273.944, 53.427, 37.89, 3.429, 154.879, 105.56, 97.575]  # Example time values (in seconds)

y_position = [3.243, 6.576, 5.482, 385.181, 396.00, 182.43, 273.944, 53.427, 37.89, 3.429, 154.879, 105.56, 97.575]
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
x_position = [0.702, 0.667, 0.730, 0.761, 0.730, 0.736, 0.745, 0.731, 0.685, 0.625, 0.755, 0.687, 0.498]  # Example accuracy values (0 to 1)
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
plt.title('Pareto plot UCI Benchmark')

plt.grid(True)
plt.show()

