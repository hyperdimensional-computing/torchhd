import matplotlib.pyplot as plt

# Fictional data (replace this with actual accuracy data for your models)
models = [
    'VanillaHD',
    'AdaptHD',
    'OnlineHD',
    'AdaptHD Iter',
    'OnlineHD Iter',
    'NeuralHD',
    'DistHD',
    'SparseHD',
    'QuantHD',
    'CompHD',
    'LeHDC',
    'SemiHD',
    'Multicentroid',
]
years_published = [
    2016,
    2019,
    2021,
    2019,
    2021,
    2021,
    2023,
    2019,
    2020,
    2019,
    2022,
    2019,
    2022

]
accuracies = {
    'VanillaHD': 0.677,
    'AdaptHD': 0.693,
    'OnlineHD': 0.703,
    'AdaptHD Iter': 0.724,
    'OnlineHD Iter': 0.725,
    'NeuralHD': 0.718,
    'DistHD': 0.725,
    'SparseHD': 0.714,
    'QuantHD': 0.594,
    'CompHD': 0.346,
    'LeHDC': 0.704,
    'SemiHD': 0.676,
    'Multicentroid': 0.375,
}


positions = {
    'VanillaHD': 0.677,
    'AdaptHD': 0.690,
    'OnlineHD': 0.699,
    'AdaptHD Iter': 0.726,
    'OnlineHD Iter': 0.726,
    'NeuralHD': 0.715,
    'DistHD': 0.722,
    'SparseHD': 0.710,
    'QuantHD': 0.594,
    'CompHD': 0.346,
    'LeHDC': 0.703,
    'SemiHD': 0.678,
    'Multicentroid': 0.372,
}





fig, axs = plt.subplots(1, 2, figsize=(40, 5))

# Create the plot
# plt.figure(figsize=(10, 6))
ax = axs[0]

marker_styles = ['o', 's', '^', 'p', 'd', 'v', '>', '<', 'h', 'x', '+', '*', 'D', '|']

# Plot the accuracy points for each model
for idx, model in enumerate(models):
    accuracy = accuracies[model]
    position = positions[model]
    years_after_publication = [years_published[models.index(model)]]
    accuracies_over_time = [accuracy] * len(years_after_publication)
    ax.scatter(years_after_publication, accuracies_over_time, marker='o', label=model, color='grey')
    ax.annotate(model,(years_published[models.index(model)]+0.1, position-0.002))

# Set the plot title and labels
ax.set_title('Models HDC Arena accuracy over time')
ax.set_xlabel('Year')
ax.set_ylabel('Accuracy')

# Set the x-axis limits to start from the first year of publication
ax.set_xlim(years_published[0]-1, 2025)

# Show the grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add a legend
# plt.legend()

# Show the plot
ax = axs[1]

models = [
    'VanillaHD',
    'AdaptHD',
    'OnlineHD',
    'AdaptHD Iter',
    'OnlineHD Iter',
    'NeuralHD',
    'DistHD',
    'SparseHD',
    'QuantHD',
    'CompHD',
    'RVFL',
    'LeHDC',
    'SemiHD',
    'Multicentroid',
]
years_published = [
    2016,
    2019,
    2021,
    2019,
    2021,
    2021,
    2023,
    2019,
    2020,
    2019,
    2019,
    2022,
    2019,
    2022

]




accuracies = {
    'VanillaHD': 0.702,
    'AdaptHD': 0.687,
    'OnlineHD': 0.730,
    'AdaptHD Iter': 0.761,
    'OnlineHD Iter': 0.760,
    'NeuralHD': 0.736,
    'DistHD': 0.770,
    'SparseHD': 0.731,
    'QuantHD': 0.685,
    'CompHD': 0.625,
    'RVFL': 0.739,
    'LeHDC': 0.755,
    'SemiHD': 0.687,
    'Multicentroid': 0.498,
}

positions = {
    'VanillaHD': 0.702,
    'AdaptHD': 0.694,
    'OnlineHD': 0.726,
    'AdaptHD Iter': 0.761,
    'OnlineHD Iter': 0.764,
    'NeuralHD': 0.736,
    'DistHD': 0.770,
    'SparseHD': 0.731,
    'QuantHD': 0.685,
    'CompHD': 0.625,
    'RVFL': 0.739,
    'LeHDC': 0.755,
    'SemiHD': 0.685,
    'Multicentroid': 0.498,
}
# Create the plot
marker_styles = ['o', 's', '^', 'p', 'd', 'v', '>', '<', 'h', 'x', '+', '*', 'D', '|']

# Plot the accuracy points for each model
for idx, model in enumerate(models):
    accuracy = positions[model]
    position = positions[model]
    years_after_publication = [years_published[models.index(model)]]
    accuracies_over_time = [accuracy] * len(years_after_publication)
    ax.scatter(years_after_publication, accuracies_over_time, marker='o', label=model, color='grey')
    ax.annotate(model,(years_published[models.index(model)]+0.1, position-0.004))

# Set the plot title and labels
ax.set_title('Models UCI Benchmark accuracy over time')
ax.set_xlabel('Year')
ax.set_ylabel('Accuracy')

# Set the x-axis limits to start from the first year of publication
ax.set_xlim(years_published[0]-1, 2025)

# Show the grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add a legend
# plt.legend()

# Show the plot
plt.show()