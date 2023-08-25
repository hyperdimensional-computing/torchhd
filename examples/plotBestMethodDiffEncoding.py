import pandas as pd
import matplotlib.pyplot as plt

# Load data from csv file
data = pd.read_csv("results/best_encodings_methods_comparison.csv")

# Get unique encoding values
encodings = data["Encoding"].unique()

# Create subplots for each encoding
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))

# Loop through each encoding and create a bar plot with error bars
for i, encoding in enumerate(encodings):
    # Filter data by encoding and retrain parameters
    encoding_data = data[(data["Encoding"] == encoding) & (data["Retrain"] == False)]

    # Calculate mean and variance for each name/method combination
    grouped_data = encoding_data.groupby(["Name", "Method"])["Accuracy"].agg(
        ["mean", "var"]
    )

    # Reshape the data to a pivot table
    pivot_data = grouped_data.unstack()

    # Create bar plot with error bars
    pivot_data["mean"].plot(
        ax=axs[i // 3][i % 3], kind="bar", yerr=pivot_data["var"], capsize=3
    )
    axs[i // 3][i % 3].set_title(
        f"Accuracy by Name and Method (Encoding: {encoding}, Retrain: False)"
    )
    axs[i // 3][i % 3].set_xlabel("Name")
    axs[i // 3][i % 3].set_ylabel("Accuracy")
    axs[i // 3][i % 3].legend(title="Method")

plt.tight_layout()
plt.show()
