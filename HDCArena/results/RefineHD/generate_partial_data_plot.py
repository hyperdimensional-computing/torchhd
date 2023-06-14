import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file
from scipy.interpolate import make_interp_spline

file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/partial_data_arena"

df = pd.read_csv(file)
# Extract the required columns
print(df)
x = list(map(int, df.columns[1:]))
# Increase the number of data points for smoother lines
x_smooth = np.linspace(min(x), max(x), 200)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

lab = [
    "Baseline",
    "AdaptHD",
    "OnlineHD",
    "RefineHD",
    "AdaptHD Iterative",
    "OnlineHD Iterative",
    "RefineHD Iterative",
]
for i, (_, row) in enumerate(df.iterrows()):
    l = row["method"]
    row = row.drop("method")
    y2_smooth = make_interp_spline(x, row, k=1)(x_smooth)
    axes[0].plot(x_smooth, y2_smooth, label=lab[i])


file = (
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/partial_data"
)

df = pd.read_csv(file)
# Extract the required columns
# Extract the required columns
x = list(map(int, df.columns[1:]))

# Increase the number of data points for smoother lines
x_smooth = np.linspace(min(x), max(x), 200)
# Increase the number of data points for smoother lines

for i, (_, row) in enumerate(df.iterrows()):
    if i < len(lab):
        l = row["method"]
        row = row.drop("method")
        y2_smooth = make_interp_spline(x, row, k=1)(x_smooth)

        axes[1].plot(x_smooth, y2_smooth, label=lab[i])

axes[0].title.set_text("HDC Benchmark")
axes[1].title.set_text("UCI Benchmark")
# Customize the plot
axes[0].set_xlabel("Data amount (%)")
axes[1].set_xlabel("Data amount (%)")
axes[0].set_ylabel("Accuracy")
axes[1].set_ylabel("Accuracy")
fig.suptitle("Partial data accuracy evaluation over all methods")
plt.tight_layout()
axes[0].legend(loc="lower right")

# Show the plot
plt.show()
