import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file
from scipy.interpolate import make_interp_spline

file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/arena_dimensions"

df = pd.read_csv(file)
# Extract the required columns
x = df["dimensions"]

# Increase the number of data points for smoother lines
x_smooth = np.linspace(x.min(), x.max(), 300)
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
for idx, method in enumerate(range(len(lab))):
    a = df["method"].unique()
    method = a[idx]
    print("m", method)
    l = df[df["method"] == method]
    x = l["dimensions"]
    y_mean = l["accuracy"]
    y_upper = y_mean + l["variance"]  # Upper bound of variance
    y_lower = y_mean - l["variance"]  # Lower bound of variance
    # Create the plot
    y_smooth = make_interp_spline(x, y_mean, k=1)(x_smooth)

    # y_smooth = np.interp(x_smooth, x, y_mean)
    upper_smooth = make_interp_spline(x, y_upper, k=1)(x_smooth)
    lower_smooth = make_interp_spline(x, y_lower, k=1)(x_smooth)

    # Plot the mean line and fill the variance
    # plt.plot(x_smooth, y_smooth, label=method)
    # plt.fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.3)

    axes[0].plot(x_smooth, y_smooth, label=lab[idx])
    # axes[0].fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.25)

file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/uci_dimensions"

df = pd.read_csv(file)
# Extract the required columns
x = df["dimensions"]

# Increase the number of data points for smoother lines

lab = ["Baseline", "AdaptHD", "OnlineHD", "RefineHD", "AdaptHD", "OnlineHD", "RefineHD"]
for idx, method in enumerate(df["method"].unique()):
    print("m", method)
    l = df[df["method"] == method]
    x = l["dimensions"]
    y_mean = l["accuracy"]
    y_upper = y_mean + l["variance"]  # Upper bound of variance
    y_lower = y_mean - l["variance"]  # Lower bound of variance
    # Create the plot
    y_smooth = make_interp_spline(x, y_mean, k=1)(x_smooth)

    # y_smooth = np.interp(x_smooth, x, y_mean)
    upper_smooth = make_interp_spline(x, y_upper, k=1)(x_smooth)
    lower_smooth = make_interp_spline(x, y_lower, k=1)(x_smooth)

    # Plot the mean line and fill the variance
    # plt.plot(x_smooth, y_smooth, label=method)
    # plt.fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.3)

    axes[1].plot(x_smooth, y_smooth, label=lab[idx])
    # plt.fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.25)
axes[0].title.set_text("HDC Benchmark")
axes[1].title.set_text("UCI Benchmark")
# Customize the plot
axes[0].set_xlabel("Dimensions")
axes[1].set_xlabel("Dimensions")
axes[0].set_ylabel("Accuracy")
axes[1].set_ylabel("Accuracy")
fig.suptitle("Accuracy comparison of Adaptive models, using different dimensions")
plt.tight_layout()
axes[0].legend(loc="lower right")

# Show the plot
plt.show()
