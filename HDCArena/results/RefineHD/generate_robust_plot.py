import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
from matplotlib import cm

file = (
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/arena_robust"
)

df = pd.read_csv(file)

x = list(map(int, df.columns[1:]))
# Generate more data points for smooth curves
x_smooth = np.linspace(min(x), max(x), 200)
# cmap = cm.get_cmap('Set1')

for i, (_, row) in enumerate(df.iterrows()):
    l = row["method"]
    row = row.drop("method")
    y2_smooth = make_interp_spline(x, row)(x_smooth)

    plt.plot(x_smooth, y2_smooth, label=l)
    # , color=cmap(i/len(x)))

# Add labels and title
plt.xlabel("Vector bit flips (%)")
plt.ylabel("Accuracy")
plt.title("Methods robustness")
plt.legend()
# Display the plot
plt.show()
