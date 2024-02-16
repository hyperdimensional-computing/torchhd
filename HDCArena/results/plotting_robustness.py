import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
from matplotlib import cm

df = pd.read_csv("/HDCArena/results/files/robustness_results")

print(df)
x = list(map(int, df.columns[1:]))
# Generate more data points for smooth curves
x_smooth = np.linspace(min(x), max(x), 200)
# cmap = cm.get_cmap('Set1')

for i, (_, row) in enumerate(df.iterrows()):
    l = row["model"]
    row = row.drop("model")
    y2_smooth = make_interp_spline(x, row, k=1)(x_smooth)

    plt.plot(x_smooth, y2_smooth, label=l)
    # , color=cmap(i/len(x)))

# Add labels and title
plt.xlabel("Vector bit flips (%)")
plt.ylabel("Accuracy")
plt.title("Methods robustness")
plt.legend()
# Display the plot
plt.show()
