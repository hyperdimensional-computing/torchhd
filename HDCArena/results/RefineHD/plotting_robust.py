import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
from matplotlib import cm

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/robustness_results_arena"
)

x = list(map(int, df.columns[1:11]))
# Generate more data points for smooth curves
x_smooth = np.linspace(min(x), max(x), 200)
# cmap = cm.get_cmap('Set1')

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

linestyle_str = [
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('solid', 'solid'),  # Same as (0, (1, 1)) or ':'
    ('solid', 'solid'),  # Same as '--'
    ('solid', 'solid')
]  # Same as '-.'

markers = [
    ",",
    ".",
    "*",
    "P",
    ".",
    "*",
    "P",
]

for i, (_, row) in enumerate(df.iterrows()):
    l = row["method"]
    row = row.drop("method")
    y2_smooth = make_interp_spline(x, row[:10], k=1)(x_smooth)

    print(linestyle_str[i][0])
    plt.plot(x_smooth, y2_smooth, linestyle=linestyle_str[i][0], marker=markers[i],markevery=10, label=l)
    # , color=cmap(i/len(x)))

# Add labels and title
plt.xlabel("Vector bit flips (%)")
plt.ylabel("Accuracy")
plt.title("Methods robustness")
plt.legend()
# Display the plot
plt.show()
