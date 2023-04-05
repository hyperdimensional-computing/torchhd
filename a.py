import scipy.io
import pandas as pd

mat = scipy.io.loadmat('S1_A1_E3.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()}) # compatible for both python 2.x and python 3.x

data.to_csv("example.csv")