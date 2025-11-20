import numpy as np
import pandas as pd

data = pd.read_csv("pts_ciudad/cloud_000447.xyz",header=None)
data2 = np.load("reflectivity_000447.npy").reshape(-1,1)
print(data2.shape,data.shape)
print(data.head())