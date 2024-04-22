import os
from os.path import join, expanduser

import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA
import pandas as pd

DATA_PATH = expanduser(os.environ.get("UTILS_DATA"))


W: int = 48; H: int = 32
D: ndarray = np.random.randn(W, H)
S: ndarray = np.random.randn(H, W)
shape_D, shape_S = D.shape, S.shape
print(f"D shape is: W {shape_D[0]} -- H {shape_D[1]}")
print(f"S shape is: W {shape_S[0]} -- H {shape_S[1]}")

# TODO principal component analysis
data = pd.read_csv(join(DATA_PATH, "nyc_taxi.csv"), parse_dates=["timestamp"])
np.random.seed(1000)

# dummy data
_d = np.random.rand(100, 4)
n_samples, feature = _d.shape
top_k = int(feature * 0.8)

# acculate mean value for origin matrix
# meanval = np.mean(_d, axis=0)
meanval = np.array([sum(_d[:, idx])/n_samples for idx in range(feature)])

# decentralization
norm_d = _d - meanval

# acculate covariance matrix
# cov_mat = 1 * (x - x^)(y - y^) / n --> d.T @ d
scatter_mat = (norm_d.T @ norm_d) / (n_samples - 1)

# accluate feature value and vector
eig_val, eig_vec = np.linalg.eig(scatter_mat)

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(feature)]
eig_pairs.sort(reverse=True)
n_features = np.array([ele[1] for ele in eig_pairs[:top_k]])

pca_d = norm_d @ n_features.T
print(pca_d.shape)

# signular value decomponent

U, s, Vh = np.linalg.svd(scatter_mat)
sort_index = np.argsort(s)[::-1]
eieg_vecs = U[:, sort_index[:top_k]]
pca_svd = norm_d @ eieg_vecs
print(pca_svd.shape)

# TODO confidenceEllipse
