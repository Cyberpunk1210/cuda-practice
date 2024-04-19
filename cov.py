import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA
from time import time

import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
W: int = 48; H: int = 32
D: ndarray = np.random.randn(W, H)
S: ndarray = np.random.randn(H, W)
shape_D, shape_S = D.shape, S.shape
print(f"D shape is: W {shape_D[0]} -- H {shape_D[1]}")
print(f"S shape is: W {shape_S[0]} -- H {shape_S[1]}")

# TODO principal component analysis






# signular value decomponent



# TODO confidenceEllipse
