import sklearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def normalize_matrix(X):
  ''' Returns matrix with each column normalized to have mean 0 and standard dev 1 '''
  new_X = np.copy(X)


mat_path = '../data/ML_info/X_and_y.mat'
data = scipy.io.loadmat
