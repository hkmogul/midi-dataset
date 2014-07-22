import sklearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def normalize_matrix(X):
  ''' Returns matrix with each column normalized to have mean 0 and standard dev 1 '''
  new_X = np.copy(X)
  col_mean = np.mean(X, axis = 0)
  col_std = np.std(X, axis = 0)
  # subtract mean from each
  for i in xrange(X.shape[0]):
    new_X[i,:] = np.divide(np.subtract(X[i,:], col_mean), col_std)

  return new_X


mat_path = '../data/ML_info/X_and_y.mat'
data = scipy.io.loadmat(mat_path)
Xn = data['X']
y = data['y']
names = data['names']
clf = RandomForestClassifier()
X = normalize_matrix(Xn)
print X.shape
print y.shape
clf.fit(X[:-1], y[:-1])
print clf.predict(X[-1])
print names[-1]
