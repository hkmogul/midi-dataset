import sklearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import cross_validation


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
yn = data['y']
names = data['names']
clf = RandomForestClassifier()
X = normalize_matrix(Xn)
y = np.reshape(yn, (yn.shape[0],))


# clf.fit(X[0:X.shape[0]-5], y[0:y.shape[0]-5])
clf.fit(X,y)
# print clf.predict(X)
# print names[-2]
print "SCORE OF WHOLE FIT"
print clf.score(X,y)
print "----------"
test_size = 0.2
print "SCORE OF TEST/RANDOM FIT; TEST SIZE = {}".format(test_size)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                    y, test_size=test_size, random_state=0)
clf2 = RandomForestClassifier().fit(X_train, y_train)
print clf2.score(X_test, y_test)
