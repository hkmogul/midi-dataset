import sklearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm

import sys
sys.path.append('../')
import ml_utility as ml



mat_path = '../data/ML_info-no_repeats/X_and_y.mat'
data = scipy.io.loadmat(mat_path)
Xn = data['X']
print Xn.shape
yn = data['y']
names = data['names']
trees = 150

clf = RandomForestClassifier(n_estimators = trees)
X = ml.normalize_matrix(Xn)
y = np.reshape(yn, (yn.shape[0],))


clf.fit(X,y)

print "SCORE OF WHOLE FIT"
print clf.score(X,y)
print "----------"
test_size = .2
clf_svm = svm.SVC(gamma = .001, C = 200)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                    y, test_size=test_size, random_state=0)
clf_svm.fit(X_train, y_train)
print "SCORE OF CROSS VAL SVC: {}".format(clf_svm.score(X_test, y_test))
# cross validation
print "SCORE OF TEST/RANDOM FIT RANDOM FOREST; TEST SIZE = {}".format(test_size)


X_train, X_test, y_train, y_test, names_train, names_test = ml.cross_val_with_names(X, y, names, test_amt = test_size, random_seed = None)
# single cross validation
print "NUMBER OF TREES: {}".format(trees)
clf2 = RandomForestClassifier(n_estimators = trees).fit(X_train, y_train)
print "SCORE OF CROSS VALIDATION, {0} TRAINING SAMPLES: {1}".format(
                                                      y_train.shape[0],
                                                      clf2.score(X_test, y_test))
y_pred = clf2.predict(X_test)
proba = clf2.predict_proba(X_test)
proba_y = ml.find_prob(y_test, proba)
precision_curve, recall_curve, thresholds = metrics.precision_recall_curve(y_test, proba_y, pos_label = 1)
area = metrics.auc(recall_curve, precision_curve)
print "AUC: {}".format(area)
# ml.util_print(y_pred, y_test, proba, names_test, save = True)
# ml.pretty_print_prob(y_pred, y_test, proba, names_test)
print "----- \n False Positive Printing"
ml.print_false_positives(y_pred, y_test, proba, names_test)

# include precision recall
prec_arr, recall_arr, fbeta_arr, support_arr = metrics.precision_recall_fscore_support(
                                                                y_test,
                                                                y_pred,
                                                                labels = None,
                                                                beta = .5
                                                                )
precision, recall, fbeta_score = prec_arr[1], recall_arr[1], fbeta_arr[1]

print "NUMBER OF TESTING SAMPLES: {}".format(y_test.shape[0])
print "F-beta of initial cross validation: {}".format(fbeta_score)
print "Precision of initial cross validation: {}".format(precision)
print "Recall of initial cross validation: {}".format(recall)
fp, fn = ml.get_fp_fn(y_pred, y_test)
print "Number of false positives: {}".format(fp)
print "Number of false negatives: {}".format(fn)
prec, rec = ml.get_precision_recall(y_pred, y_test)
# print "Self calculated precision: {}".format(prec)
# print "Self calculated recall: {}".format(rec)
print "----------"







# multiple cross validation
amt = 50

print "BEGINNING MULTIPLE CROSS VALIDATION, ITERATIONS = {}".format(amt)
print "-----"

scores = np.empty((amt,))
false_pos = np.empty((amt,))
false_neg = np.empty((amt,))
precision_arr = np.empty((amt,))
recall_arr = np.empty((amt,))
fb_arr = np.empty((amt,))
amt_success = np.empty((amt,)) # amount of test items the ML deems = 1

for i in xrange(amt):
  print "----- \n Iteration # {}".format(i)
  # use the index to generate random state because why not
  X_train, X_test, y_train, y_test, names_train, names_test = ml.cross_val_with_names(X,
                                      y, names, test_amt = test_size)

  clf_mult = RandomForestClassifier(n_estimators = trees).fit(X_train, y_train)

  y_pred = clf_mult.predict(X_test)
  # get score instead of calculating
  s = clf_mult.score(X_test, y_test)
  prec_arr, rec_arr, fbeta_arr, support_arr = metrics.precision_recall_fscore_support(
                                                                  y_test,
                                                                  y_pred,
                                                                  labels = None,
                                                                  beta = .5
                                                                  )
  precision, recall, fbeta_score = prec_arr[1], rec_arr[1], fbeta_arr[1]
  fp, fn = ml.get_fp_fn(y_pred, y_test)
  # assign values to all arrays
  proba = clf2.predict_proba(X_test)
  ml.print_false_positives(y_pred, y_test, proba, names_test)

  scores[i] = s
  false_pos[i] = fp
  false_neg[i] = fn
  precision_arr[i] = precision
  recall_arr[i] = recall
  fb_arr[i] = fbeta_score
  amt_success[i] = np.argwhere(y_pred).shape[0]


print "Amount of test datapoints per interation: {}".format(y_test.shape[0])
print "AVERAGE SCORE IN MULTIPLE: {}".format(np.mean(scores))
print "Maximum score: {}".format(np.amax(scores))
print "AVERAGE AMT OF FALSE POSITIVES: {}".format(np.mean(false_pos))
print "AVERAGE PRECISION: {}".format(np.mean(precision_arr))
print "RATE OF FALSE POSITIVES PER TEST SIZE: {}".format(np.mean(false_pos)/y_test.shape[0])
print "RATE OF FALSE POSITIVES PER AMT OF POSITIVES: {}".format(np.mean(false_pos)/np.mean(amt_success))
print "AVERAGE RECALL: {}".format(np.mean(recall_arr))
print "AVERAGE F-BETA: {}".format(np.mean(fbeta_arr))




print "----------"
# print "Repeating cross validation experiment for SVC"
# for i in xrange(amt):
#   # use the index to generate random state because why not
#   X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
#                                       y, test_size=test_size, random_state=i)
#
#   clf_mult = svm.SVC(gamma = .001, C = 100).fit(X_train, y_train)
#
#   y_pred = clf_mult.predict(X_test)
#   # get score instead of calculating
#   s = clf_mult.score(X_test, y_test)
#   prec_arr, rec_arr, fbeta_arr, support_arr = metrics.precision_recall_fscore_support(
#                                                                   y_test,
#                                                                   y_pred,
#                                                                   labels = None,
#                                                                   beta = .5
#                                                                   )
#   precision, recall, fbeta_score = prec_arr[1], rec_arr[1], fbeta_arr[1]
#   fp, fn = ml.get_fp_fn(y_pred, y_test)
#   # assign values to all arrays
#
#   scores[i] = s
#   false_pos[i] = fp
#   false_neg[i] = fn
#   precision_arr[i] = precision
#   recall_arr[i] = recall
#   fb_arr[i] = fbeta_score
#   amt_success[i] = np.argwhere(y_pred).shape[0]
#
#
#
# print "AVERAGE SCORE IN MULTIPLE: {}".format(np.mean(scores))
# print "Maximum score: {}".format(np.amax(scores))
# print "AVERAGE AMT OF FALSE POSITIVES: {}".format(np.mean(false_pos))
# print "AVERAGE PRECISION: {}".format(np.mean(precision_arr))
# print "RATE OF FALSE POSITIVES PER TEST SIZE: {}".format(np.mean(false_pos)/y_test.shape[0])
# print "RATE OF FALSE POSITIVES PER AMT OF POSITIVES: {}".format(np.mean(false_pos)/np.mean(amt_success))
# print "AVERAGE RECALL: {}".format(np.mean(recall_arr))
# print "AVERAGE F-BETA: {}".format(np.mean(fbeta_arr))
