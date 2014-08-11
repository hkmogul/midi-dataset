import sklearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
import pickle
import sys
sys.path.append('../')
import ml_utility as ml
from matplotlib.backends.backend_pdf import PdfPages



mat_path = '../data/ML_new/X_and_y.mat'
# mat_path = '../data/ML_info-no_repeats/X_and_y.mat'
dump_file = open('../data/ML_info/Rf-classifier.pkl','w')
data = scipy.io.loadmat(mat_path)
Xn = data['X']
print Xn.shape
yn = data['y']
names = data['names']
labels = data['labels']
trees = 150

clf = RandomForestClassifier(n_estimators = trees)
X, col_mean, col_std = ml.normalize_matrix(Xn)
y = np.reshape(yn, (yn.shape[0],))


clf.fit(X,y)

print "SCORE OF WHOLE FIT"
print clf.score(X,y)
pickle.dump(clf,dump_file)
print "Pickling successful"
print "----------"
test_size = .25
# clf_svm = svm.SVC(gamma = .001, C = 200)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                    y, test_size=test_size, random_state=0)
# clf_svm.fit(X_train, y_train)
# print "SCORE OF CROSS VAL SVC: {}".format(clf_svm.score(X_test, y_test))
# cross validation
print "SCORE OF TEST/RANDOM FIT RANDOM FOREST; TEST SIZE = {}".format(test_size)


X_train, X_test, y_train, y_test, names_train, names_test = ml.cross_val_with_names(X, y, names, test_amt = test_size, random_seed = None)
# single cross validation

# ml.print_data_info(y_test, names_test)



print "NUMBER OF TREES: {}".format(trees)
clf2 = RandomForestClassifier(n_estimators = trees).fit(X_train, y_train)
print "SCORE OF CROSS VALIDATION, {0} TRAINING SAMPLES: {1}".format(
                                                      y_train.shape[0],
                                                      clf2.score(X_test, y_test))
y_pred = clf2.predict(X_test)
proba = clf2.predict_proba(X_test)
# proba_y = ml.find_prob(y_test, proba)
precision_curve, recall_curve, thresholds = metrics.precision_recall_curve(y_test, proba[:,1], pos_label = 1)
plt.plot(precision_curve, recall_curve, '.')
plt.savefig("precision-recall.pdf")
plt.close()
print thresholds
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
# fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)

roc_auc = metrics.roc_auc_score(y_test, proba[:,1])
print "ROC-AUC Score: {}".format(roc_auc)
fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, proba[:,1])
print roc_thresholds
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc-plot.pdf')
plt.close()


# multiple cross validation
amt = 50

print "BEGINNING MULTIPLE CROSS VALIDATION, ITERATIONS = {}".format(amt)
print "-----"

scores = np.empty((amt,))
scores_thresh = np.empty((amt,))
false_pos = np.empty((amt,))
false_neg = np.empty((amt,))
precision_arr = np.empty((amt,))
recall_arr = np.empty((amt,))
fb_arr = np.empty((amt,))
amt_success = np.empty((amt,)) # amount of test items the ML deems = 1
auc_arr = np.empty((amt,))
roc_auc_arr = np.empty((amt,))

# pdf = PdfPages('Multiple Iterations ROC.pdf')
with PdfPages('Multiple Iterations ROC.pdf') as pdf:
  for i in xrange(amt):
    print "Iteration # {}".format(i)
    # use the index to generate random state because why not
    X_train, X_test, y_train, y_test, names_train, names_test = ml.cross_val_with_names(X,
                                        y, names, test_amt = test_size)

    clf_mult = RandomForestClassifier(n_estimators = trees).fit(X_train, y_train)

    y_pred = clf_mult.predict(X_test)
    # get score instead of calculating
    s = clf_mult.score(X_test, y_test)
    print "Accuracy: %0.2f %%" %(s*100)
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
    thresh = .8
    y_thresh = ml.output_with_threshold(proba[:,1], threshold = thresh)
    print "Accuracy with threshold of {0}: {1}".format(thresh, ml.get_accuracy(y_test, y_thresh))
    ml.print_false_positives(y_thresh, y_test, proba, names_test)
    # proba_y = ml.find_prob(y_test, proba)
    precision_curve, recall_curve, thresholds = metrics.precision_recall_curve(y_test, proba[:,1], pos_label = 1)
    area = metrics.auc(recall_curve, precision_curve)
    print "Precision-Recall AUC: {}".format(area)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, proba[:,1])
    roc_auc = metrics.roc_auc_score(y_test, proba[:,1])
    print "ROC-AUC Score: {}".format(roc_auc)
    print "# of Thresholds: {}".format(roc_thresholds.shape[0])
    scores[i] = s
    scores_thresh[i] = ml.get_accuracy(y_test, y_thresh)
    false_pos[i] = fp
    false_neg[i] = fn
    precision_arr[i] = precision
    recall_arr[i] = recall
    fb_arr[i] = fbeta_score
    amt_success[i] = np.argwhere(y_pred).shape[0]
    auc_arr[i] = area
    roc_auc_arr[i] = roc_auc


    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic- iteration # {}'.format(i+1))
    plt.legend(loc="lower right")
    pdf.savefig()
    plt.close()
    print "-----"
print "Amount of test datapoints per iteration: {}".format(y_test.shape[0])
print "AVERAGE ACCURACY IN MULTIPLE: %0.2f %%" %(100*np.mean(scores))
print "Maximum accuracy: %0.2f %%" %(100*np.amax(scores))
print "Minimum accuracy: %0.2f %%" %(100*np.amin(scores))

print "AVERAGE ACCURACY IN MULTIPLE (using threshold): %0.2f %%" %(100*np.mean(scores_thresh))
print "Maximum accuracy (using threshold): %0.2f %%" %(100*np.amax(scores_thresh))
print "Minimum accuracy (using threshold): %0.2f %%" %(100*np.amin(scores_thresh))

print "AVERAGE AMT OF FALSE POSITIVES: {}".format(np.mean(false_pos))
print "Minimum # of false positives: {}".format(np.amin(false_pos))
print "AVERAGE PRECISION: {}".format(np.mean(precision_arr))
print "RATE OF FALSE POSITIVES PER TEST SIZE: {}".format(np.mean(false_pos)/y_test.shape[0])
print "RATE OF FALSE POSITIVES PER AMT OF POSITIVES: {}".format(np.mean(false_pos)/np.mean(amt_success))
print "AVERAGE RECALL: {}".format(np.mean(recall_arr))
print "AVERAGE F-BETA: {}".format(np.mean(fbeta_arr))
print "Average PR-AUC: {}".format(np.mean(auc_arr))
print "Min PR-AUC: {}".format(np.amin(auc_arr))
print "Average ROC-AUC: {}".format(np.mean(roc_auc_arr))
print "Min ROC-AUC: {}".format(np.amin(roc_auc_arr))

print "----------"


print "Data about first classifier"

print np.argsort(clf.feature_importances_)
# for i in xrange(labels.shape[0]):
for i in np.argsort(clf.feature_importances_):
  print '{0}, Importance: {1}'.format(labels[i], clf.feature_importances_[i])


print 'least effective: \n {0}, {1}'.format(labels[np.argmin(clf.feature_importances_)], np.amin(clf.feature_importances_))
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
