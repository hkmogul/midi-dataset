import numpy as np

def normalize_matrix(X):
  ''' Returns matrix with each column normalized to have mean 0 and standard dev 1 '''
  new_X = np.copy(X)
  col_mean = np.mean(X, axis = 0)
  col_std = np.std(X, axis = 0)
  # subtract mean from each
  for i in xrange(X.shape[0]):
    new_X[i,:] = np.divide(np.subtract(X[i,:], col_mean), col_std)

  return new_X

def get_fp_fn(y_pred, y_test):
  ''' Get amount of false positives and negatives in specified output (using
      this data, where target 1 is success, 0 is failure).  Returns an int if
      properly formatted '''

  fp = 0
  fn = 0
  if y_pred.shape[0] != y_test.shape[0]:
    return None, None
  else:
    for i in xrange(y_pred.shape[0]):
      if y_test[i] == 1 and y_pred[i] == 0:
        fn += 1
      elif y_test[i] == 0 and y_pred[i] == 1:
        fp += 1
  return fp, fn

def get_tp_tn(y_pred, y_test):
  ''' Get amount of true positives and negatives based
      on this specific data (1 is positive) '''
  tp = 0
  tn = 0
  if y_pred.shape[0] != y_test.shape[0]:
    return None, None
  else:
    for i in xrange(y_pred.shape[0]):
      if y_test[i] == 1 and y_pred[i] == 1:
        tp += 1
      elif y_test[i] == 0 and y_pred[i] == 0:
        tn += 1
  return tp, tn


def get_precision_recall(y_pred, y_test):
  fp, fn = get_fp_fn(y_pred, y_test)
  tp, tn = get_tp_tn(y_pred, y_test)
  precision = float(tp)/(tp+fp)
  recall = float(tp)/(tp+fn)
  return precision, recall
