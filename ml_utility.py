import numpy as np
import random
import matplotlib.pyplot as plt
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

def cross_val_with_names(X, y, names, test_amt, random_seed = None):
  ''' Similar to sklearn cross-validation, but also outputs the names of the
      test set songs and training data songs '''
  # initialize RNG
  random.seed(random_seed)
  test_len = int(test_amt*y.shape[0])
  train_len = y.shape[0]-test_len
  indices = range(y.shape[0])


  # print indices
  random.shuffle(indices)
  test_indices = indices[0:test_len]
  train_indices = indices[test_len:]

  # print indices
  # we know the sizes of the intended arrays, use that to instantiate the arrays

  X_train = np.empty((train_len,X.shape[1]))
  # X_train = np.copy(X)
  X_test = np.empty((test_len, X.shape[1]))


  y_train = np.empty((train_len,))
  # y_train = np.copy(y)
  y_test = np.empty((test_len,))

  dt = np.dtype('a50')
  names_train = np.empty((train_len,), dtype = dt)
  # names_train = np.copy(names)
  names_test = np.empty((test_len,), dtype = dt)


  #fill in testing data
  for i in xrange(test_len):

    X_test[i,:] = X[test_indices[i],:]

    y_test[i] = y[test_indices[i]]

    names_test[i] = names[test_indices[i]]
  for i in xrange(train_len):
    X_train[i,:] = X[train_indices[i],:]

    y_train[i] = y[train_indices[i]]

    names_train[i] = names[train_indices[i]]

  return X_train, X_test, y_train, y_test, names_train, names_test

def pretty_print_prob(y_pred, y, prob, names):
  ''' For printing out probabilities of outputs and what they actually were '''
  for i in xrange(y_pred.shape[0]):
    print "Name of file: {}".format(names[i])
    print "\t P(0)= {0}, P(1)= {1}".format(prob[i,0], prob[i,1])
    print "\t RF classified as {0}, was actually {1}".format(y_pred[i],y[i])
    print "----------"

def util_print(y_pred, y, prob, names = None, save = False):
  ''' not so pretty, but useful print. also an option to save '''
  info = np.empty((y_pred.shape[0],2))
  if names is None:
    for i in xrange(y_pred.shape[0]):
      if y_pred[i] == y[i]:
        success = 1
      else:
        success = 0
      print '{0}, {1}'.format(max(prob[i,0],prob[i,1]), success)
      info[i,0] = max(prob[i,0], prob[i,1])
      info[i,1] = success

  else:
    for i in xrange(y_pred.shape[0]):
      if y_pred[i] == y[i]:
        success = 1
      else:
        success = 0
      print '{0}, {1}, {2}'.format(max(prob[i,0],prob[i,1]), success, names[i])
      info[i,0] = max(prob[i,0], prob[i,1])
      info[i,1] = success
  if save:
    # get final form into pdf
    plt.plot(info[:,0],info[:,1], '.')
    plt.ylim([-1,2])
    plt.savefig('../comparison.pdf')


def print_false_positives(y_pred, y, prob, names=None):
  # info = np.empty((y_pred.shape[0],2))
  if names is None:
    for i in xrange(y_pred.shape[0]):
      if y_pred[i] == 1 and y[i] == 0:
        print '{0}'.format(max(prob[i,0],prob[i,1]))
      # info[i,0] = max(prob[i,0], prob[i,1])
      # info[i,1] = success

  else:
    for i in xrange(y_pred.shape[0]):
      if y_pred[i] == 1 and y[i] == 0:
        print '{0}, {1}'.format(max(prob[i,0],prob[i,1]),names[i])
      # info[i,0] = max(prob[i,0], prob[i,1])
      # info[i,1] = success
def find_prob(y_true, prob):
  prob_y = np.empty((y_true.shape))
  for i in xrange(y_true.shape[0]):
    if y_true[i] == 1:
      prob_y[i] == prob[1]
    else:
      prob_y[i] == prob[0]

  return prob_y

def print_data_info(data_y, data_names):
  for i in xrange(data_y.shape[0]):
    print "{0}, {1}".format(data_names[i], data_y[i])

def output_with_threshold(prob_y, threshold = .5):
  output = np.empty(prob_y.shape)
  for i in xrange(prob_y.shape[0]):
    if prob_y[i] >= threshold:
      output[i] = 1
    else:
      output[i] = 0
  return output

def get_accuracy(y_true, y_pred):
  amt_correct = 0
  amt_total = y_pred.shape[0]
  for i in xrange(y_pred.shape[0]):
    if y_true[i] == y_pred[i]:
      amt_correct += 1
  return float(amt_correct)/amt_total
