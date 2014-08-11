import os
import numpy as np
import sys
import scipy.io
import pretty_midi
import scipy.stats
import librosa
import pretty_midi
import scipy.signal

''' Module of functions for running comparisons of different paths '''

def compare_paths(p1,q1,p2,q2, percent_in = 0):
  ''' Enter in path vectors of two different alignments, get percent difference of the paths '''
  # first easy check,if the shapes are different no need to iterate through
  nErrors = 0
  total_possible = 1
  if p1.shape[0] != p2.shape[0] or q1.shape[0] != q2.shape[0]:
    nErrors += 1

  pq1 = np.rot90(np.vstack((p1,q1)),3)

  pq2 = np.rot90(np.vstack((p2,q2)),3)

  # we just want last 90% of this
  mat_list_1= pq1.tolist()
  mat_list_2 = pq2.tolist()
  # for pair in mat_list_1:
  #   if pair not in mat_list_2:
  #     nErrors += 1
  #   total_possible += 1

  for i in xrange(len(mat_list_1)-1):
    if i > .01*percent_in*(len(mat_list_1)-1):
      pair = mat_list_1[i]
      if pair not in mat_list_2:
        nErrors += 1
      total_possible += 1


  # print nErrors
  # print total_possible
  return float(nErrors)/total_possible


def get_unweighted_score(p,q,similarity_matrix):
  ''' Calculates what the score would have been without the penalty '''
  score = 0
  for i in xrange(p.shape[0]):
    index1 = p[i]
    index2 = q[i]
    score = score + similarity_matrix[index1,index2];
  return score/p.shape[0];


def get_offsets(m_aligned, m):
  ''' Forms and returns alignment offsets based on the adjusted midi and the original '''
  note_ons = np.array([note.start for instrument in m.instruments for note in instrument.notes])
  aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.notes])
  diff = np.array([[]])
  last = min(note_ons.shape[0],aligned_note_ons.shape[0])
  for i in xrange(last):
    diff = np.append(diff,np.array([aligned_note_ons[i]-note_ons[i]]))
  return diff, note_ons[0:last]

def get_cost_path(p,q,similarity_matrix):
  ''' Forms and returns the cost per step of alignment path '''
  cost_path = np.zeros((0,))
  for i in xrange(p.shape[0]):
    cost_path = np.append(cost_path, similarity_matrix[p[i],q[i]])
  return cost_path

def get_regression_stats(offsets, note_ons):
  ''' Used for performing linear regression stats on the aligned offsets '''
  last = min(note_ons.shape[0],offsets.shape[0])
  slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(note_ons[0:last-1],offsets[0:last-1])
  return slope, intercept, r_value, p_value, std_err


def get_non_diagonal_steps(p,q):
  ''' Returns number of non-diagonal steps in path
  both horizontal and vertical are returned as 2 different values '''

  # first, get number of horizontal steps from p
  horiz = 0
  current = p[0]
  for i in xrange(1,p.shape[0]):
    if p[i] == current:
      horiz += 1
    else:
      current = p[i]

  # get verticals the same way
  vert = 0
  current = q[0]
  for j in xrange(1, q.shape[0]):
    if q[j] == current:
      vert +=1
    else:
      current = q[j]
  return horiz, vert

def parabola_fit(filter_cost_path, cost_path, start, end):
  ''' Returns polynomial coefficients and fit value for parabolic fit of data '''
  x = np.arange(start = 0, stop = filter_cost_path.shape[0])
  p = np.polyfit(x =x, y =filter_cost_path, deg = 2)
  # build residuals because apparently numpy just gives the sum of them, and actual parabola because why not
  parab = p[2]+p[1]*x+p[0]*x**2

  # residuals = np.zeros(x.shape)
  residuals_filt = np.subtract(filter_cost_path, parab)
  res_original = np.subtract(cost_path[start:end], parab)
  # for i in xrange(residuals.shape[0]):
  #   residuals[i] = cost_path[i]-parab[i]
  return parab, residuals_filt, res_original

def pad_lesser_vec(vec1, vec2):
  ''' Returns input vectors with the one of greater length unchanged, and
      the lesser padded with zeros at end. Also returns number of indices
      it added for padding '''
  amt = 0
  if vec1.shape[0]  == vec2.shape[0]:
    return vec1, vec2, amt
  elif vec1.shape[0] > vec2.shape[0]:
    while vec1.shape[0] > vec2.shape[0]:
      vec2 = np.append(vec2, 0)
      amt += 1
    return vec1, vec2, amt
  else:
    while vec2.shape[0] > vec1.shape[0]:
      vec1 = np.append(vec1,0)
      amt+=1
    return vec1, vec2, amt

def truncate_greater_vec(vec1, vec2):
  ''' Returns input vectors with the one of shorter length unchanged, and the
      greater one's extra indices removed. Also returns amount it shortened it by.'''
  amt = 0
  if vec1.shape[0]  == vec2.shape[0]:
    return vec1, vec2, amt
  elif vec1.shape[0] > vec2.shape[0]:
    while vec1.shape[0] > vec2.shape[0]:
      vec1 = vec1[:-1]
    return vec1, vec2, amt
  else:
    while vec2.shape[0] > vec1.shape[0]:
      vec2 = vec2[:-1]
      amt+=1
    return vec1, vec2, amt

def util_print(data_y, data_names):
  for i in xrange(data_y.shape[0]):
    print '{0}, {1}'.format(data_names[i], data_y[i])


def get_weighted_score(mat_path = None, score = None, mat_file = None):
  ''' Collects weighted score by loading data from a mat file, or regurgitating back an input '''
  if score is not None:
    return score
  elif mat_file is not None:
    return mat_file['score']
  elif mat_path is not None:
    mf = scipy.io.loadmat(mat_path)
    return mf['score']
  else:
    return 1

def get_normalized_sim_mat(mat_path = None, mat_file = None, similarity_matrix = None):
  ''' Returns normalized magnitude of similarity matrix '''
  if similarity_matrix is not None:
    return np.sum(similarity_matrix)/(similarity_matrix.shape[0]*similarity_matrix.shape[1])
  elif mat_file is not None:
    sim_mat = mat_file['similarity_matrix']
    return np.sum(sim_mat)/(sim_mat.shape[0]*sim_mat.shape[1])
  elif mat_path is not None:
    mf = scipy.io.loadmat(mat_path)
    sim_mat = mf['similarity_matrix']
    np.sum(sim_mat)/(sim_mat.shape[0]*sim_mat.shape[1])
  else:
    return 1



def get_mag_diff(score, similarity_matrix):
  return abs(score-get_normalized_sim_mat(similarity_matrix = similarity_matrix))

def lin_regress_stats(offsets, note_ons):
  ''' gets relevant regression stats for offsets '''
  if offsets is not None and note_ons is not None:
    slope, intercept, r_value, p_value, std_err = get_regression_stats(offsets, note_ons)
    return r_value, std_err, intercept


def filter_cost_path(cost_path):
  cost_path_filtered = np.copy(cost_path)
  size = cost_path_filtered.shape[0]/2
  if size % 2 == 0:
    size +=1
  cost_path_filtered = scipy.signal.medfilt(cost_path, kernel_size = size)
  start = int(cost_path_filtered.shape[0]*.05)
  end = int(cost_path_filtered.shape[0]*.95)
  return cost_path_filtered[start:end], start, end


def get_feature_vector(aligned_midi, old_midi, similarity_matrix, p,q,score, include_labels = False):
  ''' Returns feature vector for machine learning application. Built to be used inline with midi alignment '''
  vec = np.empty((0,))
  dtype = np.dtype('S50')
  fLabels = np.empty((0,))
  #  - weighted score - gotten right from alignment
  vec = np.append(vec, score)
  fLabels = np.append(fLabels, 'Weighted Score')
  #  - normalized magnitude of matrix
  vec = np.append(vec, get_normalized_sim_mat(similarity_matrix = similarity_matrix))
  fLabels = np.append(fLabels, 'Average Sim-Mat Value')
  #  - difference between score and magnitude of matrix
  vec = np.append(vec, get_mag_diff(score, similarity_matrix))
  fLabels = np.append(fLabels, 'Difference of Score and Average Sim-Mat')
  #  - R value of linear regression of offsets (will collect all stats now)
  # first get offsets
  offsets, note_ons = get_offsets(aligned_midi, old_midi)
  r, stderr, intercept = lin_regress_stats(offsets = offsets, note_ons = note_ons)
  vec = np.append(vec, r)
  fLabels = np.append(fLabels, 'Lin Fit of Offsets- R')
  vec = np.append(vec, stderr)
  fLabels = np.append(fLabels, 'Lin Fit of Offsets- Stderr')
  vec = np.append(vec, intercept/old_midi.get_end_time())
  fLabels = np.append(fLabels, 'Lin Fit of Offsets- Intercept/End Time')

  #  variance of offsets
  vec = np.append(vec, np.var(offsets))
  fLabels = np.append(fLabels, 'Variance of Offsets')
  # standard dev of offsets
  vec = np.append(vec, np.std(offsets))
  fLabels = np.append(fLabels, 'Standard Dev of Offsets')
  # linear relations using first 20 percent of offset data
  r2, stderr2, intercept2 = lin_regress_stats(offsets = offsets[0:offsets.shape[0]*.2], note_ons = note_ons[0:offsets.shape[0]*.2])
  # r value of first 20 percent
  vec = np.append(vec, r2)
  fLabels = np.append(fLabels,'20 Percent Lin Fit Offsets -R')
  # intercept of first 20 percent
  vec = np.append(vec, intercept/old_midi.get_end_time())
  fLabels = np.append(fLabels, '20 Percent Lin Fit Offsets - Intercept/End Time')
  # ratio of max value in first 10 percent of offsets to end time
  vec = np.append(vec, np.amax(offsets[0:offsets.shape[0]*.1])/old_midi.get_end_time())
  fLabels = np.append(fLabels, 'First 10 percent Offsets- Max Value/End Time')



  # cost path information
  cost_path = get_cost_path(p,q,similarity_matrix)
  cost_path_filt, start, end = filter_cost_path(cost_path)
  # get parabolic fit info
  parab, residual_filt, res_original = parabola_fit(cost_path_filt, cost_path, start, end)
  # get datapoints of cost path- variance, standard dev, variance and std of residuals
  vec = np.append(vec, np.var(cost_path))
  fLabels = np.append(fLabels, 'Variance of Cost Path')
  vec = np.append(vec, np.std(cost_path))
  fLabels = np.append(fLabels, 'Standard Dev of Cost Path')

  vec =np.append(vec, np.var(res_original))
  fLabels = np.append(fLabels, 'Var of Residuals of Parabolic Fit (cost path)')
  vec = np.append(vec, np.std(res_original))
  fLabels = np.append(fLabels, 'Std of Residuals of Parabolic fit (cost path)')

  #datapoints of filtered cost path
  vec = np.append(vec, np.var(cost_path_filt))
  fLabels = np.append(fLabels, 'Variance of Filtered Cost Path')
  vec = np.append(vec, np.std(cost_path_filt))
  fLabels = np.append(fLabels, 'Standard Dev of Filtered Cost Path')
  vec =np.append(vec, np.var(residual_filt))
  fLabels = np.append(fLabels,'Var of Residuals of filtered Parab Fit')
  vec = np.append(vec, np.std(residual_filt))
  fLabels = np.append(fLabels,'Std of residuals of filtered Parab Fit')

  if include_labels:
    return vec, fLabels
  else:
    return vec
