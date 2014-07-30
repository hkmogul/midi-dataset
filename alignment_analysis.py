import os
import numpy as np
import sys
import scipy.io
import pretty_midi
import scipy.stats
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

def get_regression_stats(m_aligned,m, offsets = None, note_ons = None):
  ''' Used for performing linear regression stats on the aligned offsets '''
  if offsets == None or note_ons == None:
    offsets, note_ons = get_offsets(m_aligned, m)

  else:
    note_ons = np.array([note.start for instrument in m.instruments for note in instrument.notes])
    aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.notes])
    last = min(note_ons.shape[0],aligned_note_ons.shape[0])

  last = min(note_ons.shape[0], offsets.shape[0])
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

def parabola_fit(cost_path):
  ''' Returns polynomial coefficients and fit value for parabolic fit of data '''
  x = np.arange(start = 0, stop = cost_path.shape[0])
  p = np.polyfit(x =x, y =cost_path, deg = 2)
  # build residuals because apparently numpy just gives the sum of them, and actual parabola because why not
  parab = p[2]+p[1]*x+p[0]*x**2
  # residuals = np.zeros(x.shape)
  residuals = np.subtract(cost_path, parab)
  # for i in xrange(residuals.shape[0]):
  #   residuals[i] = cost_path[i]-parab[i]
  return p, parab,residuals

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
