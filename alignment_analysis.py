import csv
import os
import numpy as np
import sys
import scipy.io

''' Module of functions for running comparisons of different paths '''

def compare_paths(p1,q1,p2,q2):
  ''' Enter in path vectors of two different alignments, get percent difference of the paths '''
  # first easy check,if the shapes are different no need to iterate through
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
