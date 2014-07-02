import csv
import os
import numpy as np
import sys
import scipy.io

''' Comparison of results from May 30 MIDI Alignment to piano roll alignment '''

def ext_to_mat(filename):
  ''' Converts any pathname to have .mat at end '''
  file_path = os.path.splitext(filename)[0]
  return file_path + '.mat'

def vs_filename_to_path(filename):
  ''' Converts output filename of aligned MIDI to the .mat file that contains path
  example: alice_in_chains-no_excuses.mp3_vs_Alice In Chains_No Excuses.1.mid
  should come out as Alice In Chains/No Excuses.1.mid '''
  # hacky way- iterate thru string until subset 0 to 3 is '_vs_', then use _ as delimiter to generate path
  # then concat with base path to
  loc = 0
  for i in xrange(0, len(filename)):
    if filename[i:(i+4)] == '_vs_':
      loc = i+4
      break
  # get list of artist and file
  file_base = filename[loc:].split('_',2)
  return ext_to_mat(file_base[0]+'/'+file_base[1])

def compare_paths(cqt_mat_path, piano_mat_path, percent_in = 0):
  ''' Checks if alignment paths are the same, returns true if that is the case, false otherwise '''
  nErrors = 0
  total_possible = 1
  cqt_mat = scipy.io.loadmat(cqt_mat_path)
  p1 = cqt_mat['p']
  p1 = p1[0][:]
  q1 = cqt_mat['q']
  q1 = q1[0][:]

  piano_mat = scipy.io.loadmat(piano_mat_path)
  p2 = piano_mat['p']
  p2 = p2[0][:]

  q2 = piano_mat['q']
  q2 = q2[0][:]

  # first easy check,if the shapes are different no need to iterate through
  if p1.shape[0] != p2.shape[0] or q1.shape[0] != q2.shape[0]:
    # print "Sizes of " + piano_mat_path + " do not match."
    nErrors += 1


  # turn vectors into p,q tuples
  pq1 = np.rot90(np.vstack((p1,q1)),3)

  pq2 = np.rot90(np.vstack((p2,q2)),3)
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

  # pq1_tuple = tuple(map(tuple, pq1))
  # pq2_tuple = tuple(map(tuple,pq2))
  # check if #2 has the tuples of #1
  # for t in pq1_tuple:
  #   if t not in pq2_tuple:
  #     nErrors += 1
  #   total_possible += 1

  return float(nErrors)/total_possible


piano_base_path = '../data/cal500_txt/midi-alignment-exp-forceH-5/'
path_to_530 = '../../MIDI_Results_5-30/'
path_to_csv = '../../CSV_Analysis/5-30-14_Alignment_Results.csv'
if not os.path.exists('../analytic-files/force_experiments'):
  os.makedirs('../analytic-files/force_experiments')
success_file = open('../analytic-files/force_experiments/forceH5-Matching_Successful_alignments-forceH5.csv','w')
diff_file = open('../analytic-files/force_experiments/forceH5-Differing_Successful_alignments-forceH5.csv','w')
success_fail = open('../analytic-files/force_experiments/forceH5-Matching_Failing_alignments-forceH5.csv','w')
diff_fail = open('../analytic-files/force_experiments/forceH5-Differing_Failing_alignments-forceH5.csv','w')

amt_match = 0
with open(path_to_csv) as csv_file:
  filereader = csv.reader(csv_file)
  next(filereader, None)
  amt_close_1 = 0
  amt_close_2 = 0
  for row in filereader:
    # if this was a successful file, get the .mat file of that one and
    mat_path = path_to_530+os.path.splitext(row[0])[0]+'.mat.mat'
    piano_mat_path = piano_base_path+vs_filename_to_path(row[0])
    if os.path.exists(piano_mat_path):
      nErrors = compare_paths(mat_path, piano_mat_path, percent_in = 5)
      print os.path.basename(piano_mat_path) +" "+ str(nErrors)
      if int(row[2]) == 1:
        if nErrors == 0:
          amt_match += 1
          success_file.write(piano_mat_path + '\t 0 \n')
        else:
          if nErrors <= .05:
            amt_close_1 += 1
          diff_file.write(piano_mat_path+ '\t'+ str(nErrors)+'\n')
      else:
        if nErrors == 0:
          amt_match += 1
          success_fail.write(piano_mat_path + '\t 0 \n')
        else:
          if nErrors <= .05:
            amt_close_2 += 1
          diff_fail.write(piano_mat_path+ '\t'+str(nErrors)+'\n')
success_file.close()
success_fail.close()
diff_file.close()
diff_fail.close()
print "Total number of files that completely match: {}".format(amt_match)
print "Total number of successful alignments that are close (<= 5 percent difference): {}".format(amt_close_1)
print "Total number of failing alignments that are close (<= 5 percent difference): {}".format(amt_close_2)