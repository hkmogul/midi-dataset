import csv
import os
import numpy as np
import sys
import scipy

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
    if filename[i:(i+3)] = '_vs_':
      loc = i+4
      break
  # get list of artist and file
  file_base = filename[loc:].split('_')
  return ext_to_mat(file_base[0]+'/'+file_base[1])

def compare_paths(cqt_mat_path, piano_mat_path):
  ''' Checks if alignment paths are the same, returns true if that is the case, false otherwise '''
  nErrors = 0
  cqt_mat = scipy.io.loadmat(cqt_mat_path)
  p1 = cqt_mat['p']
  q1 = cqt_mat['q']
  piano_mat = scipy.io.loadmat(piano_mat_path)
  p2 = piano_mat['p']
  q2 = piano_mat['q']
  # first easy check,if the shapes are different no need to iterate through
  if p1.shape[1] != p2.shape[1] or q1.shape[1] != q2.shape[1]:
    print "Sizes of " + piano_mat_path + " do not match."
    nErrors += 1
  # check p vectors
  for a,b in zip(p1, p2):
    if a != b:
      nErrors += 1
  # check q vectors
  for a,b in zip(q1, q2):
    if a != b:
      nErrors += 1
  return nErrors

path_to_530 = '../../MIDI_Results_5-30/'
path_to_csv = '../../CSV_Analysis/5-30-14_Alignment_Results.csv'
success_file = open('../analytic-files/Matching_Successful_alignments.txt','w')
diff_file = open('../analytic-files/Differing_Successful_alignments.txt','w')

success_fail = open('../analytic-files/Matching_Failing_alignments.txt','w')
diff_fail = open('../analytic-files/Differing_Failing_alignments.txt','w')
with open(path_to_csv) as csv_file:
  filereader = csv.reader(csv_file)
  for row in filereader:
    # if this was a successful file, get the .mat file of that one and

    mat_path = path_to_530+os.path.splitext(row[0])[0]+'.mat'
    piano_mat_path = vs_filename_to_path(row[0])
    nErrors = compare_paths(mat_path, piano_mat_path)
    if row[2] == 1:
      if nErrors == 0:
        print piano_mat_path + " matches CQT path, will write down"
        success_file.write(piano_mat_path + ',0 \n')
      else:
        diff_file.write(piano_mat_path+ ','+ str(nErrors)+'\n')
    else:
      if nErrors == 0:
        success_fail.write(piano_mat_path + ',0 \n')
      else:
        diff_fail.write(piano_mat_path+ ','+str(nErrors),+'\n')
success_file.close()
success_fail.close()
diff_file.close()
diff_fail.close()
