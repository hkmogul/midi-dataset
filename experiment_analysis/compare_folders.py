import csv
import os
import numpy as np
import sys
import scipy.io
import glob

''' Data processing to find differences in different folders of results'''

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

def compare_paths(mat_path_1, mat_path_2,percent_in = 0):
  ''' Checks if alignment paths are the same, returns true if that is the case, false otherwise '''
  nErrors = 0
  total_possible = 1 # we know there will be a size amount
  mat_1 = scipy.io.loadmat(mat_path_1)
  p1 = mat_1['p']
  p1 = p1[0][:]
  q1 = mat_1['q']
  q1 = q1[0][:]

  mat_2 = scipy.io.loadmat(mat_path_2)
  p2 = mat_2['p']
  p2 = p2[0][:]

  q2 = mat_2['q']
  q2 = q2[0][:]

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


base_path = '../data/pass-fail_distinguish/'
if len(sys.argv) <3:
  path_1 = raw_input('First folder name? ')
  path_2 = raw_input('Second folder name? ')
else:
  path_1 = sys.argv[1]
  path_2 = sys.argv[2]
  print "path1 is {0}, path2 is {1}".format(path_1, path_2)
if not os.path.exists(base_path+path_1) or not os.path.exists(base_path+path_2):
  print "Error: one of these paths doesn't exist.  Please try again"
else:
  OUTPUT_PATH = '../analytic-files/'+path_1+'-vs-'+path_2
  if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
  success_file = open(OUTPUT_PATH+'/Matching_alignments.csv','w')
  diff_file = open(OUTPUT_PATH+'/Differing_alignments.csv','w')


  amt_match = 0
  amt_diff = 0
  # get all .mat filenames in common (error check for those of different filenames)
  glob_1 = glob.glob(base_path+path_1+'/*/*.mat')
  glob_2 = glob.glob(base_path+path_2+'/*/*.mat')
  files_to_compare = []
  if len(glob_2) <= len(glob_1):
    for mat_file in glob_2:
      if mat_file.replace(path_2, path_1) in glob_1:
        files_to_compare.append(mat_file.replace(base_path+path_2,''))
  else:
    for mat_file in glob_1:
      if mat_file.replace(path_1, path_2) in glob_2:
        files_to_compare.append(mat_file.replace(base_path+path_1,''))

  for file in files_to_compare:
    error_rate = compare_paths(base_path+path_1+'/'+file, base_path+path_2+'/'+file, percent_in = 0)
    print file +" {} error_rate".format(error_rate)
    if error_rate == 0:
      amt_match += 1
      success_file.write(file+ '\t 0 \n')
    else:
      amt_diff += 1
      diff_file.write(file+'\t'+str(error_rate)+'\n')

success_file.close()
diff_file.close()
print "Total # of files that match paths: {}".format(amt_match)
print "Total # of files that differ: {}".format(amt_diff)
