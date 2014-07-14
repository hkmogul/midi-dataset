import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
import librosa
import midi
import pretty_midi
import glob
import subprocess
import joblib
import os
import sys
sys.path.append('../')
import align_midi
import alignment_analysis
import scipy.io
import csv


''' Start of post analysis of offset information to gain confidence measure in
    running alignment. Ultimate goal is to have failing alignments come out with
    low alignments, and passing alignments have high ones.'''
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
def to_cqt_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_cqt.npy '''
    return os.path.splitext(filename)[0] + '_cqt.npy'
def to_beats_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_beats.npy '''
    return os.path.splitext(filename)[0] + '_beats.npy'
def to_onset_strength_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_onset_strength.npy '''
    return os.path.splitext(filename)[0] + '_onset_strength.npy'
def to_piano_cqt_npy(filename):
    return os.path.splitext(filename)[0]+'-piano.npy'
def to_chroma_npy(filename):
    return os.path.splitext(filename)[0]+'-chroma.npy'



BASE_PATH = '../data/cal500_txt/'
cqt_scores_pass  = np.zeros((0,))
cqt_scores_fail = np.zeros((0,))
cqt_scores_passUW = np.zeros((0,))
cqt_scores_failUW = np.zeros((0,))

piano_diff_pass = np.zeros((0,))
piano_diff_fail = np.zeros((0,))
norm_mat_pass = np.zeros((0,))
norm_mat_fail = np.zeros((0,))

path_to_may_30 = '../../CSV_Analysis/5-30-14_Alignment_Results.csv'
may_30_file = open(path_to_may_30)
csv_may = csv.reader(may_30_file)
csv_may.next()
for row in csv_may:
  piano_out = os.path.join(BASE_PATH, 'midi-aligned-additive-dpmod-piano',vs_filename_to_path(row[0]))
  success = row[2]
  mat_out = os.path.join('../../MIDI_Results_5-30',row[0]).replace('.mid', '.mat')+'.mat'
  # load cqt based results
  cqt_mat = scipy.io.loadmat(mat_out)
  p1 = cqt_mat['p'][0,:]
  q1 = cqt_mat['q'][0,:]
  sim_mat_1 = cqt_mat['similarity_matrix']
  score1 = cqt_mat['score'][0,0]
  print "Analyzing {}".format(vs_filename_to_path(row[0]))

  print "Successful alignment: {}".format(success)
  print "Weighted score: {}".format(score1)
  uw_score1 = alignment_analysis.get_unweighted_score(p1,q1,sim_mat_1)
  print "Unweighted score: {}".format(uw_score1)
  norm_mat1 = np.sum(sim_mat_1)/(sim_mat_1.shape[0]*sim_mat_1.shape[1])
  print "Normalized similarity matrix magnitude: {}".format(norm_mat1)
  print "-----"
  print "Analyzing piano roll information."
  piano_mat = scipy.io.loadmat(piano_out)
  p2 = piano_mat['p'][0,:]
  q2 = piano_mat['q'][0,:]
  sim_mat_2 = piano_mat['similarity_matrix']
  score2 = piano_mat['score'][0,0]

  print "Weighted score (piano):{}".format(score2)
  uw_score2 = alignment_analysis.get_unweighted_score(p2,q2,sim_mat_2)
  print "Unweighted score (piano):{}".format(uw_score2)
  norm_mat2 = np.sum(sim_mat_2)/(sim_mat_2.shape[0]*sim_mat_2.shape[1])
  print "Normalized similarity matrix magnitude: {}".format(norm_mat2)
  path_diff = alignment_analysis.compare_paths(p1,q1,p2,q2)
  print "Percent difference in paths: {}".format(path_diff*100)
  print "--------------------"

  if int(success) == 1:
    cqt_scores_pass = np.append(cqt_scores_pass, score1)
    cqt_scores_passUW =np.append(cqt_scores_passUW, uw_score1)
    norm_mat_pass = np.append(norm_mat_pass, norm_mat1)
    piano_diff_pass = np.append(piano_diff_pass, path_diff)
  else:
    cqt_scores_fail = np.append(cqt_scores_fail, score1)
    cqt_scores_failUW = np.append(cqt_scores_failUW, uw_score1)
    norm_mat_fail = np.append(norm_mat_fail, norm_mat1)
    piano_diff_fail = np.append(piano_diff_fail, path_diff)
print "Passing weighted score statistics:"
print "Average value: {}".format(np.mean(cqt_scores_pass))
print "Maximum: {}".format(np.amax(cqt_scores_pass))
print "Minimum: {}".format(np.amin(cqt_scores_pass))
print "Standard deviation: {}".format(np.std(cqt_scores_pass))
print "-----"
print "Passing unweighted score statistics:"
print "Average value: {}".format(np.mean(cqt_scores_passUW))
print "Maximum: {}".format(np.amax(cqt_scores_passUW))
print "Minimum: {}".format(np.amin(cqt_scores_passUW))
print "Standard deviation: {}".format(np.std(cqt_scores_passUW))
print "-----"
print "Passing similarity matrix magnitude statistics:"
print "Average value: {}".format(np.mean(norm_mat_pass))
print "Maximum: {}".format(np.amax(norm_mat_pass))
print "Minimum: {}".format(np.amin(norm_mat_pass))
print "Standard deviation: {}".format(np.std(norm_mat_pass))
print "-----"
print "Passing difference in piano path statistics:"
print "Average value: {}".format(np.mean(piano_diff_pass))
print "Maximum: {}".format(np.amax(piano_diff_pass))
print "Minimum: {}".format(np.amin(piano_diff_pass))
print "Standard deviation: {}".format(np.std(piano_diff_pass))
print "-----------"

print "Failing weighted score statistics:"
print "Average value: {}".format(np.mean(cqt_scores_fail))
print "Maximum: {}".format(np.amax(cqt_scores_fail))
print "Minimum: {}".format(np.amin(cqt_scores_fail))
print "Standard deviation: {}".format(np.std(cqt_scores_fail))
print "-----"

print "Failing unweighted score statistics:"
print "Average value: {}".format(np.mean(cqt_scores_failUW))
print "Maximum: {}".format(np.amax(cqt_scores_failUW))
print "Minimum: {}".format(np.amin(cqt_scores_failUW))
print "Standard deviation: {}".format(np.std(cqt_scores_failUW))
print "-----"

print "Failing similarity matrix magnitude statistics:"
print "Average value: {}".format(np.mean(norm_mat_fail))
print "Maximum: {}".format(np.amax(norm_mat_fail))
print "Minimum: {}".format(np.amin(norm_mat_fail))
print "Standard deviation: {}".format(np.std(norm_mat_fail))

print "-----"
print "Failing difference from piano path statistics:"
print "Average value: {}".format(np.mean(piano_diff_fail))
print "Maximum: {}".format(np.amax(piano_diff_fail))
print "Minimum: {}".format(np.amin(piano_diff_fail))
print "Standard deviation: {}".format(np.std(piano_diff_fail))
