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
  file_base = filename[i:].split('_')
  return ext_to_mat(file_base[0]+'/'+file_base[1]

def compare_paths(cqt_mat, piano_mat):
  p1,q1
