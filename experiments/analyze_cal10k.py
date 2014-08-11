import scipy.io
import pretty_midi
import sys
import os
import csv
import numpy as np

''' For loading feature vectors of cal10k results '''

path_to_folder = '../../../../../../Volumes/HKM-MEDIA'
filename = os.path.join(path_to_folder, '???')

output_path = '../data/cal500-ml_info'
if not os.path.exists(output_path):
  os.mkdir(output_path)

dataX = np.zeros((0,19))
dataY = np.zeros((0,1))
dtype = np.dtype('S50')
feature_labels = np.empty((0,))
firstRun = True
song_names = np.empty((0,))

with open(filename, 'r') as csvfile:
  reader = csv.reader(csvfile)
  reader.next()
  for row in reader:
