import scipy.io
import pretty_midi
import sys
import os
import csv
import numpy as np
sys.path.append('../')
import alignment_analysis

''' For loading feature vectors of cal10k results '''

def row_to_filenames(row):
  ''' parses row name to generate MIDI and audio filenames
      example: AC?DC - Back In Black.mp3_vs_Ac Dc_back in black.1.mp3
      should output AC?DC - Back In Black.mp3, AC?DC - Back In Black.1.mid
      and original to .mid '''
  output_midi = os.path.splitext(row)[0]+'.mid'
  sep = row.split('_vs_', 2)
  orig_mp3 = sep[0]
  midiSep = sep[1].split('_',2)
  midiArt = midiSep[0]
  midiName = os.path.splitext(midiSep[1])[0]
  return output_midi, orig_mp3, os.path.join(midiArt, midiName)+'.mid'

base_path = '../../../../../../Volumes/HKM-MEDIA/cal10k'
filename = os.path.join(base_path, 'Cal10k Labeling.csv')

output_path = '../data/cal500-ml_info'
if not os.path.exists(output_path):
  os.mkdir(output_path)

dataX = np.zeros((0,19))
dataY = np.zeros((0,1))
dataNames = np.empty((0,))
dtype = np.dtype('S50')
feature_labels = np.empty((0,))
firstRun = True
song_names = np.empty((0,))

with open(filename, 'r') as csvfile:
  reader = csv.reader(csvfile)
  # skip first row of headers
  reader.next()
  for row in reader:
    output_name, mp3_name, midi_name = row_to_filenames(row[0])
    if not firstRun:
      if row[2] is '':
        print "score is blank!"
      if mp3_name in song_names or row[2] is '' or row[2] is ' ':
        print "{} already analyzed or no score, moving on".format(midi_name)
        continue
    dataNames = np.append(dataNames, midi_name)
    song_names = np.append(song_names, mp3_name)
    print row
    mat_name = os.path.splitext(row[0])[0]+'.mat'
    mat_file = scipy.io.loadmat(os.path.join(base_path,'midi-aligned-additive-dpmod', mat_name))
    p = mat_file['p']
    q = mat_file['q']
    score = mat_file['score']
    sim_mat = mat_file['similarity_matrix']
    success = int(row[2])
    aligned_midi = pretty_midi.PrettyMIDI(os.path.join(base_path,'midi-aligned-additive-dpmod', output_name))
    old_midi = pretty_midi.PrettyMIDI(os.path.join('../data/cal500_txt/Clean_MIDIs',midi_name))
    if firstRun:
      vec, feature_labels = alignment_analysis.get_feature_vector(aligned_midi, old_midi, sim_mat, p,q,score, include_labels = True)
      firstRun = False
    else:
      vec =  alignment_analysis.get_feature_vector(aligned_midi, old_midi, sim_mat, p,q,score, include_labels = True)

    dataX = np.vstack((dataX, vec))
    dataY = np.vstack((dataY, np.array([success])))
  scipy.io.savemat(os.path.join(output_path, 'X and Y.mat'), {'X': dataX, 'y' : dataY, 'labels': feature_labels, 'names': dataNames})
