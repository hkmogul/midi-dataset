import numpy as np
import os
import sys
import csv
import scipy.io
sys.path.append('../')
import alignment_analysis
import pretty_midi

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

def get_mp3_name(line):
  ''' Reads line of data to get mp3 filename of string, used for preventing repeats
      Example: gloria_gaynor-i_will_survive.mp3_vs_Gaynor, Gloria_I Will Survive.pdf
      should come out as gloria_gaynor-i_will_survive.mp3 '''
  sep = line.split('_vs_', 2)
  return sep[0]


BASE_PATH = '../data/cal500_txt/'
dataX = np.zeros((0,alignment_analysis.get_current_feature_amt()))
dataY = np.zeros((0,1))
dataNames = np.empty((0,))
dtype = np.dtype('S50')
feature_labels = np.empty((0,))
firstRun = True
song_names = np.empty((0,))
path_to_may_30 = '../../CSV_Analysis/5-30-14_Alignment_Results.csv'
may_30_file = open(path_to_may_30)
csv_may = csv.reader(may_30_file)
csv_may.next()

for row in csv_may:
  title_path = vs_filename_to_path(row[0])
  mp3_name = get_mp3_name(row[0])
  if mp3_name in song_names:
    print "{} has already been compared, moving on.".format(mp3_name)
    continue
  else:
    song_names = np.append(song_names, mp3_name)
    dataNames = np.append(dataNames, title_path)
  success = int(row[2])
  mat_out = os.path.join('../../MIDI_Results_5-30',row[0]).replace('.mid', '.mat')+'.mat'
  # load cqt based results
  cqt_mat = scipy.io.loadmat(mat_out)
  p = cqt_mat['p'][0,:]
  q = cqt_mat['q'][0,:]
  sim_mat = cqt_mat['similarity_matrix']
  score = cqt_mat['score'][0,0]
  old_midi_path = os.path.join(BASE_PATH, 'Clean_MIDIs',title_path.replace('.mat','.mid'))
  aligned_midi_path = os.path.join('../../MIDI_Results_5-30',row[0]+'.mid')

  old_midi = pretty_midi.PrettyMIDI(old_midi_path)
  aligned_midi = pretty_midi.PrettyMIDI(aligned_midi_path)

  print "Analyzing {}".format(title_path)

  print "Successful alignment: {}".format(success)
  if firstRun:
    vec, fLabels = alignment_analysis.get_feature_vector(aligned_midi, old_midi,sim_mat, p, q, score, include_labels = True)
  else:
    vec = alignment_analysis.get_feature_vector(aligned_midi, old_midi,sim_mat, p, q, score)
  dataX = np.vstack((dataX, vec))
  dataY = np.vstack((dataY, np.array([success])))
  firstRun = False
  print "------"


path_for_dataX = '../data/ML_new'
# path_for_dataX = '../data/ML_info'

if not os.path.exists(path_for_dataX):
  os.mkdir(path_for_dataX)
for i in xrange(feature_labels.shape[0]):
  print feature_labels[i]
scipy.io.savemat(os.path.join(path_for_dataX,'X_and_y.mat'),{'X': dataX,'y': dataY, 'names' : dataNames, 'labels' : fLabels})
