''' Experiment to use piano roll as faster substitute for CQT of MIDI '''
from pretty_midi import PrettyMIDI
import numpy as np
import librosa
import os
import sys
sys.path.append('../')
import scipy.io
import matplotlib.pyplot as plt
import midi
import glob
import subprocess
import joblib

OUTPUT_PATH = 'piano_roll_CQT_comparisons'
BASE_PATH = '../data/sanity'


if not os.path.exists(os.path.join(BASE_PATH, OUTPUT_PATH)):
    os.makedirs(os.path.join(BASE_PATH, OUTPUT_PATH))

# Utility functions for converting between filenames
def to_cqt_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_cqt.npy '''
    return os.path.splitext(filename)[0] + '_cqt.npy'
def to_beats_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_beats.npy '''
    return os.path.splitext(filename)[0] + '_beats.npy'
def to_onset_strength_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_onset_strength.npy '''
    return os.path.splitext(filename)[0] + '_onset_strength.npy'

# <codecell>

def make_piano_CQT(midi):
  piano_roll = midi.get_piano_roll()
  piano_gram = piano_roll
  return piano_gram

def run_comparison(midi_filename, output_midi_filename):

  # Load in the corresponding midi file in the midi directory, and return if there is a problem loading it
  try:
      m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
  except:
      print "Error loading {}".format(midi_filename)
      return

  midi_gram = np.load(to_cqt_npy(midi_filename))
  piano_gram = make_piano_CQT(m)
  # Plot log-fs grams
  plt.figure(figsize=(36, 24))
  ax = plt.subplot2grid((4, 3), (0, 0), colspan=3)
  plt.title('MIDI Synthesized')
  librosa.display.specshow(midi_gram,
                           x_axis='frames',
                           y_axis='cqt_note',
                           fmin=librosa.midi_to_hz(36),
                           fmax=librosa.midi_to_hz(96))
  ax = plt.subplot2grid((4, 3), (1, 0), colspan=3)
  plt.title('Piano Roll data')
  librosa.display.specshow(piano_gram,
                           x_axis='frames',
                           y_axis='cqt_note',
                           fmin=librosa.midi_to_hz(36),
                           fmax=librosa.midi_to_hz(96))

  plt.savefig(output_midi_filename.replace('.mid', '.pdf'))
  plt.close()

midi_glob = sorted(glob.glob(os.path.join(BASE_PATH, 'midi', '*.mid')))
joblib.Parallel(n_jobs=7)(joblib.delayed(run_comparison)(midi_filename,
                                                         midi_filename.replace('midi', OUTPUT_PATH))
                                                         for midi_filename in midi_glob)
