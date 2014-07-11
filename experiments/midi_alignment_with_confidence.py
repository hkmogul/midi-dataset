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
import scipy.io
import csv


''' Start of post analysis of offset information to gain confidence measure in
    running alignment. Ultimate goal is to have failing alignments come out with
    low alignments, and passing alignments have high ones.'''

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


OUTPUT_PATH = 'confidence_experiments'

BASE_PATH = '../data/pass-fail_distinguish'

path_to_txt = os.path.join(BASE_PATH, 'midi_mp3_paths.txt')

path_file = open(path_to_txt,'r')

file_reader = csv.reader(path_file, delimiter = '\t')

for row in file_reader:
  midi_path = os.path.join(BASE_PATH,'Clean_MIDIs', row[0])
  mp3_path = os.path.join(BASE_PATH,'audio', row[1])

  try:
      m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_path))
  except:
      print "Error loading {}".format(midi_filename)
      return
  # generate CQT info in standard way
  if os.path.exists(to_cqt_npy(mp3_filename)) and os.path.exists(to_onset_strength_npy(mp3_filename)):
    print "Using pre-existing CQT and onset strength data for {}".format(os.path.split(mp3_filename)[1])
    # Create audio CQT, which is just frame-wise power, and onset strength
    audio_gram = np.load(to_cqt_npy(mp3_filename))
    audio_onset_strength = np.load(to_onset_strength_npy(mp3_filename))
  else:
    print "Creating CQT and onset strength signal for {}".format(os.path.split(mp3_filename)[1])
    audio_gram, audio_onset_strength = align_midi.audio_to_cqt_and_onset_strength(audio, fs=fs)
    np.save(to_cqt_npy(mp3_filename), audio_gram)
    np.save(to_onset_strength_npy(mp3_filename), audio_onset_strength)


  if os.path.exists(to_cqt_npy(midi_filename)):
    midi_gram = np.load(to_cqt_npy(midi_filename))
  else:
    print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
    midi_gram = make_midi_cqt(midi_filename, piano,chroma, m)

  midi_beats, bpm = align_midi.midi_beat_track(m)
  audio_beats = librosa.beat.beat_track(onset_envelope=audio_onset_strength, hop_length=512/4, bpm=bpm)[1]/4
  # Beat-align and log/normalize the audio CQT
  audio_gram = align_midi.post_process_cqt(audio_gram, audio_beats)
  # Get similarity matrix
  similarity_matrix = scipy.spatial.distance.cdist(midi_gram.T, audio_gram.T, metric='cosine')
  # Get best path through matrix
  p, q, score = align_midi.dpmod(similarity_matrix,experimental = False, forceH = False)
