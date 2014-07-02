
import os
import sys
import numpy as np
import glob
from pretty_midi import PrettyMIDI
import midi
from scipy import stats

''' Analyzes some output characteristics of running MIDI alignment,
    including the variance and regressions of the offset over time '''
def linear_fit_offsets(midi_filename, original_midi_filename):
  new_midi = PrettyMIDI(midi.read_midifile(midi_filename))
  old_midi = PrettyMIDI(midi.read_midifile(original_midi_filename))
  note_ons = np.array([note.start for instrument in old_midi.instruments for note in instrument.events])
  aligned_note_ons = np.array([note.start for instrument in new_midi.instruments for note in instrument.events])
  # diff = np.subtract(aligned_note_ons,note_ons)
  diff = np.array([[]])
  last = min(note_ons.shape[0],aligned_note_ons.shape[0])
  for i in xrange(last):
    diff = np.append(diff,np.array([aligned_note_ons[i]-note_ons[i]]))
  print diff.shape
  print note_ons[0:last-1].shape
  slope, intercept, r_value, p_value, std_err = stats.linregress(note_ons[0:(last)],diff)
  return slope, intercept, r_value, std_err


BASE_PATH = '../data/sanity'
OUTPUT_PATH = os.path.join(BASE_PATH,'midi-alignment-exp-force-piano')
midi_glob = glob.glob(OUTPUT_PATH+'/*.mid')
for midi_filename in midi_glob:
  # get original midi filename
  original_midi_filename = os.path.join(BASE_PATH, 'midi',os.path.basename(midi_filename))
  slope, intercept, r_value, std_err = linear_fit_offsets(midi_filename, original_midi_filename)
  print os.path.basename(midi_filename)+ " statistics: "
  print "slope: {}".format(slope)
  print "intercept: {}".format(intercept)
  print "r_value: {}".format(r_value)
  print "standard error: {}".format(std_err)
