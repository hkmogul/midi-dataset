
import os
import sys
import numpy as np
from pretty_midi import PrettyMIDI
import midi
from scipy import stats
import csv

''' Analyzes some output characteristics of running MIDI alignment,
    including the variance and regressions of the offset over time '''
def linear_fit_offsets(new_midi= None, old_midi = None,midi_filename = None, original_midi_filename = None):
  if new_midi == None:
    new_midi = PrettyMIDI(midi.read_midifile(midi_filename))
  if old_midi == None:
    old_midi = PrettyMIDI(midi.read_midifile(original_midi_filename))
  note_ons = np.array([note.start for instrument in old_midi.instruments for note in instrument.events])
  aligned_note_ons = np.array([note.start for instrument in new_midi.instruments for note in instrument.events])
  # diff = np.subtract(aligned_note_ons,note_ons)
  diff = np.array([[]])
  last = min(note_ons.shape[0],aligned_note_ons.shape[0])
  for i in xrange(last):
    diff = np.append(diff,np.array([aligned_note_ons[i]-note_ons[i]]))
  slope, intercept, r_value, p_value, std_err = stats.linregress(note_ons[0:(last)],diff)
  return slope, intercept, r_value, std_err, diff

def simple_statistics(new_midi= None, old_midi = None,midi_filename = None, original_midi_filename = None, diff = None):
  if diff == None:
    if new_midi == None:
      new_midi = PrettyMIDI(midi.read_midifile(midi_filename))
    if old_midi == None:
      old_midi = PrettyMIDI(midi.read_midifile(original_midi_filename))
    note_ons = np.array([note.start for instrument in old_midi.instruments for note in instrument.events])
    aligned_note_ons = np.array([note.start for instrument in new_midi.instruments for note in instrument.events])
    # diff = np.subtract(aligned_note_ons,note_ons)
    diff = np.array([[]])
    last = min(note_ons.shape[0],aligned_note_ons.shape[0])
    for i in xrange(last):
      diff = np.append(diff,np.array([aligned_note_ons[i]-note_ons[i]]))
  diff_max = np.amax(diff)
  diff_min = np.amin(diff)
  diff_mean = np.mean(diff)
  diff_dev = np.std(diff)
  return diff_max, diff_min, diff_mean, diff_dev

BASE_PATH = '../data/sanity'
OUTPUT_PATH = os.path.join(BASE_PATH,'midi-alignment-exp-force-piano')
txt_path = os.path.join(BASE_PATH,'sanity_paths.txt')
path_file = open(txt_path, 'rb')
filereader = csv.reader(path_file, delimiter = '\t')
for row in filereader:
  basename = row[0]
  midi_filename = os.path.join(OUTPUT_PATH, basename)
  # get original midi filename
  original_midi_filename = os.path.join(BASE_PATH, 'midi',basename)
  new_midi = PrettyMIDI(midi.read_midifile(midi_filename))
  old_midi = PrettyMIDI(midi.read_midifile(original_midi_filename))
  slope, intercept, r_value, std_err, diff = linear_fit_offsets(new_midi = new_midi, old_midi = old_midi)
  diff_max, diff_min, diff_mean, diff_dev = simple_statistics(diff = diff)

  print os.path.basename(midi_filename)+ " alignment statistics: "
  print "slope: {}".format(slope)
  print "intercept: {}".format(intercept)
  print "r_value: {}".format(r_value)
  print "standard error: {}".format(std_err)
  print "----"
  print "Max difference: {}".format(diff_max)
  print "Min difference: {}".format(diff_min)
  print "Unweighted mean of difference: {}".format(diff_mean)
  print "Standard dev of difference: {}".format(diff_dev)
  print "----------"
path_file.close()
