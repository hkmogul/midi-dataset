import os
import sys
import csv

''' Writes out files to figure out which were shifted '''

BASE_PATH = '../data/cal500_txt'
OUTPUT_PATH = 'key-experiment/attempt_with_full_shift-piano_12'
if not os.path.exists('../analytic-files/'):
  os.makedirs('../analytic-files')
zero_shift_files = open('../analytic-files/octave_shift_process-zero_shift.txt', 'w')
other_shift_files = open('../analytic-files/octave_shift_process-other_shift.txt', 'w')


path_to_txt = '../data/cal500_txt/Clean_MIDIs-path_to_cal500_path.txt'
path_file = open(path_to_txt, 'rb')
amt_zero = 0
amt_other = 0
filereader = csv.reader(path_file, delimiter='\t')
for row in filereader:
  midi_filename = BASE_PATH+'/Clean_MIDIs/'+row[0]
  output_midi_filename = midi_filename.replace('Clean_MIDIs', OUTPUT_PATH)
  diagnostic_filename = os.path.splitext(output_midi_filename)[0]+'-interval_diagnostics.txt'
  if os.path.exists(diagnostic_filename):
    diagnostic = open(diagnostic_filename,'r').read()
    if 'Minimum Interval via dot: 0' in diagnostic:
      zero_shift_files.write(output_midi_filename+'\n')
      amt_zero += 1
    else:
      other_shift_files.write(output_midi_filename+'\n')
      amt_other += 1
  else:
    print diagnostic_filename + " not found."
print "Total processed: {} files".format(str(amt_zero + amt_other))
print "Amount of zero shift files: {}".format(amt_zero)
print "Amount of other files: {}".format(amt_other)
zero_shift_files.close()
other_shift_files.close()
