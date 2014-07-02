import os
import sys
import glob
import csv
print len(glob.glob('../data/cal500_txt/audio/*.mp3'))
audio_glob = glob.glob('../data/cal500_txt/audio/*.mp3')
midi_glob = glob.glob('../data/call500_txt/Clean_MIDIs/*/*.mid')

with open('../data/cal500_txt/Clean_MIDIs-path_to_cal500_path.txt') as file:
  filereader = csv.reader(file, delimiter = '\t')
  amt = 0
  for row in filereader:
    amt += 1
    print row
  print amt
