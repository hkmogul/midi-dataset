# This version is different by fully shifting instruments instead of piano roll
# <codecell>

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


OUTPUT_PATH = 'key-experiment/attempt_with_full_shift'
piano = False
write_mp3 = False
use_prev_data = False# choice of using preexisting data
initial_interval = 12
if '-w' in sys.argv:
  write_mp3 = True
if '-p' in sys.argv:
  OUTPUT_PATH = OUTPUT_PATH + '-piano'
  piano = True
if '-u' in sys.argv:
  use_prev_data = True
if '-i' in sys.argv:
  initial_interval += -1
# <codecell>
OUTPUT_PATH = OUTPUT_PATH+'_'+str(initial_interval)

SF2_PATH = '../../Performer Synchronization Measure/SGM-V2.01.sf2'
BASE_PATH = '../data/cal500_txt'
if not os.path.exists(os.path.join(BASE_PATH, OUTPUT_PATH)):
    os.makedirs(os.path.join(BASE_PATH, OUTPUT_PATH))

# <codecell>

def shift_midi(midi, interval):
  for instrument in midi.instruments:
    # Check whether the instrument is a drum track
    if not instrument.is_drum:
    # Iterate over note events for this instrument
      for note in instrument.events:
        # Shift them up by 4 semitones
        note.pitch += interval
  return midi

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
def to_piano_cqt_npy(filename):
    return os.path.splitext(midi_filename)[0]+'-piano.npy'
# <codecell>

def make_midi_cqt(midi_filename, piano, midi_info = None):
  if midi_info is None:
    midi_info = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
  if piano:
    # print "Using piano roll for CQT"
    midi_gram = align_midi.midi_to_piano_cqt(midi_info)
    midi_beats, bpm = align_midi.midi_beat_track(midi_info)
    midi_gram = align_midi.post_process_cqt(midi_gram, midi_beats)
    piano_cqt_path = os.path.splitext(midi_filename)[0]+'-piano.npy'
    # np.save(to_piano_cqt_npy(midi_filename), midi_gram)
    return midi_gram
  else:
    # print "Synthesizing MIDI for CQT"
    midi_gram = align_midi.midi_to_cqt(midi_info, SF2_PATH)
    # Get beats
    midi_beats, bpm = align_midi.midi_beat_track(midi_info)
    # Beat synchronize and normalize
    midi_gram = align_midi.post_process_cqt(midi_gram, midi_beats)
    # np.save(to_cqt_npy(midi_filename),midi_gram)
    return midi_gram


def align_one_file(mp3_filename, midi_filename, output_midi_filename, output_diagnostics=True, initial_interval=0):
    '''
    Helper function for aligning a MIDI file to an audio file.

    :parameters:
        - mp3_filename : str
            Full path to a .mp3 file.
        - midi_filename : str
            Full path to a .mid file.
        - output_midi_filename : str
            Full path to where the aligned .mid file should be written.  If None, don't output.
        - output_diagnostics : bool
            If True, also output a .pdf of figures, a .mat of the alignment results,
            and a .mp3 of audio and synthesized aligned audio
    '''
    # Load in the corresponding midi file in the midi directory, and return if there is a problem loading it
    try:
        m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
    except:
        print "Error loading {}".format(midi_filename)
        return

    print "Aligning {}".format(os.path.split(midi_filename)[1])

    #check if output path exists, and create it if necessary
    if not os.path.exists(os.path.split(output_midi_filename)[0]):
      os.makedirs(os.path.split(output_midi_filename)[0])

    # Cache audio CQT and onset strength

    # Don't need to load in audio multiple times
    # if mp3_filename is None:
    #   filename = os.path.basename(midi_filename)
    #   filename_raw = os.path.splitext(filename)[0]
    #   mp

    audio, fs = librosa.load(mp3_filename)
    if use_prev_data:
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
    else:
      print "Creating CQT and onset strength signal for {}".format(os.path.split(mp3_filename)[1])
      audio_gram, audio_onset_strength = align_midi.audio_to_cqt_and_onset_strength(audio, fs=fs)
      np.save(to_cqt_npy(mp3_filename), audio_gram)
      np.save(to_onset_strength_npy(mp3_filename), audio_onset_strength)

    if use_prev_data and initial_interval == 0:
      if piano:
        if os.path.exists(to_piano_cqt_npy(midi_filename)):
          midi_gram = np.load(to_piano_cqt_npy(midi_filename))
        else:
          print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
          midi_gram = make_midi_cqt(midi_filename, piano, m)
      else:
        if os.path.exists(to_cqt_npy(midi_filename)):
          midi_gram = np.load(to_cqt_npy(midi_filename))
        else:
          print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
          midi_gram = make_midi_cqt(midi_filename, piano, m)
    else:
      print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
      # Generate synthetic MIDI CQT
      if initial_interval == 0:
        midi_gram = make_midi_cqt(midi_filename, piano, m)
      else:
        print "shifting midi at instrument level by {}".format(str(initial_interval))
        m = shift_midi(m, initial_interval)
        midi_gram = make_midi_cqt(midi_filename, piano, m)



    # Compute beats
    midi_beats, bpm = align_midi.midi_beat_track(m)
    audio_beats = librosa.beat.beat_track(onsets=audio_onset_strength, hop_length=512/4, bpm=bpm)[1]/4
    # Beat-align and log/normalize the audio CQT
    audio_gram = align_midi.post_process_cqt(audio_gram, audio_beats)
    # what we're going to do to see if it works:
    #  take a few alignments we know are in key, shift them some amount, see if the checker shifts them back

    # new method of checking "in key-ness"- dot product

    audio_net = np.sum(audio_gram, axis = 1)
    audio_net = audio_net - np.amin(audio_gram)

    midi_net = np.sum(midi_gram, axis = 1)
    midi_net = midi_net - np.amin(midi_net)

    score_list = np.zeros(13)
    diff_list = np.zeros(13)
    #dictionary of data to choose from when finding min for plotting
    # dict = {}
    diagnostic_file = open(os.path.splitext(output_midi_filename)[0]+'-interval_diagnostics.txt','w')
    diagnostic_file.write('we want minimum interval to be '+str(-1*initial_interval)+'\n')
    for interval in range(-6,7,1):

      m_gram = align_midi.shift_cqt(midi_gram,interval)

      midi_net = np.sum(m_gram, axis = 1)
      midi_net = midi_net - np.amin(midi_net)
      # compute cosine similarity
      sim = np.dot(midi_net, audio_net)/(np.linalg.norm(midi_net)*np.linalg.norm(audio_net))
      # and euclidean distance
      diff = np.sum((midi_net-audio_net)**2)/(np.linalg.norm(midi_net)*np.linalg.norm(audio_net))
      score_list[interval+6] = sim
      diff_list[interval+6] = diff

      diagnostic_file.write('Interval: '+str(interval) + '\n')
      diagnostic_file.write('Cosine-similarity: '+ str(sim) + '  Euclidean Distance: '+str(diff))
      diagnostic_file.write('\n')
      diagnostic_file.write('\n')




    min_index = np.argmax(score_list)
    min_diff = np.argmin(diff_list)
    diff_in = min_diff - 6
    dot_in = min_index-6
    interval = dot_in
    diagnostic_file.write('Minimum Interval via dot: '+str(min_index-6))
    diagnostic_file.write('Minimum Interval via euclid: '+str(diff_in))
    if dot_in != interval:
      diagnostic_file.write(' <-LOOK AT THIS!!!!')
      print 'LOOOK AT THIS'
    diagnostic_file.close()

    print interval
    m_gram = align_midi.shift_cqt(midi_gram, interval)
    similarity_matrix = scipy.spatial.distance.cdist(m_gram.T, audio_gram.T, metric='cosine')



    # Plot log-fs grams
    plt.figure(figsize=(36, 24))
    ax = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    plt.title('MIDI Synthesized- originally shifted by {0}, auto fix by {1}'.format(initial_interval, interval))
    librosa.display.specshow(m_gram,
                             x_axis='frames',
                             y_axis='cqt_note',
                             fmin=librosa.midi_to_hz(36),
                             fmax=librosa.midi_to_hz(96))
    ax = plt.subplot2grid((4, 3), (1, 0), colspan=3)
    plt.title('Audio data')
    librosa.display.specshow(audio_gram,
                             x_axis='frames',
                             y_axis='cqt_note',
                             fmin=librosa.midi_to_hz(36),
                             fmax=librosa.midi_to_hz(96))

    # Get similarity matrix
    # Get best path through matrix
    p, q, score = align_midi.dpmod(similarity_matrix)

    # Plot distance at each point of the lowst-cost path
    ax = plt.subplot2grid((4, 3), (2, 0), rowspan=2)
    plt.plot([similarity_matrix[p_v, q_v] for p_v, q_v in zip(p, q)])
    plt.title('Distance at each point on lowest-cost path- Minimum Harmonic Interval: {} Half Steps'.format(interval))

    # Plot similarity matrix and best path through it
    ax = plt.subplot2grid((4, 3), (2, 1), rowspan=2)
    plt.imshow(similarity_matrix.T,
               aspect='auto',
               interpolation='nearest',
               cmap=plt.cm.gray)
    tight = plt.axis()
    plt.plot(p, q, 'r.', ms=.2)
    plt.axis(tight)
    plt.title('Similarity matrix and lowest-cost path, cost={}'.format(score))

    # Adjust MIDI timing
    m_aligned = align_midi.adjust_midi(m, librosa.frames_to_time(midi_beats)[p], librosa.frames_to_time(audio_beats)[q])

    # Plot alignment
    ax = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
    note_ons = np.array([note.start for instrument in m.instruments for note in instrument.events])
    aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.events])
    plt.plot(note_ons, aligned_note_ons - note_ons, '.')
    plt.xlabel('Original note location (s)')
    plt.ylabel('Shift (s)')
    plt.title('Corrected offset')

    # Write out the aligned file
    if output_midi_filename is not None:
        m_aligned.write(output_midi_filename)

    if output_diagnostics:
        # Save the figures
        plt.savefig(output_midi_filename.replace('.mid', '.pdf'))
        if write_mp3:
          # Load in the audio data (needed for writing out)
          audio, fs = librosa.load(mp3_filename, sr=None)
          # Synthesize the aligned midi
          # midi_audio_aligned = m_aligned.fluidsynth()
          midi_audio_aligned = m_aligned.fluidsynth(fs=fs, sf2_path=SF2_PATH)

          # Trim to the same size as audio
          if midi_audio_aligned.shape[0] > audio.shape[0]:
              midi_audio_aligned = midi_audio_aligned[:audio.shape[0]]
          else:
              midi_audio_aligned = np.append(midi_audio_aligned, np.zeros(audio.shape[0] - midi_audio_aligned.shape[0]))
          # Write out to temporary .wav file
          librosa.output.write_wav(output_midi_filename.replace('.mid', '.wav'),
                                   np.vstack([midi_audio_aligned, audio]).T, fs)
          # Convert to mp3
          subprocess.check_output(['ffmpeg',
                           '-i',
                           output_midi_filename.replace('.mid', '.wav'),
                           '-ab',
                           '128k',
                           '-y',
                           output_midi_filename.replace('.mid', '.mp3')])
          # Remove temporary .wav file
          os.remove(output_midi_filename.replace('.mid', '.wav'))
          # Save a .mat of the results
          scipy.io.savemat(output_midi_filename.replace('.mid', '.mat'),
                           {'similarity_matrix': similarity_matrix,
                            'p': p, 'q': q, 'score': score})
    # If we aren't outputting a .pdf, show the plot
    else:
        plt.show()
    plt.close()

# <codecell>

# Parallelization!
mp3_glob = sorted(glob.glob(os.path.join(BASE_PATH, 'audio', '*.mp3')))
midi_glob = sorted(glob.glob(os.path.join(BASE_PATH, 'midi', '*.mid')))

path_to_txt = '../data/cal500_txt/Clean_MIDIs-path_to_cal500_path.txt'
path_file = open(path_to_txt, 'rb')
filereader = csv.reader(path_file, delimiter='\t')
amt = 0
# for row in filereader:
#   amt +=1
# print "Processing {} MIDI files.".format(amt)
for row in filereader:
  midi_filename = BASE_PATH+'/Clean_MIDIs/'+row[0]
  mp3_filename =  BASE_PATH+ '/audio/'+row[1]
  align_one_file(mp3_filename, midi_filename, midi_filename.replace('Clean_MIDIs', OUTPUT_PATH),initial_interval = initial_interval)





# print "Processing {} files.".format(max(len(mp3_glob), len(midi_glob)))
# for mp3_filename, midi_filename in zip(mp3_glob, midi_glob):
#   align_one_file(mp3_filename, midi_filename, midi_filename.replace('midi', OUTPUT_PATH),interval = interval)
