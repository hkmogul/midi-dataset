# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import librosa
import copy
import numba
import scipy.ndimage

# <codecell>

# @numba.jit
def dpcore(M, pen, experimental = False,forceH = False, forceV = False):
    '''
    Helper function for populating path cost and traceback matrices
    '''
    # Matrix of local costs, initialized to input matrix
    D = np.copy(M, order='C')
    # Store the traceback
    phi = np.zeros(D.shape)
    changed_yet = False
    if experimental:
      pen = float(np.amax(M))
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # make a way that this forces the first moves to be horizontal or vertical?
    if not forceH and not forceV:
      for i in xrange(D.shape[0] - 1):
          for j in xrange(D.shape[1] - 1):
              # Diagonal move (which has no penalty) is lowest
              if D[i, j] <= D[i, j + 1] + pen and D[i, j] <= D[i + 1, j] + pen:
                  phi[i + 1, j + 1] = 0
                  D[i + 1, j + 1] += D[i, j]
              # Horizontal move (has penalty)
              elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] + pen <= D[i, j]:
                  phi[i + 1, j + 1] = 1
                  D[i + 1, j + 1] += D[i, j + 1] + pen
              # Vertical move (has penalty)
              elif D[i + 1, j] <= D[i, j + 1] and D[i + 1, j] + pen <= D[i, j]:
                  phi[i + 1, j + 1] = 2
                  D[i + 1, j + 1] += D[i + 1, j] + pen
              if not changed_yet and experimental:
                if float(i)/D.shape[1] >= .20: #if we are at least 10% through iteration, lower penalty again
                  pen = float(np.percentile(M,80))
                changed_yet = True
    elif forceH:
      # if forceH is true, we want the first move to be a horizontal one

      for i in xrange((D.shape[0] - 1)):
          for j in xrange((D.shape[1] - 1)):
              # Diagonal move (which has no penalty) is lowest
              if j < (.1*(D.shape[1]-1)):
                phi[i + 1, j + 1] = 1
                D[i + 1, j + 1] += D[i, j + 1] + pen
              else:
                if D[i, j] <= D[i, j + 1] + pen and D[i, j] <= D[i + 1, j] + pen:
                    phi[i + 1, j + 1] = 0
                    D[i + 1, j + 1] += D[i, j]
                # Horizontal move (has penalty)
                elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] + pen <= D[i, j]:
                    phi[i + 1, j + 1] = 1
                    D[i + 1, j + 1] += D[i, j + 1] + pen
                # Vertical move (has penalty)
                elif D[i + 1, j] <= D[i, j + 1] and D[i + 1, j] + pen <= D[i, j]:
                    phi[i + 1, j + 1] = 2
                    D[i + 1, j + 1] += D[i + 1, j] + pen
                if not changed_yet and experimental:
                  if float(i)/D.shape[1] >= .20: #if we are at least 10% through iteration, lower penalty again
                    pen = float(np.percentile(M,90))
                  changed_yet = True
    else:
      for i in xrange((D.shape[0] - 1)):
          for j in xrange((D.shape[1] - 1)):
              if j < .05*(D.shape[1]-1):
                phi[i + 1, j + 1] = 2
                D[i + 1, j + 1] += D[i + 1, j] + pen
              else:
              # Diagonal move (which has no penalty) is lowest
                if D[i, j] <= D[i, j + 1] + pen and D[i, j] <= D[i + 1, j] + pen:
                    phi[i + 1, j + 1] = 0
                    D[i + 1, j + 1] += D[i, j]
                # Horizontal move (has penalty)
                elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] + pen <= D[i, j]:
                    phi[i + 1, j + 1] = 1
                    D[i + 1, j + 1] += D[i, j + 1] + pen
                # Vertical move (has penalty)
                elif D[i + 1, j] <= D[i, j + 1] and D[i + 1, j] + pen <= D[i, j]:
                    phi[i + 1, j + 1] = 2
                    D[i + 1, j + 1] += D[i + 1, j] + pen
                if not changed_yet and experimental:
                  if float(i)/D.shape[1] >= .20: #if we are at least 10% through iteration, lower penalty again
                    pen = float(np.percentile(M,90))
                  changed_yet = True
    return D, phi

# <codecell>

def dpmod(M, gully=.95, pen=None, experimental= False, forceH = False, forceV = False):
    '''
    Use dynamic programming to find a min-cost path through matrix M.

    Input:
        M - Matrix to find path through
        gully - Sequences must match up to this proportion of shorter sequence, default .95
        pen - additional cost for for (0,1) and (1,0) steps, default None which means np.percentile(M, 90)
    Output:
        p, q - State sequence
        score - DP score
    '''

    # Set penality = median(M) if none was provided
    if pen is None:
        pen = np.percentile(M, 90)

    pen = float(pen)

    # Compute path cost matrix
    D, phi = dpcore(M, pen,experimental, forceH, forceV)

    # Traceback from lowest-cost point on bottom or right edge
    gully = int(gully*min(D.shape[0], D.shape[1]))
    i = np.argmin(D[gully:, -1]) + gully
    j = np.argmin(D[-1, gully:]) + gully

    if D[-1, j] > D[i, -1]:
        j = D.shape[1] - 1
    else:
        i = D.shape[0] - 1

    # Score is the final score of the best path
    score = D[i, j]

    # These vectors will give the lowest-cost path
    p = np.array([i])
    q = np.array([j])

    # Until we reach an edge
    while i > 0 and j > 0:
        # If the tracback matrix indicates a diagonal move...
        if phi[i, j] == 0:
            i = i - 1
            j = j - 1
        # Horizontal move...
        elif phi[i, j] == 1:
            i = i - 1
        # Vertical move...
        elif phi[i, j] == 2:
            j = j - 1
        # Add these indices into the path arrays
        p = np.append(i, p)
        q = np.append(j, q)

    # Normalize score
    score = score/q.shape[0]

    return p, q, score

# <codecell>

def maptimes(t, intime, outtime):
    '''
    map the times in t according to the mapping that each point in intime corresponds to that value in outtime
    2008-03-20 Dan Ellis dpwe@ee.columbia.edu

    Input:
        t - list of times to map
        intimes - original times
        outtime - mapped time
    Output:
        u - mapped version of t
    '''

    # Make sure both time ranges start at or before zero
    pregap = max(intime[0], outtime[0])
    intime = np.append(intime[0] - pregap, intime)
    outtime = np.append(outtime[0] - pregap, outtime)

    # Make sure there's a point beyond the end of both sequences
    din = np.diff(np.append(intime, intime[-1] + 1))
    dout = np.diff(np.append(outtime, outtime[-1] + 1))

    # Decidedly faster than outer-product-array way
    u = np.array(t)
    for i in xrange(t.shape[0]):
      ix = -1 + np.min(np.append(np.flatnonzero(intime > t[i]), outtime.shape[0]))
      # offset from that time
      dt = t[i] - intime[ix];
      # perform linear interpolation
      u[i] = outtime[ix] + (dt/din[ix])*dout[ix]
    return u

# <codecell>

def midi_to_cqt(midi, sf2_path=None, fs=22050, hop=512):
    '''
    Feature extraction routine for midi data, converts to a drum-free, percussion-suppressed CQT.

    Input:
        midi - pretty_midi.PrettyMIDI object
        sf2_path - path to .sf2 file to pass to pretty_midi.fluidsynth
        fs - sampling rate to synthesize audio at, default 22050
        hop - hop length for cqt, default 512
    Output:
        midi_gram - Simulated CQT of the midi data
    '''
    # Create a copy of the midi object
    midi_no_drums = copy.deepcopy(midi)
    # Remove the drums
    for n, instrument in enumerate(midi_no_drums.instruments):
        if instrument.is_drum:
            del midi_no_drums.instruments[n]
    # Synthesize the MIDI using the supplied sf2 path
    midi_audio = midi_no_drums.fluidsynth(fs=fs, sf2_path=sf2_path)
    # midi_audio = midi_no_drums.synthesize(fs = fs)
    # Use the harmonic part of the signal
    H, P = librosa.decompose.hpss(librosa.stft(midi_audio))
    midi_audio_harmonic = librosa.istft(H)
    # Compute log frequency spectrogram of audio synthesized from MIDI
    midi_gram = np.abs(librosa.cqt(y=midi_audio_harmonic,
                                   sr=fs,
                                   hop_length=hop,
                                   fmin=librosa.midi_to_hz(36),
                                   n_bins = 60,
                                   tuning=0.0))**2
    return midi_gram

# <codecell>
def midi_to_piano_cqt(midi):
  piano_roll = midi.get_piano_roll(times = librosa.frames_to_time(np.arange(midi.get_end_time()*22050/512)))
  piano_subset = piano_roll[36:96]+1e-10 #want just C3 to C8 of piano roll
  return piano_subset

def piano_roll_fuzz(piano_roll):
  ''' Fuzzes a CQT emulating piano roll so the note-ons look more like synthesized notes '''
  fuzzed_piano = np.copy(piano_roll)
  # off notes are -1, on are nearing 0 (like dB)
  # i denotes column, j is row
  for i in xrange(piano_roll.shape[1]):
      col = piano_roll[:,i]
      for j in xrange(col.shape[0]):
          if col[j] != -1.0:
              # if j < col.shape[0]-2 :
                  # fuzzed_piano[j+2,i] = col[j]/0.06
              if j < col.shape[0]-1 :
                  fuzzed_piano[j+1,i] = col[j]/.2
              if j > 1:
                  # fuzzed_piano[j-2,i] = col[j]/.06
                  fuzzed_piano[j-1,i] = col[j]/.2
              elif j > 0:
                  fuzzed_piano[j-1,i] = col[j]/.2
  return fuzzed_piano

def clean_audio_gram(audio_gram, threshold = None):
  ''' Sets any low valued cells to the min of the graph '''
  min_value = np.amin(audio_gram)
  clean_gram = np.copy(audio_gram)
  if threshold is None:
    threshold = np.percentile(audio_gram, 20)
  for i in xrange(audio_gram.shape[1]):
    col = audio_gram[:,i]
    for j in xrange(col.shape[0]):
      if col[j] <= threshold:
        clean_gram[j,i] = -1
  return clean_gram

def midi_to_chroma(midi):
  return midi.get_chroma(times = librosa.frames_to_time(np.arange(midi.get_end_time()*22050/512)))
  # return midi.get_chroma()


def shift_cqt(cqt, interval):
    ''' Shifts a cqt matrix by the given interval '''

    # If we are shifting the cqt down, we need to replace the first rows with
    # zero vectors.  If we are shifting it upwards, we need to replace the last
    # rows with zero vectors.
    if interval != 0:
      min_value = np.amin(cqt)
      fill_array = min_value*np.ones((abs(interval), cqt.shape[1]))
      if interval > 0:
        cqt_slice = cqt[0:(cqt.shape[0]-interval)]
        new_cqt = np.vstack((fill_array, cqt_slice))
      else:
        cqt_slice = cqt[abs(interval):]
        new_cqt = np.vstack((cqt_slice, fill_array))
    else:
        new_cqt = cqt
    return new_cqt

def audio_to_cqt_and_onset_strength(audio, fs=22050, hop=512):
    '''
    Feature extraction for audio data.
    Gets a power CQT of harmonic component and onset strength signal of percussive.

    Input:
        midi - pretty_midi.PrettyMIDI object
        fs - sampling rate to synthesize audio at, default 22050
        hop - hop length for cqt, default 512, onset strength hop will be 1/4 of this
    Output:
        audio_gram - CQT of audio data
        audio_onset_strength - onset strength signal
    '''
    # Use harmonic part for gram, percussive part for onsets
    H, P = librosa.decompose.hpss(librosa.stft(audio))
    audio_harmonic = librosa.istft(H)
    audio_percussive = librosa.istft(P)
    # Compute log-frequency spectrogram of original audio
    audio_gram = np.abs(librosa.cqt(y=audio_harmonic,
                                    sr=fs,
                                    hop_length=hop,
                                    fmin=librosa.midi_to_hz(36),
                                    n_bins = 60))**2

    # Beat track the audio file at 4x the hop rate
    audio_onset_strength = librosa.onset.onset_strength(audio_percussive , hop_length=hop/4, sr=fs)
    return audio_gram, audio_onset_strength

# <codecell>
def audio_to_chroma_and_onset_strength(audio, fs = 22050, hop = 512):
  H,P = librosa.decompose.hpss(librosa.stft(audio))
  audio_harmonic = librosa.istft(H)
  audio_percussive = librosa.istft(P)
  chroma_gram = librosa.feature.chromagram(audio_harmonic)
  audio_onset_strength = librosa.onset.onset_strength(audio_percussive, hop_length = hop/4, sr = fs)
  return chroma_gram, audio_onset_strength

def midi_beat_track(midi, fs=22050, hop=512.):
    '''
    Perform midi beat tracking and force the tempo to be high

    Input:
        midi - pretty_midi.PrettyMIDI object
        fs - sample rate to sample beats with
        hop - hop size to sample beats with
    Output:
        midi_beats - np.array of beat times, in frames, with sample rate fs and hop size 512
        midi_tempo - tempo, at least 240 bpm
    '''
    # Estimate MIDI beat times
    midi_beats = np.array(midi.get_beats()*fs/hop, dtype=np.int)
    # Estimate the MIDI tempo
    midi_tempo = 60.0/np.mean(np.diff(midi.get_beats()))
    # Make tempo faster for better temporal resolution
    scale = 1
    while midi_tempo < 240:
        midi_tempo *= 2
        scale *= 2
    # Interpolate the beats to match the higher tempo
    midi_beats = np.array(np.interp(np.linspace(0, scale*(midi_beats.shape[0] - 1), scale*(midi_beats.shape[0] - 1) + 1),
                                    np.linspace(0, scale*(midi_beats.shape[0] - 1), midi_beats.shape[0]),
                                    midi_beats), dtype=np.int)
    return midi_beats, midi_tempo

# <codecell>

def post_process_cqt(gram, beats):
    '''
    Given a power CQT, beat-synchronize it, take log, and normalize

    Input:
        gram - np.ndarray, power CQT
        beats - np.ndarray, beat locations in frame number
    Output:
        gram_normalized - CQT, normalized and beat-synchronized
    '''
    # Truncate to length of audio
    truncated_beats = beats[beats < gram.shape[1]]
    # Synchronize the log-fs gram with MIDI beats
    synchronized_gram = librosa.feature.sync(gram, truncated_beats)[:, 1:]
    # Compute log-amplitude spectrogram
    log_gram = librosa.logamplitude(synchronized_gram, ref_power=synchronized_gram.max())
    # Normalize columns and return
    return librosa.util.normalize(log_gram, axis=0)

# <codecell>

def adjust_midi(midi, original_times, new_times):
    '''
    Wrapper function to adjust all time locations in a midi object using maptimes

    Input:
        midi - pretty_midi.PrettyMIDI object
        original_times - np.ndarray of reference times
        new_times - np.ndarray of times to map to
    Output:
        aligned_midi - midi object with its times adjusted
    '''
    # Get array of note-on locations and correct them
    note_ons = np.array([note.start for instrument in midi.instruments for note in instrument.events])
    aligned_note_ons = maptimes(note_ons, original_times, new_times)
    # Same for note-offs
    note_offs = np.array([note.end for instrument in midi.instruments for note in instrument.events])
    aligned_note_offs = maptimes(note_offs, original_times, new_times)
    # Same for pitch bends
    pitch_bends = np.array([bend.time for instrument in midi.instruments for bend in instrument.pitch_bends])
    aligned_pitch_bends = maptimes(pitch_bends, original_times, new_times)
    # Create copy (not doing this in place)
    midi_aligned = copy.deepcopy(midi)
    # Correct notes
    for n, note in enumerate([note for instrument in midi_aligned.instruments for note in instrument.events]):
        note.start = (aligned_note_ons[n] > 0)*aligned_note_ons[n]
        note.end = (aligned_note_offs[n] > 0)*aligned_note_offs[n]
    # Correct pitch changes
    for n, bend in enumerate([bend for instrument in midi_aligned.instruments for bend in instrument.pitch_bends]):
        bend.time = (aligned_pitch_bends[n] > 0)*aligned_pitch_bends[n]
    return midi_aligned
