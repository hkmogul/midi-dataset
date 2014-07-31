import os
import numpy as np
import sys
import scipy.io
import pretty_midi
import scipy.stats
import librosa
import pretty_midi
import scipy.signal

''' Module of functions for running comparisons of different paths '''

def compare_paths(p1,q1,p2,q2, percent_in = 0):
  ''' Enter in path vectors of two different alignments, get percent difference of the paths '''
  # first easy check,if the shapes are different no need to iterate through
  nErrors = 0
  total_possible = 1
  if p1.shape[0] != p2.shape[0] or q1.shape[0] != q2.shape[0]:
    nErrors += 1

  pq1 = np.rot90(np.vstack((p1,q1)),3)

  pq2 = np.rot90(np.vstack((p2,q2)),3)

  # we just want last 90% of this
  mat_list_1= pq1.tolist()
  mat_list_2 = pq2.tolist()
  # for pair in mat_list_1:
  #   if pair not in mat_list_2:
  #     nErrors += 1
  #   total_possible += 1

  for i in xrange(len(mat_list_1)-1):
    if i > .01*percent_in*(len(mat_list_1)-1):
      pair = mat_list_1[i]
      if pair not in mat_list_2:
        nErrors += 1
      total_possible += 1


  # print nErrors
  # print total_possible
  return float(nErrors)/total_possible


def get_unweighted_score(p,q,similarity_matrix):
  ''' Calculates what the score would have been without the penalty '''
  score = 0
  for i in xrange(p.shape[0]):
    index1 = p[i]
    index2 = q[i]
    score = score + similarity_matrix[index1,index2];
  return score/p.shape[0];


def get_offsets(m_aligned, m):
  ''' Forms and returns alignment offsets based on the adjusted midi and the original '''
  note_ons = np.array([note.start for instrument in m.instruments for note in instrument.notes])
  aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.notes])
  diff = np.array([[]])
  last = min(note_ons.shape[0],aligned_note_ons.shape[0])
  for i in xrange(last):
    diff = np.append(diff,np.array([aligned_note_ons[i]-note_ons[i]]))
  return diff, note_ons[0:last]

def get_cost_path(p,q,similarity_matrix):
  ''' Forms and returns the cost per step of alignment path '''
  cost_path = np.zeros((0,))
  for i in xrange(p.shape[0]):
    cost_path = np.append(cost_path, similarity_matrix[p[i],q[i]])
  return cost_path

def get_regression_stats(m_aligned,m, offsets = None, note_ons = None):
  ''' Used for performing linear regression stats on the aligned offsets '''
  if offsets == None or note_ons == None:
    offsets, note_ons = get_offsets(m_aligned, m)

  else:
    note_ons = np.array([note.start for instrument in m.instruments for note in instrument.notes])
    aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.notes])
    last = min(note_ons.shape[0],aligned_note_ons.shape[0])

  last = min(note_ons.shape[0], offsets.shape[0])
  slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(note_ons[0:last-1],offsets[0:last-1])
  return slope, intercept, r_value, p_value, std_err


def get_non_diagonal_steps(p,q):
  ''' Returns number of non-diagonal steps in path
  both horizontal and vertical are returned as 2 different values '''

  # first, get number of horizontal steps from p
  horiz = 0
  current = p[0]
  for i in xrange(1,p.shape[0]):
    if p[i] == current:
      horiz += 1
    else:
      current = p[i]

  # get verticals the same way
  vert = 0
  current = q[0]
  for j in xrange(1, q.shape[0]):
    if q[j] == current:
      vert +=1
    else:
      current = q[j]
  return horiz, vert

def parabola_fit(cost_path):
  ''' Returns polynomial coefficients and fit value for parabolic fit of data '''
  x = np.arange(start = 0, stop = cost_path.shape[0])
  p = np.polyfit(x =x, y =cost_path, deg = 2)
  # build residuals because apparently numpy just gives the sum of them, and actual parabola because why not
  parab = p[2]+p[1]*x+p[0]*x**2
  # residuals = np.zeros(x.shape)
  residuals = np.subtract(cost_path, parab)
  # for i in xrange(residuals.shape[0]):
  #   residuals[i] = cost_path[i]-parab[i]
  return p, parab,residuals

def pad_lesser_vec(vec1, vec2):
  ''' Returns input vectors with the one of greater length unchanged, and
      the lesser padded with zeros at end. Also returns number of indices
      it added for padding '''
  amt = 0
  if vec1.shape[0]  == vec2.shape[0]:
    return vec1, vec2, amt
  elif vec1.shape[0] > vec2.shape[0]:
    while vec1.shape[0] > vec2.shape[0]:
      vec2 = np.append(vec2, 0)
      amt += 1
    return vec1, vec2, amt
  else:
    while vec2.shape[0] > vec1.shape[0]:
      vec1 = np.append(vec1,0)
      amt+=1
    return vec1, vec2, amt

def truncate_greater_vec(vec1, vec2):
  ''' Returns input vectors with the one of shorter length unchanged, and the
      greater one's extra indices removed. Also returns amount it shortened it by.'''
  amt = 0
  if vec1.shape[0]  == vec2.shape[0]:
    return vec1, vec2, amt
  elif vec1.shape[0] > vec2.shape[0]:
    while vec1.shape[0] > vec2.shape[0]:
      vec1 = vec1[:-1]
    return vec1, vec2, amt
  else:
    while vec2.shape[0] > vec1.shape[0]:
      vec2 = vec2[:-1]
      amt+=1
    return vec1, vec2, amt

def util_print(data_y, data_names):
  for i in xrange(data_y.shape[0]):
    print '{0}, {1}'.format(data_names[i], data_y[i])



def get_X(mat_file_path, mp3_path, old_midi_path, aligned_midi_path):
  ''' gets the current version of the vector used for the classifier '''
  vec = np.zeros((0,))

  # load all relevant files
  mat_file = scipy.io.loadmat(mat_file_path)
  comp_mp3 = librosa.load(mp3_path, mono = 0)
  aligned_midi = pretty_midi.PrettyMIDI(aligned_midi_path)
  old_midi = pretty_midi.PrettyMIDI(old_midi_path)
  # 1. weighted score
  score = mat_file['score']

  vec = np.append(vec, score)
  # 2. magnitude of sim_mat
  sim_mat = mat_file['similarity_matrix']
  norm_mat = np.sum(sim_mat)/(sim_mat.shape[0]*sim_mat.shape[1])

  vec = np.append(vec, norm_mat)
  # 3-5 - r value, stderr, and intercept of offset path
  offsets, note_ons = get_offsets(aligned_midi, old_midi)
  slope, intercept, r, p_err, stderr = get_regression_stats(aligned_midi, old_midi, offsets, note_ons)

  vec = np.append(vec, r)

  vec = np.append(vec, stderr)

  vec = np.append(vec, intercept)

  # 6 variance of all offsets

  vec = np.append(vec, np.var(offsets))


  # 7-9 r, stderr, intercept of first 20% of offsets
  slope, intercept, r, p_err, stderr = get_regression_stats(aligned_midi, old_midi, offsets[0:offsets.shape[0]*.2], note_ons[0:offsets.shape[0]*.2])

  vec = np.append(vec, r)


  vec = np.append(vec, stderr)


  vec = np.append(vec, intercept)

  # 10-11 variance of cost path and filtered cost path
  cost_path = get_cost_path(mat_file['p'],mat_file['q'],sim_mat)

  vec = np.append(vec, np.var(cost_path))

  cost_path_filtered = np.copy(cost_path)
  size = cost_path_filtered.shape[0]/2
  if size % 2 == 0:
    size +=1

  cost_path_filtered = scipy.signal.medfilt(cost_path, kernel_size = size)
  start = int(cost_path_filtered.shape[0]*.05)
  end = int(cost_path_filtered.shape[0]*.95)
  cost_path_filtered = cost_path_filtered[start:end]

  vec = np.append(vec, np.var(cost_path_filtered))

  # 12/13 np.var(residuals) from doing parabolic fit between filtered and original
  p, parab,residuals = parabola_fit(cost_path_filtered)
  vec = np.append(vec, np.var(residuals))


  vec = np.append(vec, np.var(np.subtract(cost_path[start:end], parab)))

  # 14 ratio of end time to max of first 10 percent of offset
  first10 = offsets[0:int(.1*offsets.shape[0])]
  vec = np.append(vec, float(np.amax(first10))/old_midi.get_end_time())

  # cosine distance of loaded comparison files
  beat_path = '../data/beat_info'
  comp_audio, fs = librosa.load(mp3_path, mono = False)
  # print "shape of comp_audio {}".format(comp_audio.shape)
  midi_audio = comp_audio[0,:]
  mp3_audio = comp_audio[1,:]
  cosine = np.dot(midi_audio, mp3_audio)/(np.linalg.norm(midi_audio)*np.linalg.norm(mp3_audio))
  beat_mat_path = os.path.join(beat_path, os.path.splitext(os.path.basename(mp3_path))[0]+'.mat')
  # print "beat mat path {}".format(beat_mat_path)
  # beat track differences
  # cache for other runs
  if not os.path.exists(beat_mat_path):
    m_tempo, m_beats = librosa.beat.beat_track(midi_audio, fs)
    a_tempo, a_beats = librosa.beat.beat_track(mp3_audio, fs)
    scipy.io.savemat(beat_mat_path, {'m_beats': m_beats,
                                                             'm_tempo': m_tempo,
                                                             'a_beats': a_beats,
                                                             'a_tempo': a_tempo})
  else:
    beat_mat = scipy.io.loadmat(beat_mat_path)
    m_tempo = beat_mat['m_tempo'][0,0]
    m_beats = beat_mat['m_beats'][0,:]
    a_tempo = beat_mat['a_tempo'][0,0]
    a_beats = beat_mat['a_beats'][0,:]
  m_beatsP, a_beatsP, amt_pad = pad_lesser_vec(m_beats, a_beats)
  m_beats, a_beats, amt_trunc = truncate_greater_vec(m_beats, a_beats)
  beat_diff = librosa.frames_to_time(np.absolute(np.subtract(m_beats, a_beats)))
  temp_mod = abs((m_tempo%2)-(a_tempo%2))
  beat_cosine = np.dot(m_beats, a_beats)/(np.linalg.norm(m_beats)*np.linalg.norm(a_beats))

  # aggregate features and targets
  vec = np.append(vec, cosine)

  vec = np.append(vec, beat_cosine)


  vec = np.append(vec, temp_mod)

  vec = np.append(vec, abs(m_tempo-a_tempo))

  vec = np.append(vec, amt_pad)

  print vec.shape
  return vec
