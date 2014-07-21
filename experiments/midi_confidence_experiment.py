import numpy as np
import matplotlib.pyplot as plt
import librosa
import midi
import pretty_midi
import os
import sys
sys.path.append('../')
import align_midi
import alignment_analysis
import scipy.io
import csv
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal



''' Start of post analysis of offset information to gain confidence measure in
    running alignment. Ultimate goal is to have failing alignments come out with
    low alignments, and passing alignments have high ones.'''
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


# size = int(raw_input("Size of median filter (must be odd)? "))
# if size % 2 == 0:
#   size = size+1
#   print "Size was even, incremented by 1 to fix."
BASE_PATH = '../data/cal500_txt/'
cqt_scores_pass  = np.zeros((0,))
cqt_scores_fail = np.zeros((0,))
cqt_scores_passUW = np.zeros((0,))
cqt_scores_failUW = np.zeros((0,))

piano_diff_pass = np.zeros((0,))
piano_diff_fail = np.zeros((0,))
norm_mat_pass = np.zeros((0,))
norm_mat_fail = np.zeros((0,))

max_offset_pass = np.zeros((0,))
max_offset_fail = np.zeros((0,))
offset_deviation_pass = np.zeros((0,))
offset_deviation_fail = np.zeros((0,))

r_offset_pass = np.zeros((0,))
r_offset_fail = np.zeros((0,))

std_err_pass = np.zeros((0,))
std_err_fail = np.zeros((0,))


cost_std_pass = np.zeros((0,))
cost_std_fail = np.zeros((0,))
cost_var_pass = np.zeros((0,))
cost_var_fail = np.zeros((0,))

filt_cost_std_pass = np.zeros((0,))
filt_cost_std_fail = np.zeros((0,))
filt_cost_var_pass = np.zeros((0,))
filt_cost_var_fail = np.zeros((0,))

cost_remain_pass = np.zeros((0,))
cost_remain_fail = np.zeros((0,))
# parabolic regression info -  will store variance of residuals
para_res_pass = np.zeros((0,))
para_res_fail = np.zeros((0,))
orig_res_pass = np.zeros((0,))
orig_res_fail = np.zeros((0,))
half_res_pass = np.zeros((0,))
half_res_fail = np.zeros((0,))

# info on nondiagonal steps
nondiag_pass = np.zeros((0,))
nondiag_fail = np.zeros((0,))

first_offsets_pass = np.zeros((0,))
first_offsets_fail = np.zeros((0,))


# cosine distances of loaded comparisons
cosine_pass = np.zeros((0,))
cosine_fail = np.zeros((0,))

path_to_may_30 = '../../CSV_Analysis/5-30-14_Alignment_Results.csv'
may_30_file = open(path_to_may_30)
csv_may = csv.reader(may_30_file)
csv_may.next()
for row in csv_may:
  # if "Guns N Roses" in row[0] or "Mary Wells" in row[0]:
  #   continue

  title_path = vs_filename_to_path(row[0])
  piano_out = os.path.join(BASE_PATH, 'midi-aligned-additive-dpmod-piano',title_path)
  success = int(row[2])
  mat_out = os.path.join('../../MIDI_Results_5-30',row[0]).replace('.mid', '.mat')+'.mat'
  # load cqt based results
  cqt_mat = scipy.io.loadmat(mat_out)
  p1 = cqt_mat['p'][0,:]
  q1 = cqt_mat['q'][0,:]
  sim_mat_1 = cqt_mat['similarity_matrix']
  score1 = cqt_mat['score'][0,0]
  print "Analyzing {}".format(title_path)

  print "Successful alignment: {}".format(success)
  print "Weighted score: {}".format(score1)
  uw_score1 = alignment_analysis.get_unweighted_score(p1,q1,sim_mat_1)
  print "Unweighted score: {}".format(uw_score1)
  norm_mat1 = np.sum(sim_mat_1)/(sim_mat_1.shape[0]*sim_mat_1.shape[1])
  print "Normalized similarity matrix magnitude: {}".format(norm_mat1)
  print "-----"
  # print "Analyzing piano roll information."
  piano_mat = scipy.io.loadmat(piano_out)
  p2 = piano_mat['p'][0,:]
  q2 = piano_mat['q'][0,:]
  sim_mat_2 = piano_mat['similarity_matrix']
  score2 = piano_mat['score'][0,0]

  # print "Weighted score (piano):{}".format(score2)
  uw_score2 = alignment_analysis.get_unweighted_score(p2,q2,sim_mat_2)
  # print "Unweighted score (piano):{}".format(uw_score2)
  norm_mat2 = np.sum(sim_mat_2)/(sim_mat_2.shape[0]*sim_mat_2.shape[1])
  # print "Normalized similarity matrix magnitude: {}".format(norm_mat2)
  path_diff = alignment_analysis.compare_paths(p1,q1,p2,q2)
  # print "Percent difference in paths: {}".format(path_diff*100)
  # print "--------------------"

  if success == 1:
    cqt_scores_pass = np.append(cqt_scores_pass, score1)
    cqt_scores_passUW =np.append(cqt_scores_passUW, uw_score1)
    norm_mat_pass = np.append(norm_mat_pass, norm_mat1)
    piano_diff_pass = np.append(piano_diff_pass, path_diff)
  else:
    cqt_scores_fail = np.append(cqt_scores_fail, score1)
    cqt_scores_failUW = np.append(cqt_scores_failUW, uw_score1)
    norm_mat_fail = np.append(norm_mat_fail, norm_mat1)
    piano_diff_fail = np.append(piano_diff_fail, path_diff)


  # linear regression on offset
  # first, get original midi path
  old_midi_path = os.path.join(BASE_PATH, 'Clean_MIDIs',title_path.replace('.mat','.mid'))
  aligned_midi_path = os.path.join('../../MIDI_Results_5-30',row[0]+'.mid')
  old_midi = pretty_midi.PrettyMIDI(old_midi_path)
  aligned_midi = pretty_midi.PrettyMIDI(aligned_midi_path)
  offsets = alignment_analysis.get_offsets(aligned_midi, old_midi)


  slope, intercept, r, p_err, stderr = alignment_analysis.get_regression_stats(aligned_midi, old_midi, offsets)
  # print "Maximum offset: {}".format(np.amax(offsets))
  if success == 1:
    max_offset_pass = np.append(max_offset_pass, np.amax(offsets))
    offset_deviation_pass = np.append(offset_deviation_pass, np.std(offsets))
    r_offset_pass = np.append(r_offset_pass,r)
    std_err_pass = np.append(std_err_pass, stderr)
  else:
    max_offset_fail = np.append(max_offset_fail, np.amax(offsets))
    offset_deviation_fail = np.append(offset_deviation_fail, np.std(offsets))
    r_offset_fail = np.append(r_offset_fail, r)
    std_err_fail = np.append(std_err_fail, stderr)

  # print "Slope of regression: {}".format(slope)
  # print "R-value of regression: {}".format(r)


  # data collection on the cost path
  cost_path = alignment_analysis.get_cost_path(p1,q1,sim_mat_1)
  if success == 1:
    cost_std_pass = np.append(cost_std_pass, np.std(cost_path))
    cost_var_pass = np.append(cost_var_pass, np.var(cost_path))
  else:
    cost_std_fail = np.append(cost_std_fail, np.std(cost_path))
    cost_var_fail = np.append(cost_var_fail, np.var(cost_path))

  cost_path_filtered = np.copy(cost_path)
  size = cost_path_filtered.shape[0]/2
  if size % 2 == 0:
    size +=1

  cost_path_filtered = scipy.signal.medfilt(cost_path, kernel_size = size)
  start = int(cost_path_filtered.shape[0]*.05)
  end = int(cost_path_filtered.shape[0]*.95)
  cost_path_filtered = cost_path_filtered[start:end]

  cost_var = cost_path[start:end] - cost_path_filtered
  # save the remainder as a plot
  var_output_folder = '../Post_Filter_Check'
  if not os.path.exists(var_output_folder):
    os.mkdir(var_output_folder)

  x = np.arange(start= 0,stop = cost_path_filtered.shape[0])
  xf = librosa.frames_to_time(x)
  # print p
  # parab = p[0]*x**2+ p[1]*x+p[2]
  plt.subplot2grid((2,1),(0,0))
  plt.plot(librosa.frames_to_time(np.arange(start= 0,stop = cost_path.shape[0])),cost_path)
  if success == 1:
    plt.title("ORIGINAL-SUCCESS")
  else:
    plt.title("ORIGINAL-FAIL")
  plt.subplot2grid((2,1),(1,0))
  plt.plot(xf,cost_var, '--', label = 'Remaining After filter')


  if success == 1:
    plt.title(str(size)+'-Excitation-'+title_path)
    plt.savefig(os.path.join(var_output_folder,row[0]+'-SUCCESS.pdf'))
    plt.close()
  else:
    plt.title(str(size)+'-Excitation-'+title_path)
    plt.savefig(os.path.join(var_output_folder,row[0]+'-FAIL.pdf'))
    plt.close()

  half = size/2
  if half % 2 == 0:
    half += 1
  cost_half_filtered = scipy.signal.medfilt(cost_path, kernel_size = half)
  if success == 1:
    filt_cost_std_pass = np.append(filt_cost_std_pass, np.std(cost_path_filtered))
    filt_cost_var_pass = np.append(filt_cost_var_pass, np.var(cost_path_filtered))
  else:
    filt_cost_std_fail = np.append(filt_cost_std_fail, np.std(cost_path_filtered))
    filt_cost_var_fail = np.append(filt_cost_var_fail, np.var(cost_path_filtered))


  p, parab,residuals = alignment_analysis.parabola_fit(cost_path_filtered)

  # build residuals of applying parab to original cost path
  res_original = np.subtract(cost_path[start:end], parab)
  res_half = np.subtract(cost_half_filtered[start:end], parab)

  x = np.arange(start= 0,stop = cost_path_filtered.shape[0])
  # print p
  # parab = p[0]*x**2+ p[1]*x+p[2]
  plt.subplot2grid((2,1),(0,0))
  plt.plot(np.arange(start= 0,stop = cost_path.shape[0]),cost_path,x,parab, '--')
  if success == 1:
    plt.title("ORIGINAL-SUCCESS")
  else:
    plt.title("ORIGINAL-FAIL")
  plt.subplot2grid((2,1),(1,0))
  plt.plot(xf,cost_path_filtered, xf, parab, '--', label = 'Parabolic Regression')

  if not os.path.exists('../Filter_Check-Half'):
    os.mkdir('../Filter_Check-Half')
  if success == 1:
    plt.title(str(size)+'-FILTERED-'+title_path)
    plt.savefig(os.path.join('../Filter_Check-Half',row[0]+'-SUCCESS.pdf'))
    plt.close()
  else:
    plt.title(str(size)+'-FILTERED-'+title_path)
    plt.savefig(os.path.join('../Filter_Check-Half',row[0]+'-FAIL.pdf'))
    plt.close()
  horiz, vert = alignment_analysis.get_non_diagonal_steps(p1,q1)
  nondag = horiz+vert



  # save info on residuals
  if success == 1:
    para_res_pass = np.append(para_res_pass, np.var(residuals))
    nondiag_pass = np.append(nondiag_pass, nondag)
    orig_res_pass = np.append(orig_res_pass, np.var(res_original))
    half_res_pass = np.append(half_res_pass, np.var(res_half))
  else:
    para_res_fail = np.append(para_res_fail, np.var(residuals))
    nondiag_fail = np.append(nondiag_fail, nondag)
    orig_res_fail = np.append(orig_res_fail, np.var(res_original))
    half_res_fail = np.append(half_res_fail, np.var(res_half))


  # data collection on first 5% of offsets
  first5 = offsets[0:int(.05*offsets.shape[0])]

  if success == 1:
    first_offsets_pass = np.append(first_offsets_pass, float(np.amax(first5))/old_midi.get_end_time())
  else:
    first_offsets_fail = np.append(first_offsets_fail, float(np.amax(first5))/old_midi.get_end_time())


  # generate spectrogram of cost path and save it to see if there is anything worthwhile there
  # specgram_folder = '../data/cost_spectrograms'
  # if not os.path.exists(specgram_folder):
  #   os.mkdir(specgram_folder)
  # plt.subplot2grid((2,1),(0,0))
  # plt.plot(np.arange(start= 0,stop = cost_path.shape[0]),cost_path)
  # plt.title(title_path+'-ORIGINAL')
  # plt.subplot2grid((2,1), (1,0))
  # # times = librosa.frames_to_time(np.arange(start= 0,stop = cost_path.shape[0]))
  # # Fs = 1/(float(times[1])-times[0])
  # plt.specgram(x = cost_path)
  # if success == 1:
  #   plt.title('Spectrogram of Cost Path-SUCCESS')
  #   plt.savefig(os.path.join(specgram_folder,row[0]+'-SUCCESS.pdf'))
  # else:
  #   plt.title('Spectrogram of Cost Path-FAIL')
  #   plt.savefig(os.path.join(specgram_folder,row[0]+'-FAIL.pdf'))
  # plt.close()



  # compare synthesized comparison mp3 (left channel is MIDI, right is MP3)
  mp3_source = mat_out.replace('.mat.mat','.mp3.mp3')
  comp_audio, fs = librosa.load(mp3_source, mono = False)
  # print "shape of comp_audio {}".format(comp_audio.shape)
  midi_audio = comp_audio[0,:]
  mp3_audio = comp_audio[1,:]
  cosine = np.dot(midi_audio, mp3_audio)/(np.linalg.norm(midi_audio)*np.linalg.norm(mp3_audio))
  print "COSINE DISTANCE {}".format(cosine)
  if success == 1:
    cosine_pass = np.append(cosine_pass, cosine)
  else:
    cosine_fail = np.append(cosine_fail, cosine)
  print midi_audio.shape


with PdfPages('Results_Comparison-Half_Length_Window.pdf') as pdf:

  # ax = plt.subplot2grid((3,2),(0,0))
  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(cqt_scores_pass.shape[0]), cqt_scores_pass, '.', color = 'g', label = 'Passing')
  plt.plot(.8*np.ones(cqt_scores_fail.shape[0]), cqt_scores_fail, '.', color = 'r', label = 'Failing' )
  plt.title('Passing vs failing scores (Weighted)', fontsize = 'small')
  plt.xlim([0,1.1])
  plt.legend(loc = 'upper right')

  pdf.savefig()
  plt.close()
  # ax = plt.subplot2grid((3,2),(0,1))
  # plt.figure(figsize = (4,4))

  plt.plot(.4*np.ones(cqt_scores_passUW.shape[0]), cqt_scores_passUW, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(cqt_scores_failUW.shape[0]), cqt_scores_failUW, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing scores (Unweighted)', fontsize = 'small')
  plt.xlim([0,1.1])
  plt.legend(loc = 'upper right')

  pdf.savefig()
  plt.close()

  # ax = plt.subplot2grid((3,2),(1,0))
  # plt.figure(figsize = (4,4))

  plt.plot(.4*np.ones(norm_mat_pass.shape[0]), norm_mat_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(norm_mat_fail.shape[0]), norm_mat_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing similarity matrix magnitudes', fontsize = 'small')
  plt.xlim([0,1.1])
  plt.legend(loc = 'upper right')
  pdf.savefig()
  plt.close()

  # ax = plt.subplot2grid((3,2),(1,1))
  # plt.figure(figsize = (4,4))

  plt.plot(.4*np.ones(max_offset_pass.shape[0]), max_offset_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(max_offset_fail.shape[0]), max_offset_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Maximum Offsets', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # ax = plt.subplot2grid((3,2),(2,0))
  # plt.figure(figsize = (4,4))

  plt.plot(.4*np.ones(offset_deviation_pass.shape[0]), offset_deviation_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(offset_deviation_fail.shape[0]), offset_deviation_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Standard Dev of Offsets', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # ax = plt.subplot2grid((3,2),(2,1))plt.figure(figsize = (4,4))
  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(r_offset_pass.shape[0]), r_offset_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(r_offset_fail.shape[0]), r_offset_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing LinReg Offsets', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))

  plt.plot(.4*np.ones(std_err_pass.shape[0]), std_err_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(std_err_fail.shape[0]), std_err_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Standard Error of LinReg', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(cost_std_pass.shape[0]), cost_std_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(cost_std_fail.shape[0]), cost_std_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Standard Deviation of Cost Path', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(cost_var_pass.shape[0]), cost_var_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(cost_var_fail.shape[0]), cost_var_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Variance of Cost Path', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()


  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(filt_cost_std_pass.shape[0]), filt_cost_std_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(filt_cost_std_fail.shape[0]), filt_cost_std_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Standard Error of Filtered Cost Path', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(filt_cost_var_pass.shape[0]), filt_cost_var_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(filt_cost_var_fail.shape[0]), filt_cost_var_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Variance of Filtered Cost Path', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(para_res_pass.shape[0]), para_res_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(para_res_fail.shape[0]), para_res_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Mean of Parabolic Residuals', fontsize = 'small')
  plt.xlim([0,1.1])
  plt.legend(loc = 'upper right')

  pdf.savefig()
  plt.close()

  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(nondiag_pass.shape[0]), nondiag_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(nondiag_fail.shape[0]), nondiag_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Amount of NonDiagonal Steps', fontsize = 'small')
  plt.legend(loc = 'upper right')

  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()


  # plt.figure(figsize = (4,4))
  plt.plot(.4*np.ones(orig_res_pass.shape[0]), orig_res_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(orig_res_fail.shape[0]), orig_res_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Variance of Parab Residuals Applied to Original', fontsize = 'small')
  plt.legend(loc = 'upper right')
  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()


  plt.plot(.4*np.ones(half_res_pass.shape[0]), half_res_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(half_res_fail.shape[0]), half_res_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Variance of Parab Residuals Applied to Half Filtered', fontsize = 'small')
  plt.legend(loc = 'upper right')
  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  plt.plot(.4*np.ones(first_offsets_pass.shape[0]), first_offsets_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(first_offsets_fail.shape[0]), first_offsets_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Maximums of first 5 percent of offsets', fontsize = 'small')
  plt.legend(loc = 'upper right')
  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

  plt.plot(cqt_scores_pass, norm_mat_pass, '.', color =  'g', label = 'Passing')
  plt.plot(cqt_scores_fail, norm_mat_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Scores Against Similarity Matrix', fontsize = 'small')
  plt.legend(loc = 'best')
  pdf.savefig()
  plt.close()

  plt.plot(.4*np.ones(cosine_pass.shape[0]), cosine_pass, '.', color =  'g', label = 'Passing')
  plt.plot(.8*np.ones(cosine_fail.shape[0]), cosine_fail, '.', color =  'r', label = 'Failing')
  plt.title('Passing vs failing Cosine Distances of synthesized Comparisons', fontsize = 'small')
  plt.legend(loc = 'upper right')
  plt.xlim([0,1.1])
  pdf.savefig()
  plt.close()

condition = filt_cost_var_pass > .00013
conditionF = filt_cost_var_fail > .00013
exP = np.extract(condition, filt_cost_var_pass)
exF = np.extract(conditionF, filt_cost_var_fail)

conditionS = cqt_scores_pass < .04099
conditionSF = cqt_scores_fail < .04099
exSP = np.extract(conditionS, cqt_scores_pass)
exSF = np.extract(conditionSF, cqt_scores_fail)


condition_origP = orig_res_pass < .000157
condition_origF = orig_res_fail < .000157
exRP = np.extract(condition_origP, orig_res_pass)
exRF = np.extract(condition_origF, orig_res_fail)

arg1 =  np.argwhere(condition_origP)
arg2 =  np.argwhere(conditionS)
inCommon = np.intersect1d(arg1,arg2)


print "Songs in common of weighted score acceptance and parabolic residual acceptance: {}".format(inCommon.shape[0])
print "Amount of remaining acceptances by weighted score: {}".format(arg2.shape[0]-inCommon.shape[0])
print "Amount of remaining acceptances by parabolic residuals: {}".format(arg1.shape[0]-inCommon.shape[0])

print "Percentage of passing variances greater than .00013: {}".format((float(exP.shape[0])/filt_cost_var_pass.shape[0])*100)
print "Percentage of failing variances greater than .00013: {}".format((float(exF.shape[0])/filt_cost_var_fail.shape[0])*100)
print np.percentile(filt_cost_var_pass, 90)
print "Percentage of passing scores less than .04099: {}".format((float(exSP.shape[0])/cqt_scores_pass.shape[0])*100)
print "Percentage of failing scores less than .04099: {}".format((float(exSF.shape[0])/cqt_scores_fail.shape[0])*100)
print "Minimum value of variance of Parab Resid applied to Original (Failing): {}".format(np.amin(orig_res_fail))
print "Percentage of Variances of Residuals applied to original cost paths < .000157 (passing) : {}".format((float(exRP.shape[0])/orig_res_pass.shape[0])*100)
print "Percentage of Variances of Residuals applied to original cost paths < .000157 (failing) : {}".format((float(exRF.shape[0])/orig_res_fail.shape[0])*100)
