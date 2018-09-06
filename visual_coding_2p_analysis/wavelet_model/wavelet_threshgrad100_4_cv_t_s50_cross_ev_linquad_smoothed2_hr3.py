import h5py
from unnatural_movie_analysis import unnatural_movie_analysis
from natural_movie_analysis import natural_movie_analysis
import os
import numpy as np
import sys
from gabor_wavelet_preprocessing_linquad import gabor_wavelet_preprocessing_linquad
from threshgrad import threshold_grad_desc_cv, lsq, dotdelay, soft_rect, soft_rect_poisson, weighted_correlation
import itertools
from scipy.ndimage.filters import gaussian_filter1d
import sklearn.mixture as mixture
    
def binarize_running(running_speed):
    run_thresh = 1.
    good_ind = np.isfinite(running_speed)
    # X = np.array(running_speed[good_ind]).reshape(-1, 1)
    X = np.array(running_speed[good_ind]).reshape(-1, 1)

    run_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.2, covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)
    run_dpgmm.fit(X)
    Y = run_dpgmm.predict(X)
    means = run_dpgmm.means_
    vars = run_dpgmm.covariances_

    labels = -1*np.ones(running_speed.shape, dtype=np.int)
    labels[good_ind] = Y

    run_states = np.unique(Y)
    means = [means[i][0] for i in run_states]
    vars = [vars[i][0] for i in run_states]

    stationary_label = [run_states[i] for i in range(len(run_states)) if means[i] < run_thresh]
    l2 = np.ones(labels.shape)
    for ll in stationary_label:
        l2[labels==ll] = 0
    return l2



exp = np.int(sys.argv[1])
thresh = np.float32(sys.argv[2])
std = np.float32(sys.argv[3])

# fileout_u = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_u' + '_lq_hr2_smoothed2_' + str(std) + '.npz'
fileout_n = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_n' + '_lq_hr3_smoothed2_' + str(std) + '.npz'
# fileout_c = '/allen/aibs/mat/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_c.npz'
# fileout_f = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_f' + '_lq_hr3_smoothed2_' + str(std) + '.npz'

if os.path.isfile(fileout_n):
    exit()

delays = np.arange(10)



temporal_window = 11
sf_gauss_ratio=.35
spatial_frequencies=[0,1.9,3.8,7.6,15.2]
temporal_frequencies = [0,2,4,8]
gabor_spacing=3
maxiter=100
folds=6
segment_size=50
output_nonlinearity = None



nma = natural_movie_analysis(exp, .15, include_scenes=True)


(stim, resp, frame_numbers, weights, shifts, dffs, pupil_size, running, mov_index, motion_correction, centers) = nma.vectorize_data(delays, correct_eye_pos=True, crop=True)


resp = gaussian_filter1d(resp, std)
running = gaussian_filter1d(running, std)

resp_train_n = np.float32(resp.T * 10)


running_train_n = binarize_running(running)

gwp_train_n = gabor_wavelet_preprocessing_linquad(stim, temporal_window=temporal_window, sf_gauss_ratio=sf_gauss_ratio, max_temp_env=.2, zscore_by_type=True, temporal_frequencies=temporal_frequencies, spatial_frequencies=spatial_frequencies, gabor_spacing=gabor_spacing, output_nonlinearity=output_nonlinearity)
# stimfile_n = '/allen/aibs/mat/michaelo/' + str(exp) + '_n.npz'
# if os.path.isfile(stimfile_n):
#     stim_train_wav_n = np.load(stimfile_n)['stim_train_wav_n']
# else:

stimfile_n = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_n_lq_hr3.npz'
try:
    stim_train_wav_n = np.load(stimfile_n)['stim_train_wav_n']
except:
    stim_train_wav_n = gwp_train_n.compute_filter_responses()
    stim_train_wav_n = np.float32(stim_train_wav_n)
    np.savez(stimfile_n, stim_train_wav_n=stim_train_wav_n)


running_train_n = np.float32(running_train_n[:,None])
if not np.all(running_train_n==1):
    running_train_n -= np.mean(running_train_n)
    running_train_n /= np.std(running_train_n)
else:
    running_train_n[:] = 0



stim_train_wav_n = np.concatenate([stim_train_wav_n, running_train_n], axis=1)
weights_train_n = weights
    # np.savez(stimfile_n, stim_train_wav_n=stim_train_wav_n)


if os.path.isfile(fileout_n):
    h_n = np.load(fileout_n)['h_n']
    b_n = np.load(fileout_n)['b_n']
else:
    (h_n, b_n, ve_n, vc_n, te_n, tc_n) = threshold_grad_desc_cv(stim_train_wav_n, resp_train_n, step=.01, maxiter=maxiter, folds=folds, threshold=thresh, delays=delays, err_function=soft_rect_poisson, weights=weights_train_n, segment_size=segment_size)
    np.savez(fileout_n, h_n=h_n, b_n=b_n, ve_n=ve_n, vc_n=vc_n, te_n=te_n, tc_n=tc_n)