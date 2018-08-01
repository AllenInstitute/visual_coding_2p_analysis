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

    # stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

    # Nstationary_labels = len(stationary_label)
    # stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
    #                     stationary_vars[i] is min(stationary_vars)]



exp = np.int(sys.argv[1])
thresh = np.float32(sys.argv[2])
std = np.float32(sys.argv[3])

fileout_u = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_u' + '_lq_hr2_smoothed2_' + str(std) + '.npz'
fileout_n = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_n' + '_lq_hr2_smoothed2_' + str(std) + '.npz'
# fileout_c = '/allen/aibs/mat/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_c.npz'
fileout_f = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_threshwavelet100_4_cv_t' + str(thresh) + '_s50_f' + '_lq_hr2_smoothed2_' + str(std) + '.npz'

if os.path.isfile(fileout_f):
    exit()

unma = unnatural_movie_analysis(exp, .15)

delays = np.arange(10)

(stim, resp, frame_numbers, weights, dffs,running, idx, loc) = unma.vectorize_data_simple(delays)
resp = gaussian_filter1d(resp, std)
running = gaussian_filter1d(running, std)

resp = np.float32(resp.T * 10)

stim_resp_dict = {}

temporal_window = 11
sf_gauss_ratio=.35
spatial_frequencies=[0,1.9,3.8,7.6,15.2]
temporal_frequencies = [0,2,4,8]
gabor_spacing=3
maxiter=100
folds=6
segment_size=50
output_nonlinearity = None

lag = temporal_window/2 + len(delays) + 1
idx_ii = np.where(idx==0)[0]
ufn = np.unique(frame_numbers[idx_ii])
respout = np.zeros(((61+lag*2)*40, resp.shape[1]), dtype='float32')
runout = np.zeros(((61+lag*2)*40), dtype='float32')
weightsout = np.zeros((61+lag*2)*40)
stimout = np.zeros(((61+lag*2)*40, stim.shape[1], stim.shape[2]), dtype='float32')
cnt = 0
idxst=[]
for f in range(40):
    fn = 61*f + 1
    fidx = np.where(frame_numbers == ufn[fn])[0]
    for dd in np.arange(1,lag+1)[::-1]:
        fidx2 = fidx - dd
        fidx2 = fidx2[fidx2>=0]
        weightsout[cnt] = np.sum(weights[fidx2])
        respout[cnt,:] = np.sum(resp[fidx2,:], axis=0) / weightsout[cnt]
        runout[cnt] = np.sum(running[fidx2], axis=0) / weightsout[cnt]
        stimout[cnt] = stim[0]
        cnt +=1
        idxst.append(fidx2)
    for fn2 in range(1,62):
        fn = 61*f + fn2
        fidx = np.where(frame_numbers == ufn[fn])[0]
        weightsout[cnt] = np.sum(weights[fidx])
        respout[cnt,:] = np.sum(resp[fidx,:], axis=0) / weightsout[cnt]
        runout[cnt] = np.sum(running[fidx], axis=0) / weightsout[cnt]
        stimout[cnt] = stim[fn]
        cnt +=1
        idxst.append(fidx)
    fn = 61*f + 61
    fidx = np.where(frame_numbers == ufn[fn])[0]
    for dd in np.arange(1,lag+1):
        fidx2 = fidx + dd
        fidx2 = fidx2[fidx2>=0]
        fidx2 = fidx2[fidx2<len(weights)]
        weightsout[cnt] = np.sum(weights[fidx2])
        respout[cnt,:] = np.sum(resp[fidx2,:], axis=0) / weightsout[cnt]
        runout[cnt] = np.sum(running[fidx2], axis=0) / weightsout[cnt]
        stimout[cnt] = stim[0]
        cnt +=1
        idxst.append(fidx2)

stim_resp_dict[0] = [stimout, respout, weightsout, runout]


image_static = np.where(idx==1)[0]
stim_resp_dict[1] = [stim[frame_numbers[image_static]], resp[image_static], weights[image_static], running[image_static]]

image_lsn = np.where(idx==2)[0]
stim_resp_dict[2] = [stim[frame_numbers[image_lsn]], resp[image_lsn], weights[image_lsn], running[image_lsn]]

image_lsn2 = np.where(idx==3)[0]
stim_resp_dict[3] = [stim[frame_numbers[image_lsn2]], resp[image_lsn2], weights[image_lsn2], running[image_lsn2]]

image_lsn3 = np.where(idx==4)[0]
stim_resp_dict[4] = [stim[frame_numbers[image_lsn3]], resp[image_lsn3], weights[image_lsn3], running[image_lsn3]]


stim_train_u = np.concatenate([stim_resp_dict[0][0], stim_resp_dict[1][0], stim_resp_dict[2][0], stim_resp_dict[3][0], stim_resp_dict[4][0]], axis=0)

resp_train_u = np.concatenate([stim_resp_dict[0][1], stim_resp_dict[1][1], stim_resp_dict[2][1], stim_resp_dict[3][1], stim_resp_dict[4][1]], axis=0)

weights_train_u = np.concatenate([stim_resp_dict[0][2], stim_resp_dict[1][2], stim_resp_dict[2][2], stim_resp_dict[3][2], stim_resp_dict[4][2]], axis=0) / 10

running_train_u = np.concatenate([stim_resp_dict[0][3], stim_resp_dict[1][3], stim_resp_dict[2][3], stim_resp_dict[3][3], stim_resp_dict[4][3]], axis=0)

running_train_u = binarize_running(running_train_u)

resp_train_u[0] = 0
weights_train_u[0] = 0

gwp_train_u = gabor_wavelet_preprocessing_linquad(stim_train_u, temporal_window=temporal_window, sf_gauss_ratio=sf_gauss_ratio, max_temp_env=.2, zscore_by_type=True, temporal_frequencies=temporal_frequencies, spatial_frequencies=spatial_frequencies, gabor_spacing=gabor_spacing, output_nonlinearity=output_nonlinearity)

stimfile_u = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_u_lq_hr2.npz'
try:
    stim_train_wav_u = np.load(stimfile_u)['stim_train_wav_u']
except:
    stim_train_wav_u = gwp_train_u.compute_filter_responses()
    stim_train_wav_u = np.float32(stim_train_wav_u)
    np.savez(stimfile_u, stim_train_wav_u=stim_train_wav_u)

running_train_u = np.float32(running_train_u[:,None])
if not np.all(running_train_u==1):
    running_train_u -= np.mean(running_train_u)
    running_train_u /= np.std(running_train_u)
else:
    running_train_u[:] = 0

stim_train_wav_u = np.concatenate([stim_train_wav_u, running_train_u], axis=1)
    

if os.path.isfile(fileout_u):
    h_u = np.load(fileout_u)['h_u']
    b_u = np.load(fileout_u)['b_u']
else:
    (h_u, b_u, ve_u, vc_u, te_u, tc_u) = threshold_grad_desc_cv(stim_train_wav_u, resp_train_u, step=.01, maxiter=maxiter, folds=folds, threshold=thresh, delays=delays, err_function=soft_rect_poisson, weights=weights_train_u, segment_size=segment_size)
    np.savez(fileout_u, h_u=h_u, b_u=b_u, ve_u=ve_u, vc_u=vc_u, te_u=te_u, tc_u=tc_u)


nma = natural_movie_analysis(exp, .15, include_scenes=True)


(stim, resp, frame_numbers, weights, dffs,running, idx, loc) = nma.vectorize_data_simple(delays)
resp = gaussian_filter1d(resp, std)
running = gaussian_filter1d(running, std)

resp = np.float32(resp.T * 10)

stim_resp_dict = {}

for ii in range(3):
    idx_ii = np.where(idx==ii)[0]
    ufn = np.unique(frame_numbers[idx_ii])
    respout = np.zeros((len(ufn), resp.shape[1]), dtype='float32')
    runout = np.zeros((len(ufn)), dtype='float32')
    weightsout = np.zeros(len(ufn))
    for fn in range(len(ufn)):
        fidx = np.where(frame_numbers == ufn[fn])[0]
        weightsout[fn] = np.sum(weights[fidx])
        respout[fn,:] = np.sum(resp[fidx,:], axis=0) / weightsout[fn]
        runout[fn] = np.sum(running[fidx], axis=0) / weightsout[fn]
    weightsout[delays] = 0
    stim_resp_dict[ii] = [stim[ufn], respout, weightsout, runout]


image_mov = np.where(idx==3)[0]

stim_resp_dict[3] = [stim[frame_numbers[image_mov]], resp[image_mov], weights[image_mov], running[image_mov]]

stim_train_n = np.concatenate([stim_resp_dict[0][0], stim_resp_dict[1][0], stim_resp_dict[2][0], stim_resp_dict[3][0]], axis=0)

resp_train_n = np.concatenate([stim_resp_dict[0][1], stim_resp_dict[1][1], stim_resp_dict[2][1], stim_resp_dict[3][1]], axis=0)

weights_train_n = np.concatenate([stim_resp_dict[0][2], stim_resp_dict[1][2], stim_resp_dict[2][2], stim_resp_dict[3][2]], axis=0) / 10

running_train_n = np.concatenate([stim_resp_dict[0][3], stim_resp_dict[1][3], stim_resp_dict[2][3], stim_resp_dict[3][3]], axis=0)

running_train_n = binarize_running(running_train_n)

gwp_train_n = gabor_wavelet_preprocessing_linquad(stim_train_n, temporal_window=temporal_window, sf_gauss_ratio=sf_gauss_ratio, max_temp_env=.2, zscore_by_type=True, temporal_frequencies=temporal_frequencies, spatial_frequencies=spatial_frequencies, gabor_spacing=gabor_spacing, output_nonlinearity=output_nonlinearity)
# stimfile_n = '/allen/aibs/mat/michaelo/' + str(exp) + '_n.npz'
# if os.path.isfile(stimfile_n):
#     stim_train_wav_n = np.load(stimfile_n)['stim_train_wav_n']
# else:

stimfile_n = '/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + str(exp) + '_n_lq_hr2.npz'
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
    # np.savez(stimfile_n, stim_train_wav_n=stim_train_wav_n)


if os.path.isfile(fileout_n):
    h_n = np.load(fileout_n)['h_n']
    b_n = np.load(fileout_n)['b_n']
else:
    (h_n, b_n, ve_n, vc_n, te_n, tc_n) = threshold_grad_desc_cv(stim_train_wav_n, resp_train_n, step=.01, maxiter=maxiter, folds=folds, threshold=thresh, delays=delays, err_function=soft_rect_poisson, weights=weights_train_n, segment_size=segment_size)
    np.savez(fileout_n, h_n=h_n, b_n=b_n, ve_n=ve_n, vc_n=vc_n, te_n=te_n, tc_n=tc_n)


yhat_n_u = dotdelay(stim_train_wav_u, h_n, b_n, delays)

(grad_n_u, err_n_u) = soft_rect_poisson(resp_train_u[:,:,None], yhat_n_u, weights=weights_train_u[:,None,None])

err_n_u = np.nanmean(err_n_u, axis=0)

yhat_n_u = soft_rect(yhat_n_u)

corrs_n_u = np.zeros(err_n_u.shape)
corrs_n_u_w = np.zeros(err_n_u.shape)
for ff in range(folds):
    corrs_n_u[:,ff] = weighted_correlation(resp_train_u, yhat_n_u[:,:,ff])
    corrs_n_u_w[:,ff] = weighted_correlation(resp_train_u, yhat_n_u[:,:,ff], weights_train_u)


yhat_u_n = dotdelay(stim_train_wav_n, h_u, b_u, delays)

(grad_u_n, err_u_n) = soft_rect_poisson(resp_train_n[:,:,None], yhat_u_n, weights=weights_train_n[:,None,None])

err_u_n = np.nanmean(err_u_n, axis=0)

yhat_u_n = soft_rect(yhat_u_n)

corrs_u_n = np.zeros(err_u_n.shape)
corrs_u_n_w = np.zeros(err_u_n.shape)
for ff in range(folds):
    corrs_u_n[:,ff] = weighted_correlation(resp_train_n, yhat_u_n[:,:,ff])
    corrs_u_n_w[:,ff] = weighted_correlation(resp_train_n, yhat_u_n[:,:,ff], weights_train_n)




starts = np.array(range(0, stim_train_wav_n.shape[0], segment_size))
foldstarts = [starts[x::folds] for x in range(folds)]
foldidxs = [(f[:,None] + np.array(range(segment_size))[:, None].T).reshape(-1) for f in foldstarts]
foldidxs = [f[f < stim_train_wav_n.shape[0]] for f in foldidxs]
p = set(range(folds))
val_mask = np.zeros((stim_train_wav_n.shape[0], folds*(folds-1)), dtype='float32')
stop_mask = np.zeros((stim_train_wav_n.shape[0], folds*(folds-1)), dtype='float32')
train_mask = np.zeros((stim_train_wav_n.shape[0], folds*(folds-1)), dtype='float32')
for ii, i in enumerate(itertools.permutations(p, 2)):
    v = i[0]
    s = i[1]
    t = list(p - set(i))
    for tt in t:
        train_mask[foldidxs[tt], ii] = 1
    stop_mask[foldidxs[s], ii] = 1
    val_mask[foldidxs[v], ii] = 1

stop_mask = np.expand_dims(stop_mask, 1)
val_mask = np.expand_dims(val_mask, 1)
train_mask = np.expand_dims(train_mask, 1)

val_mask = val_mask[:,:,::(folds-1)]

yhat_n_n = dotdelay(stim_train_wav_n, h_n, b_n, delays)

(grad_n_n, err_n_n) = soft_rect_poisson(resp_train_n[:,:,None], yhat_n_n, weights=weights_train_n[:,None,None])

err_n_n = np.nanmean(err_n_n, axis=0)

yhat_n_n = soft_rect(yhat_n_n)

val_corrs_n_n_w = np.zeros(err_n_n.shape)
train_corrs_n_n_w = np.zeros(err_n_n.shape)
val_corrs_n_n = np.zeros(err_n_n.shape)
train_corrs_n_n = np.zeros(err_n_n.shape)
for ff in range(folds):
    fidx = np.where(val_mask[:,:,ff])[0]
    fidx2 = np.where(train_mask[:,:,ff])[0]
    val_corrs_n_n_w[:,ff] = weighted_correlation(resp_train_n[fidx,:], yhat_n_n[fidx,:,ff], weights_train_n[fidx])
    train_corrs_n_n_w[:,ff] = weighted_correlation(resp_train_n[fidx2,:], yhat_n_n[fidx2,:,ff], weights_train_n[fidx2])
    val_corrs_n_n[:,ff] = weighted_correlation(resp_train_n[fidx,:], yhat_n_n[fidx,:,ff])
    train_corrs_n_n[:,ff] = weighted_correlation(resp_train_n[fidx2,:], yhat_n_n[fidx2,:,ff])




starts = np.array(range(0, stim_train_wav_u.shape[0], segment_size))
foldstarts = [starts[x::folds] for x in range(folds)]
foldidxs = [(f[:,None] + np.array(range(segment_size))[:, None].T).reshape(-1) for f in foldstarts]
foldidxs = [f[f < stim_train_wav_u.shape[0]] for f in foldidxs]
p = set(range(folds))
val_mask = np.zeros((stim_train_wav_u.shape[0], folds*(folds-1)), dtype='float32')
stop_mask = np.zeros((stim_train_wav_u.shape[0], folds*(folds-1)), dtype='float32')
train_mask = np.zeros((stim_train_wav_u.shape[0], folds*(folds-1)), dtype='float32')
for ii, i in enumerate(itertools.permutations(p, 2)):
    v = i[0]
    s = i[1]
    t = list(p - set(i))
    for tt in t:
        train_mask[foldidxs[tt], ii] = 1
    stop_mask[foldidxs[s], ii] = 1
    val_mask[foldidxs[v], ii] = 1

stop_mask = np.expand_dims(stop_mask, 1)
val_mask = np.expand_dims(val_mask, 1)
train_mask = np.expand_dims(train_mask, 1)

val_mask = val_mask[:,:,::(folds-1)]

yhat_u_u = dotdelay(stim_train_wav_u, h_u, b_u, delays)

(grad_u_u, err_u_u) = soft_rect_poisson(resp_train_u[:,:,None], yhat_u_u, weights=weights_train_u[:,None,None])

err_u_u = np.nanmean(err_u_u, axis=0)

yhat_u_u = soft_rect(yhat_u_u)

val_corrs_u_u_w = np.zeros(err_u_u.shape)
train_corrs_u_u_w = np.zeros(err_u_u.shape)
val_corrs_u_u = np.zeros(err_u_u.shape)
train_corrs_u_u = np.zeros(err_u_u.shape)
for ff in range(folds):
    fidx = np.where(val_mask[:,:,ff])[0]
    fidx2 = np.where(train_mask[:,:,ff])[0]
    val_corrs_u_u_w[:,ff] = weighted_correlation(resp_train_u[fidx,:], yhat_u_u[fidx,:,ff], weights_train_u[fidx])
    train_corrs_u_u_w[:,ff] = weighted_correlation(resp_train_u[fidx2,:], yhat_u_u[fidx2,:,ff], weights_train_u[fidx2])
    val_corrs_u_u[:,ff] = weighted_correlation(resp_train_u[fidx,:], yhat_u_u[fidx,:,ff])
    train_corrs_u_u[:,ff] = weighted_correlation(resp_train_u[fidx2,:], yhat_u_u[fidx2,:,ff])



np.savez(fileout_f, err_n_u=err_n_u, err_u_n=err_u_n, corrs_n_u=corrs_n_u, corrs_u_n=corrs_u_n, corrs_n_u_w=corrs_n_u_w, corrs_u_n_w=corrs_u_n_w, val_corrs_u_u_w=val_corrs_u_u_w, val_corrs_u_u=val_corrs_u_u, train_corrs_u_u_w=train_corrs_u_u_w, train_corrs_u_u=train_corrs_u_u, val_corrs_n_n_w=val_corrs_n_n_w, val_corrs_n_n=val_corrs_n_n, train_corrs_n_n_w=train_corrs_n_n_w, train_corrs_n_n=train_corrs_n_n)
