'''
compute noise and signal correlation matrices for each experiment
'''

import os, sys, pickle, corr_functions
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from visual_coding_2p_analysis.l0_analysis import L0_analysis
from visual_coding_2p_analysis import core

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    manifest_file = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache/brain_observatory_manifest.json'
    boc = BrainObservatoryCache(manifest_file=manifest_file)

elif sys.platform == 'linux2':
    try:
        # manifest_file = '/allen/aibs/mat/gkocker/BrainObservatory/boc/manifest.json'
        manifest_file = core.get_manifest_path()
        boc = BrainObservatoryCache(manifest_file=manifest_file)

    except:
        manifest_file = 'boc/manifest.json'
        boc = BrainObservatoryCache(manifest_file=manifest_file)


import numpy as np
from pandas import HDFStore
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from cam_decode_final import label_run_stationary_dpgmm, get_tables_exp, interpolate_pupil_size
import sklearn.mixture as mixture
from scipy.stats import ttest_rel, ttest_1samp, spearmanr



def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=1), axis=1)
    return (cumsum[:, N:] - cumsum[:, :-N]) / N


def running_sum(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=1), axis=1)
    return (cumsum[:, N:] - cumsum[:, :-N])



def compute_noise_signal_corrs_dimension_split(response_array, corr='pearson', num_boots=100):

    '''
    bootstrap independent samples for each neuron to compute signal correlations, averaging over bootstrap samples

    :param response_array:
    :param corr:
    :return:
    '''

    # calculate signal correlations, averaging over splits of the trials
    Ncells, Nstim, Ntrials = response_array.shape
    signal_corr_boots = np.zeros((num_boots, Ncells, Ncells))

    for boot in range(num_boots):

        split1 = np.random.choice(Ntrials, replace=False, size=Ntrials/2)
        split2 = [i for i in range(Ntrials) if i not in split1]

        response_array_mean1 = np.mean(response_array[:, :, split1], axis=2)
        response_array_mean2 = np.mean(response_array[:, :, split2], axis=2)

        if corr == 'spearman':
            response_rank1 = corr_functions.rank_response_array(response_array_mean1).astype(np.float32)
            response_rank2 = corr_functions.rank_response_array(response_array_mean2).astype(np.float32)
        else:
            response_rank1 = response_array_mean1
            response_rank2 = response_array_mean2


        response_rank1 -= np.outer(np.mean(response_rank1, axis=1), np.ones(Nstim)) # subtract mean for correlation, outer product with ones for broadcasting
        response_rank2 -= np.outer(np.mean(response_rank2, axis=1), np.ones(Nstim))

        signal_cov = np.dot(response_rank1, response_rank2.T)
        signal_std = np.sqrt(np.diag(signal_cov))
        signal_corr_boots[boot] = signal_cov / np.outer(signal_std, signal_std)


    noise_corr = np.zeros((Ncells, Ncells, Nstim))
    noise_cov = np.zeros((Ncells, Ncells, Nstim))

    for n in range(Nstim):

        if corr == 'spearman':
            response_rank = corr_functions.rank_response_array(response_array[:, n, :]).astype(np.float32)
        else:
            response_rank = response_array[:, n, :]

        noise_cov_temp = np.cov(response_rank)
        noise_cov[:, :, n] = noise_cov_temp

        noise_std = np.sqrt(np.diag(noise_cov_temp))
        noise_corr[:, :, n] = noise_cov_temp / np.outer(noise_std, noise_std)
        noise_std = np.sqrt(np.diag(noise_cov_temp))
        noise_corr[:, :, n] = noise_cov_temp / np.outer(noise_std, noise_std)

    noise_corr = np.ma.MaskedArray(noise_corr, mask=np.isnan(noise_corr)) # if cell doesn't ever respond to a frame, have 0 denominator

    # what is the spearman correlation between the bootstrap signal and noise correlations?
    noise_corr_mean = np.mean(noise_corr, axis=2)
    inds = np.triu_indices(n=Ncells, k=1)

    noise_cov_mean = np.mean(noise_cov, axis=2)

    try:
        noise_cov_eigs = np.linalg.eigvalsh(noise_cov_mean)
        dimension_frac = (np.sum(noise_cov_eigs) ** 2) / np.sum(noise_cov_eigs ** 2) / float(Ncells)
        dimension = (np.sum(noise_cov_eigs) ** 2) / np.sum(noise_cov_eigs ** 2)
    except:
        dimension_frac = np.nan
        dimension = np.nan

    signal_corr_boots = np.ma.MaskedArray(signal_corr_boots, mask=np.isnan(signal_corr_boots))
    signal_corr = np.mean(signal_corr_boots, axis=0)[inds] # average over splits
    noise_corr_mean = noise_corr_mean[inds]

    signal_corr_rank = np.argsort(np.argsort(signal_corr))
    noise_corr_mean_rank = np.argsort(np.argsort(noise_corr_mean))
    rhoNoiseSignalCorr = np.corrcoef(signal_corr_rank, noise_corr_mean_rank)[0, 1]

    return noise_corr, signal_corr, rhoNoiseSignalCorr, dimension, dimension_frac


def compute_noise_signal_corrs_dimension(response_array, corr='pearson'):

    # calculate signal and noise correlations for that bootstrap sample
    Ncells, Nstim, Ntrials = response_array.shape

    response_array_mean = np.mean(response_array, axis=2)
    if corr == 'spearman':
        response_rank = corr_functions.rank_response_array(response_array_mean).astype(np.float32)
    else:
        response_rank = response_array_mean

    signal_cov = np.cov(response_rank)
    signal_std = np.sqrt(np.diag(signal_cov))
    signal_corr = signal_cov / np.outer(signal_std, signal_std)

    noise_corr = np.zeros((Ncells, Ncells, Nstim))
    noise_cov = np.zeros((Ncells, Ncells, Nstim))

    for n in range(Nstim):

        if corr == 'spearman':
            response_rank = corr_functions.rank_response_array(response_array[:, n, :]).astype(np.float32)
        else:
            response_rank = response_array[:, n, :]

        noise_cov_temp = np.cov(response_rank)
        noise_cov[:, :, n] = noise_cov_temp

        noise_std = np.sqrt(np.diag(noise_cov_temp))
        noise_corr[:, :, n] = noise_cov_temp / np.outer(noise_std, noise_std)
        noise_std = np.sqrt(np.diag(noise_cov_temp))
        noise_corr[:, :, n] = noise_cov_temp / np.outer(noise_std, noise_std)

    noise_corr = np.ma.MaskedArray(noise_corr, mask=np.isnan(noise_corr)) # if cell doesn't ever respond to a frame, have 0 denominator

    # what is the spearman correlation between the bootstrap signal and noise correlations?
    noise_corr_mean = np.mean(noise_corr, axis=2)
    inds = np.triu_indices(n=Ncells, k=1)

    noise_cov_mean = np.mean(noise_cov, axis=2)

    try:
        noise_cov_eigs = np.linalg.eigvalsh(noise_cov_mean)
        dimension_frac = (np.sum(noise_cov_eigs) ** 2) / np.sum(noise_cov_eigs ** 2) / float(Ncells)
        dimension = (np.sum(noise_cov_eigs) ** 2) / np.sum(noise_cov_eigs ** 2)
    except:
        dimension_frac = np.nan
        dimension = np.nan

    signal_corr = signal_corr[inds]
    noise_corr_mean = noise_corr_mean[inds]

    signal_corr_rank = np.argsort(np.argsort(signal_corr))
    noise_corr_mean_rank = np.argsort(np.argsort(noise_corr_mean))
    rhoNoiseSignalCorr = np.corrcoef(signal_corr_rank, noise_corr_mean_rank)[0, 1]

    return noise_corr, signal_corr, rhoNoiseSignalCorr, dimension, dimension_frac


def compute_signal_corrs_split(response_array, corr='pearson'):

    # calculate signal and noise correlations for that bootstrap sample
    Ncells, Nstim, Ntrials = response_array.shape

    split1 = np.random.choice(Ntrials, size=Ntrials/2, replace=False)
    split2 = np.array([i for i in range(Ntrials) if i not in split1])

    response_array_mean = np.mean(response_array, axis=2)
    response_array_mean1 = np.mean(response_array[:, :, split1], axis=2)
    response_array_mean2 = np.mean(response_array[:, :, split2], axis=2)

    if corr == 'spearman':
        response_rank1 = corr_functions.rank_response_array(response_array_mean1).astype(np.float32)
        response_rank2 = corr_functions.rank_response_array(response_array_mean2).astype(np.float32)
    else:
        response_rank1 = response_array_mean1
        response_rank2 = response_array_mean2

    signal_cov_split = np.zeros((Ncells, Ncells))
    for i in range(Ncells):
        for j in range(i, Ncells):
            signal_cov_split[i, j] = np.cov(response_rank1[i], response_rank2[j])[0, 1]

    signal_std_split = np.diag(signal_cov_split)

    signal_corr_split = signal_cov_split / np.sqrt(np.outer(signal_std_split, signal_std_split))

    inds = np.triu_indices(Ncells, k=1)
    signal_corr_split = signal_corr_split[inds]

    return signal_corr_split


def compute_spont_corrs_dimension(response_array, corr='pearson'):

    Ncells, Ntrials = response_array.shape

    if corr == 'spearman':
        response_rank = corr_functions.rank_response_array(response_array)
    else:
        response_rank = response_array

    spont_cov = np.cov(response_rank)
    spont_std = np.sqrt(np.diag(spont_cov))
    spont_corr = spont_cov / np.outer(spont_std, spont_std)

    cov_eigs = np.linalg.eigvalsh(spont_cov)
    dimension_frac = (np.sum(cov_eigs) ** 2) / np.sum(cov_eigs ** 2) / float(Ncells)
    dimension = (np.sum(cov_eigs) ** 2) / np.sum(cov_eigs ** 2)


    return spont_corr, dimension, dimension_frac


def plot_mixture(run_states, means, vars):

    import matplotlib

    x_min = min([np.amin(means)*2, -1])
    x_max = np.amax(means)*2

    x = np.linspace(x_min, x_max, 10000)

    plt.figure()
    for m in range(len(run_states)):
        plt.plot(x, matplotlib.mlab.normpdf(x, means[m], np.sqrt(vars[m])), label=run_states[m])

    plt.legend(loc=0)


def main_expt_bootstrap(expt, num_boots=50, regress_run='none', run_thresh=1., run_window=1., detect_events=True, filter_window=None, corr='pearson', split=True, calc_stim=['natural_scenes', 'static_gratings', 'drifting_gratings', 'natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'spontaneous'], save_dir='/allen/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/noise_signal_correlations_bootstrap_split'):

    expt = int(expt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)
    expt_stimuli = data_set.list_stimuli()

    dfftime, dff = data_set.get_dff_traces()
    dxcm, _ = data_set.get_running_speed()
    dxcm = dxcm[None, :]

    if detect_events:
        l0 = L0_analysis(data_set)
        dff = l0.get_events()

    Ncells = dff.shape[0]

    if Ncells > 1:

        try:
            master_stim_table = data_set.get_stimulus_table('master')
        except:
            print 'no good master stim table'
            master_stim_table = None

        try:
            pupil_t, pupil_size = data_set.get_pupil_size()
            pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
            response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind,
                                                                  width=run_window)
        except:
            print 'no pupil size information'
            response, running_speed, _ = get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, width=run_window)
            pupil_array = None


        ''' visual stimulus '''
        for stim in calc_stim:

            if stim not in expt_stimuli:
                continue

            results = dict()

            stim_table = data_set.get_stimulus_table(stim)
            response_table = corr_functions.get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
            running_speed = corr_functions.get_response_table(dxcm, stim_table, stim_type=stim, width=filter_window)

            if regress_run == 'speed':
                response_table = regress_runSpeed(response_table, running_speed)
            elif regress_run == 'state':
                response_table = regress_runState(response_table, running_speed)

            response_array = corr_functions.reshape_response_table(response_table, stim_table, stim_type=stim)
            Ncells, Nstim, Ntrials = response_array.shape

            rho_signal_noise_corr = []
            noise_corr_list = []
            signal_corr_list = []
            noise_dim_list = []
            noise_dim_frac_list = []

            for boot in range(num_boots):

                # bootstrap a sample of trials for each stimulus
                if split: # compute noise and signal correlations from disjoint sets of trials
                    bootstrap = np.random.choice(range(Ntrials), size=Ntrials, replace=False)

                    bootstrap1 = bootstrap[:Ntrials/2]
                    bootstrap2 = bootstrap[Ntrials/2:]
                    response_boot1 = response_array[:, :, bootstrap1]
                    response_boot2 = response_array[:, :, bootstrap2]

                    noise_corr_tmp, _, _, noise_dim_tmp, noise_dim_frac_tmp = compute_noise_signal_corrs_dimension(response_boot1, corr)
                    _, signal_corr_tmp, _, _, _ = compute_noise_signal_corrs_dimension(response_boot2, corr)

                    noise_corr_mean = np.mean(noise_corr_tmp, axis=2)
                    inds = np.triu_indices(n=Ncells, k=1)
                    noise_corr_mean = noise_corr_mean[inds]

                    signal_corr_rank = np.argsort(np.argsort(signal_corr_tmp))
                    noise_corr_mean_rank = np.argsort(np.argsort(noise_corr_mean))
                    rho_signal_noise_corr_tmp = np.corrcoef(signal_corr_rank, noise_corr_mean_rank)[0, 1]

                else:
                    bootstrap = np.random.choice(range(Ntrials), size=Ntrials, replace=True)
                    response_boot = response_array[:, :, bootstrap]
                    noise_corr_tmp, signal_corr_tmp, rho_signal_noise_corr_tmp, noise_dim_tmp, noise_dim_frac_tmp = compute_noise_signal_corrs_dimension(response_boot, corr)

                noise_corr_list.append(noise_corr_tmp)
                signal_corr_list.append(signal_corr_tmp)
                rho_signal_noise_corr.append(rho_signal_noise_corr_tmp)
                noise_dim_list.append(noise_dim_tmp)
                noise_dim_frac_list.append(noise_dim_frac_tmp)

            results['bootstrap_NoiseCorr_list'] = noise_corr_list
            results['bootstrap_SignalCorr_list'] = signal_corr_list
            results['bootstrap_RhoSignalNoiseCorrs_list'] = rho_signal_noise_corr
            results['bootstrap_NoiseDim_list'] = noise_dim_list
            results['bootstrap_NoiseDimFrac_list'] = noise_dim_frac_list

            savefile = os.path.join(save_dir, str(expt)+'_'+stim+'.pkl')
            error_file = open(savefile, 'wb')
            pickle.dump(results, error_file)
            error_file.close()


        if 'spontaneous' in calc_stim:
            stim = 'spontaneous'
            stim_table = data_set.get_stimulus_table(stim)

            response_table = corr_functions.get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
            if regress_run == 'speed':
                response_table = regress_runSpeed(response_table, running_speed)
            elif regress_run == 'state':
                response_table = regress_runState(response_table, running_speed)

            response_array = response_table.T
            Ncells, Ntrials = response_array.shape
            spont_corr_list = []
            spont_dim_list = []
            spont_dim_frac_list = []

            for boot in range(num_boots):

                # bootstrap a sample of trials for each stimulus
                bootstrap = np.random.choice(range(Ntrials), size=Ntrials, replace=True)
                response_boot = response_array[:, bootstrap]

                spont_corr_tmp, dim_tmp, dim_frac_tmp = compute_spont_corrs_dimension(response_array, corr)
                spont_corr_list.append(spont_corr_tmp)
                spont_dim_list.append(dim_tmp)
                spont_dim_frac_list.append(dim_frac_tmp)

            results = dict()
            results['bootstrap_SpontCorr_list'] = spont_corr_list

            savefile = os.path.join(save_dir, str(expt)+'_'+stim+'.pkl')
            error_file = open(savefile, 'wb')
            pickle.dump(results, error_file)
            error_file.close()


def main_expt(expt, regress_run='none', run_thresh=1., run_window=1, filter_window=None, corr='pearson', detect_events=True, calc_stim=['natural_scenes', 'static_gratings', 'drifting_gratings', 'natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'spontaneous'], save_dir_base='/allen/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/noise_signal_correlations_split'):

    expt = int(expt)

    if (filter_window is not None) and (save_dir_base[-6:] == 'window'):
        save_dir_base += str(filter_window)

    if regress_run == 'speed':
        save_dir = os.path.join(save_dir_base, 'regressRunSpeed')
    elif regress_run == 'state':
        save_dir = os.path.join(save_dir_base, 'regressRunState')
    else:
        save_dir = os.path.join(save_dir_base, 'regressNone')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)
    expt_stimuli = data_set.list_stimuli()

    dfftime, dff = data_set.get_dff_traces()
    dxcm, _ = data_set.get_running_speed()
    dxcm = dxcm[None, :]

    if detect_events:
        l0 = L0_analysis(data_set)
        dff = l0.get_events()

    Ncells = dff.shape[0]

    if Ncells > 1:

        try:
            master_stim_table = data_set.get_stimulus_table('master')
        except:
            print 'no good master stim table'
            master_stim_table = None

        try:
            pupil_t, pupil_size = data_set.get_pupil_size()
            pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
            response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind,
                                                                  width=run_window)
        except:
            print 'no pupil size information'
            response, running_speed, _ = get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, width=run_window)
            pupil_array = None

        #    fit mixture model to running speeds to classify running / stationary
        good_ind = np.isfinite(running_speed)
        # X = np.array(running_speed[good_ind]).reshape(-1, 1)
        X = np.array(running_speed[good_ind]).reshape(-1, 1)

        run_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.1, covariance_type='diag',
                                                    max_iter=2000, tol=1e-3, n_init=1)
        # run_dpgmm = mixture.BayesianGaussianMixture(n_components=10)

        run_dpgmm.fit(X)
        Y = run_dpgmm.predict(X)
        means = run_dpgmm.means_
        vars = run_dpgmm.covariances_

        labels = -1 * np.ones(running_speed.shape, dtype=np.int)
        labels[good_ind] = Y

        run_states = np.unique(Y)
        means = [means[i][0] for i in run_states]
        vars = [vars[i][0] for i in run_states]

        stationary_label = [run_states[i] for i in range(len(run_states)) if means[i] < run_thresh]
        stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

        Nstationary_labels = len(stationary_label)
        stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
                            stationary_vars[i] is min(stationary_vars)]

        if len(stationary_label) == 0:
            stationary_label = -2

        ''' visual stimulus '''
        for stim in calc_stim:

            if stim not in expt_stimuli:
                continue

            savefile = os.path.join(save_dir, str(expt)+'_'+stim+'.pkl')
            # if os.path.exists(savefile):
            #     print('removing old file: '+savefile)
            #     os.remove(savefile)

            results = dict()

            stim_table = data_set.get_stimulus_table(stim)

            response_table = corr_functions.get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
            running_speed = corr_functions.get_response_table(dxcm, stim_table, stim_type=stim, width=filter_window)

            if regress_run == 'speed':
                response_table = regress_runSpeed(response_table, running_speed)
            elif regress_run == 'state':
                response_table = regress_runState(response_table, running_speed)

            labels = run_dpgmm.predict(running_speed)
            ind_stationary = (labels == stationary_label)  # scenes x reps

            response_table_run = response_table[~ind_stationary]
            response_table_stat = response_table[ind_stationary]

            if stim == 'spontaneous':
                response_array = response_table.T
                results = dict()
                results['SpontCorr'], results['Dimension'], results['DimensionFrac'] = compute_spont_corrs_dimension(response_array, corr)

            else:
                response_array = corr_functions.reshape_response_table(response_table, stim_table, stim_type=stim)
                if response_array is not None and np.prod(response_array.shape) > 0:
                    results['NoiseCorr'], results['SignalCorr'], results['RhoSignalNoiseCorrs'], results['NoiseDimension'], results['NoiseDimensionFrac'] = compute_noise_signal_corrs_dimension_split(response_array, corr)


                if np.sum(~ind_stationary) > 9:
                    response_array = corr_functions.reshape_response_table(response_table_run, stim_table[~ind_stationary], stim_type=stim, shuffle=True)
                    if response_array is not None and np.prod(response_array.shape) > 0:
                        results['NoiseCorrRun'], results['SignalCorrRun'], results['RhoSignalNoiseCorrsRun'], results['NoiseDimensionRun'], results['NoiseDimensionFracRun'] = compute_noise_signal_corrs_dimension_split(response_array, corr)

                if np.sum(ind_stationary) > 9:
                    response_array = corr_functions.reshape_response_table(response_table_stat, stim_table[ind_stationary], stim_type=stim, shuffle=True)
                    if response_array is not None and np.prod(response_array.shape) > 0:
                        results['NoiseCorrStat'], results['SignalCorrStat'], results['RhoSignalNoiseCorrsStat'], results['NoiseDimensionStat'], results['NoiseDimensionFracStat'] = compute_noise_signal_corrs_dimension_split(response_array, corr)

            error_file = open(savefile, 'wb')
            pickle.dump(results, error_file)
            error_file.close()


def regress_runSpeed(response_table, run_table):

    Nstim, N = response_table.shape
    residuals = np.zeros((Nstim, N))
    run_table -= run_table.mean()

    for n in range(N):
        response = response_table[:, n]
        response -= response.mean()
        residuals[:, n] = response - lstsq(run_table, response)[0]

    return residuals


def regress_runState(response_table, running_speed, run_thresh=1.):


    labels, means, vars = corr_functions.label_run_stationary_dpgmm(running_speed)
    run_states = np.unique(labels)
    means = [means[i][0] for i in run_states]
    vars = [vars[i][0] for i in run_states]

    stationary_label = [run_states[i] for i in range(len(run_states)) if means[i] < run_thresh]
    stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

    Nstationary_labels = len(stationary_label)
    stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
                        stationary_vars[i] is min(stationary_vars)]

    run_table = (labels == stationary_label)  # stim presentations

    Nstim, N = response_table.shape
    residuals = np.zeros((Nstim, N))

    for n in range(N):
        response = response_table[:, n]
        response -= response.mean()
        slope = lstsq(run_table, response)[0]
        residuals[:, n] = response - slope * run_table[:, 0]

    return residuals


def make_results_loc_dict(stim_type='ns', results_path='/allen/programs/braintv/workgroups/cortexmodels/gocker/cam_decode/noise_signal_corr_results', plot_key='bootstrap_RhoSignalNoiseCorrs_list', compute_missing=False):

    if stim_type == 'ns' or stim_type == 'natural_scenes':
        long_stim = 'natural_scenes'
    elif stim_type == 'sg' or stim_type == 'static_gratings':
        long_stim = 'static_gratings'
    elif stim_type == 'spontaneous' or stim_type=='spont':
        long_stim = 'spontaneous'
    elif stim_type == 'drifting_gratings' or stim_type == 'dg':
        long_stim = 'drifting_gratings'
    elif stim_type == 'natural_movie_three' or stim_type == 'nm3':
        long_stim = 'natural_movie_three'
    elif stim_type == 'natural_movie_two' or stim_type == 'nm2':
        long_stim = 'natural_movie_two'
    elif stim_type == 'natural_movie_one' or stim_type == 'nm1':
        long_stim = 'natural_movie_one'
    else:
        raise Exception('undefined stim type')

    file_suffix = long_stim + '.pkl'

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    results_dict = dict()
    shuffle_dict = dict()

    for area in areas:
        for depth in depths:
            for cre_line in cre_lines:

                exps = boc.get_ophys_experiments(targeted_structures=[area], imaging_depths=[depth], cre_lines=[cre_line], stimuli=[long_stim])

                if len(exps) == 0:
                    print 'no experiments for '+area+str(depth)+cre_line
                    continue
                else:
                    exp_ids = [exp['id'] for exp in exps]

                    results_dict[area+str(depth)+cre_line] = []

                    for x, exp in enumerate(exp_ids):

                        try:
                            results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_'+file_suffix)))
                            results_dict[area+str(depth)+cre_line].append(results_file[plot_key])

                        except:
                            print 'no results for exp '+str(exp)
                            if compute_missing:
                                print 'computing results'
                                main_expt(expt=exp)
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_'+file_suffix)))
                                results_dict[area+str(depth)+cre_line].append(results_file[plot_key])

                            # continue

    return results_dict


def plot_r_noise_signal_corrs(plot_stim='natural_scenes', storage_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone', bootstrap=False):


    if bootstrap:
        results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='bootstrap_RhoSignalNoiseCorrs_list')

    else:
        results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir,
                                                 plot_key='RhoSignalNoiseCorrs')


    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    cre_colors = core.get_cre_colors()
    fail_tag = False

    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    plot_depths = [100, 200, 300, 500]

    fig, ax = plt.subplots(ncols=len(areas), nrows=len(plot_depths)-1, figsize=(10, 6), sharex=True, sharey=True)

    for i, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):
            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]
            num_this_depths = len(this_depths)

            count = 0
            for j, depth in enumerate(this_depths):

                for cre_line in cre_lines:

                    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                    if len(ecs) > 0:
                        if len(results_loc_dict[area+str(depth)+cre_line]) > 0:

                            results_array = np.array(results_loc_dict[area+str(depth)+cre_line])
                            for k in range(results_array.shape[0]):
                                ax[plot_depth_ind, i].errorbar(count+k, results_array[k].mean(), yerr = results_array[k].std() / float(len(results_array[k])), fmt='o', color=cre_colors[cre_line], markersize=5, ecolor=cre_colors[cre_line])

                            if j == 0:
                                ax[plot_depth_ind, i].set_title(area)
                            if i == 0:
                                ax[plot_depth_ind, i].set_ylabel(str(plot_depths[plot_depth_ind])+'-'+str(plot_depths[plot_depth_ind+1]))

                            count += k + 1

            ax[plot_depth_ind, i].plot(range(0, count), np.zeros((count)), 'k--')
            ax[plot_depth_ind, i].set_ylim((-1., 1.))

    fig.text(0.01, 0.5, 'Spearman Correlation of Noise & Signal Corrs', va='center', rotation='vertical')


def plot_r_noise_signal_corrs_one_histogram(plot_stim='natural_scenes', storage_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone', bootstrap=False):

    if bootstrap:
        results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='bootstrap_RhoSignalNoiseCorrs_list')

    else:
        results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir,
                                                 plot_key='RhoSignalNoiseCorrs')

    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    cre_colors = core.get_cre_colors()
    fail_tag = False

    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    plot_depths = [100, 200, 300, 500, 1000]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    first_exp = True

    for i, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):
            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]
            num_this_depths = len(this_depths)

            count = 0
            for j, depth in enumerate(this_depths):

                for cre_line in cre_lines:

                    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                    if len(ecs) > 0:
                        if len(results_loc_dict[area+str(depth)+cre_line]) > 0:

                            results_array = np.array(results_loc_dict[area+str(depth)+cre_line])
                            if first_exp:
                                results_full = results_array
                                first_exp = False
                            else:
                                results_full = np.concatenate((results_full, results_array), axis=0)

    if len(results_full.shape) == 2:
        ax.hist(results_full.mean(axis=1), bins=np.arange(-1, 1.1, .1), facecolor='k', density=False)
    else:
        ax.hist(results_full, bins=np.arange(-1, 1.1, .1), facecolor='k', density=False)

    ax.set_xlim((-1, 1))
    ax.plot(np.zeros((2)), [0, 50], 'k--')
    ax.set_xlabel('Corr. of noise and signal corrs, '+plot_stim, fontsize=24)
    ax.set_ylabel('Experiments (density)', fontsize=24)


    fig, ax = plt.subplots(1, 1)

    for cre_line in cre_lines:

        first_exp = True

        for area in areas:
            for depth in depths:
                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                    cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                if len(ecs) > 0:
                    if len(results_loc_dict[area + str(depth) + cre_line]) > 0:
                        results_array = np.array(results_loc_dict[area + str(depth) + cre_line])
                        if first_exp:
                            results_full = results_array
                            first_exp = False
                        else:
                            results_full = np.concatenate((results_full, results_array), axis=0)
        # ax.hist(results_full.mean(axis=1), facecolor=cre_colors[cre_line], alpha=0.5)
        if len(results_full.shape) == 2:
            ax.hist(results_full.mean(axis=1), histtype='step', bins=np.arange(-1, 1.1, .1), edgecolor=cre_colors[cre_line], linewidth=2, facecolor='none', density=False)
        else:
            ax.hist(results_full, histtype='step', bins=np.arange(-1, 1.1, .1), edgecolor=cre_colors[cre_line], linewidth=2, facecolor='none', density=False)

    ax.set_xlim((-1, 1))
    ax.plot(np.zeros((2)), [0, 20], 'k--')
    ax.set_xlabel('Corr. of noise and signal corrs, '+plot_stim)
    ax.set_ylabel('Experiments (density)')


def plot_dimensionality_one_histogram(plot_stim='natural_scenes', storage_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='DimensionFrac')

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    cre_colors = core.get_cre_colors()
    fail_tag = False

    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    plot_depths = [100, 200, 300, 500]

    fig, ax = plt.subplots(1, 1)

    first_exp = True

    for i, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):
            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]
            num_this_depths = len(this_depths)

            count = 0
            for j, depth in enumerate(this_depths):

                for cre_line in cre_lines:

                    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                    if len(ecs) > 0:
                        if len(results_loc_dict[area+str(depth)+cre_line]) > 0:

                            results_array = np.array(results_loc_dict[area+str(depth)+cre_line])
                            if first_exp:
                                results_full = results_array
                                first_exp = False
                            else:
                                results_full = np.concatenate((results_full, results_array), axis=0)

    ax.hist(results_full, bins=20, facecolor='k')
    ax.set_xlim((0, 1))
    ax.set_xlabel('Dimensionality Fraction')
    ax.set_ylabel('Experiments')

    fig, ax = plt.subplots(1, 1)

    for cre_line in cre_lines:

        first_exp = True

        for area in areas:
            for depth in depths:
                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                    cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                if len(ecs) > 0:
                    if len(results_loc_dict[area + str(depth) + cre_line]) > 0:
                        results_array = np.array(results_loc_dict[area + str(depth) + cre_line])
                        if first_exp:
                            results_full = results_array
                            first_exp = False
                        else:
                            results_full = np.concatenate((results_full, results_array), axis=0)
        # ax.hist(results_full.mean(axis=1), facecolor=cre_colors[cre_line], alpha=0.5)
        ax.hist(results_full, histtype='step', edgecolor=cre_colors[cre_line], linewidth=2, facecolor='none', label=cre_line, bins=np.arange(0, 1.1, .1), density=False)

    ax.legend(loc=0)
    ax.set_xlim((0, 1))
    ax.set_xlabel('Dimensionality Fraction')
    ax.set_ylabel('Experiments (density)')

    fig, ax = plt.subplots(1, 1)

    for i, plot_depth in enumerate(plot_depths[:-1]):

        this_depths = [d for d in depths if d>= plot_depth and d < plot_depths[i+1]]

        for depth in this_depths:
            for cre_line in cre_lines:
                for area in areas:
                    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                    if len(ecs) > 0:
                        if len(results_loc_dict[area + str(depth) + cre_line]) > 0:
                            results_array = np.array(results_loc_dict[area + str(depth) + cre_line])
                            if first_exp:
                                results_full = results_array
                                first_exp = False
                            else:
                                results_full = np.concatenate((results_full, results_array), axis=0)

        ax.hist(results_full, histtype='step', linewidth=2, facecolor='none', label=str(plot_depth)+'-'+str(plot_depths[i+1]), bins=np.arange(0, 1.1, .1), density=False)

    ax.legend(loc=0)
    ax.set_xlim((0, 1))
    ax.set_xlabel('Dimensionality Fraction')
    ax.set_ylabel('Experiments (density)')

    fig, ax = plt.subplots(1, 1)

    for i, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):
            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]
            num_this_depths = len(this_depths)

            count = 0
            for j, depth in enumerate(this_depths):

                for cre_line in cre_lines:

                    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)
                    if len(ecs) > 0:
                        if len(results_loc_dict[area+str(depth)+cre_line]) > 0:

                            results_array = np.array(results_loc_dict[area+str(depth)+cre_line])
                            if first_exp:
                                results_full = results_array
                                first_exp = False
                            else:
                                results_full = np.concatenate((results_full, results_array), axis=0)
        ax.hist(results_full, histtype='step', linewidth=2, facecolor='none', label=area, bins=np.arange(0, 1.1, .1), density=False)

    ax.set_xlim((0, 1))
    ax.legend(loc=0)
    ax.set_xlabel('Dimensionality Fraction')
    ax.set_ylabel('Experiments (density)')


def plot_example_noise_signal_corrs(plot_stim='natural_scenes', storage_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    noise_corr_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='NoiseCorr')
    signal_corr_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir,
                                            plot_key='SignalCorr')

    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    cre_colors = core.get_cre_colors()
    fail_tag = False

    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    plot_depths = [100, 200, 300, 500]

    fig, ax = plt.subplots()
    area = 'VISp'
    depth = 175
    cre_line = 'Cux2-CreERT2'

    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)

    noise_corr = np.array(noise_corr_dict[area+str(depth)+cre_line][0]).mean(axis=2)
    noise_corr = noise_corr[np.triu_indices(len(noise_corr), k=1)]
    sig_corr = np.array(signal_corr_dict[area+str(depth)+cre_line][0])

    r, p = spearmanr(sig_corr, noise_corr)

    plt.figure()
    plt.plot(sig_corr, noise_corr, 'ko')
    plt.text(.6, .8, r'$\rho=$'+str(np.round(r, decimals=3)))
    plt.xlabel('Signal correlation')
    plt.ylabel('Noise correlation')


def plot_example_noise_signal_corrs_hist(plot_stim='natural_scenes', storage_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    noise_corr_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='bootstrap_NoiseCorr_list')
    signal_corr_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir,
                                            plot_key='bootstrap_SignalCorr_list')

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    cre_colors = core.get_cre_colors()
    fail_tag = False

    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    plot_depths = [100, 200, 300, 500]

    fig, ax = plt.subplots()
    area = 'VISp'
    depth = 175
    cre_line = 'Cux2-CreERT2'

    ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth],
                                        cre_lines=[cre_line], simple=False, include_failed=fail_tag)

    noise_corr = np.array(noise_corr_dict[area+str(depth)+cre_line][0]).mean(axis=0)
    sig_corr = np.array(signal_corr_dict[area+str(depth)+cre_line][0]).mean(axis=0)

    H, xedges, yedges = np.histogram2d(sig_corr, noise_corr, bins=20, normed=True)

    plt.figure()
    plt.imshow(H, origin='low', extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap='gray_r')
    plt.colorbar()
    plt.xlabel('Signal correlation')
    plt.ylabel('Noise correlation')


def plot_run_stat_hist(cumulative=True, ttest=False):

    '''
    plot histograms of the distributions of noise and signal correlations computed from running or non-running trials
    if ttest, do a 2-sample t test for whether the mean of each correlation is different between the two behavioral conditions and write the results on the plot

    :param cumulative: whether to plot cumulative histograms
    :param ttest: whether to do the t-test
    :return: makes 2 plots, one for noise correlations and one for signal correlations
    '''

    corr_dir = '/allen/programs/braintv/workgroups/cortexmodels/gocker/cam_decode/noise_signal_corr_results_noBoots/'
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

    # areas = boc.get_all_targeted_structures()
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    depths = boc.get_all_imaging_depths()
    Nareas = len(areas)
    Ndepths = len(depths)

    plot_depths = [100, 200, 300, 500]

    stim = '_natural_scenes.pkl'

    fig1, ax1 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    fig2, ax2 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    bins = np.arange(-2., 2., .05)

    for j, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):

            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]

            noise_corr_hist = np.zeros(bins.shape[0] - 1)
            noise_corr_hist_run = np.zeros(bins.shape[0] - 1)
            noise_corr_hist_stat = np.zeros(bins.shape[0] - 1)

            signal_corr_hist = np.zeros(bins.shape[0] - 1)
            signal_corr_hist_run = np.zeros(bins.shape[0] - 1)
            signal_corr_hist_stat = np.zeros(bins.shape[0] - 1)

            noise_corr_all_run = []
            noise_corr_all_stat = []
            signal_corr_all_run = []
            signal_corr_all_stat = []

            num_run = 0
            num_stat = 0

            for i, depth in enumerate(this_depths):
                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth])

                ec_ids = [ec['id'] for ec in ecs]
                exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids)
                exp_ids = [exp['id'] for exp in exps]

                for exp_num, exp_id in enumerate(exp_ids):

                    if 'B' in exps[exp_num]['session_type']:  # session B is one with natural scenes

                        corr_path = os.path.join(corr_dir, str(exp_id) + stim)

                        if not os.path.exists(corr_path):
                            continue

                        corr_file = pickle.load(open(corr_path))

                        if ('NoiseCorrRun' in corr_file.keys()) and ('NoiseCorrStat' in corr_file.keys()):
                            noise_corr_bootstrap_list_run = corr_file['NoiseCorrRun']
                            signal_corr_bootstrap_list_run = corr_file['SignalCorrRun']
                            noise_corr_array_run = np.ma.masked_array(noise_corr_bootstrap_list_run, mask=np.isnan(noise_corr_bootstrap_list_run)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_run = np.ma.masked_array(signal_corr_bootstrap_list_run, mask=np.isnan(signal_corr_bootstrap_list_run)).reshape(-1, 1)

                            hist, _ = np.histogram(noise_corr_array_run, bins, normed=True)
                            if cumulative:
                                noise_corr_hist_run += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                noise_corr_hist_run += hist

                            hist, _ = np.histogram(signal_corr_array_run, bins, normed=True)
                            if cumulative:
                                signal_corr_hist_run += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                signal_corr_hist_run += hist

                            if (len(noise_corr_all_run) == 0):
                                noise_corr_all_run = noise_corr_array_run.copy()
                            else:
                                noise_corr_all_run = np.vstack((noise_corr_all_run, noise_corr_array_run))

                            if (len(signal_corr_all_run) == 0):
                                signal_corr_all_run = signal_corr_array_run
                            else:
                                signal_corr_all_run = np.vstack((signal_corr_all_run, signal_corr_array_run))

                            num_run += 1

                        if ('NoiseCorrRun' in corr_file.keys()) and ('NoiseCorrStat' in corr_file.keys()):
                            noise_corr_bootstrap_list_stat = corr_file['NoiseCorrStat']
                            signal_corr_bootstrap_list_stat = corr_file['SignalCorrStat']
                            noise_corr_array_stat = np.ma.masked_array(noise_corr_bootstrap_list_stat, mask=np.isnan(noise_corr_bootstrap_list_stat)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_stat = np.ma.masked_array(signal_corr_bootstrap_list_stat, mask=np.isnan(signal_corr_bootstrap_list_stat)).reshape(-1, 1)

                            hist, _ = np.histogram(noise_corr_array_stat, bins, normed=True)
                            if cumulative:
                                noise_corr_hist_stat += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                noise_corr_hist_stat += hist

                            hist, _ = np.histogram(signal_corr_array_stat, bins, normed=True)
                            if cumulative:
                                signal_corr_hist_stat += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                signal_corr_hist_stat += hist

                            if (len(noise_corr_all_stat) == 0):
                                noise_corr_all_stat = noise_corr_array_stat.copy()
                            else:
                                noise_corr_all_stat = np.vstack((noise_corr_all_stat, noise_corr_array_stat))

                            if (len(signal_corr_all_stat) == 0):
                                signal_corr_all_stat = signal_corr_array_stat
                            else:
                                signal_corr_all_stat = np.vstack((signal_corr_all_stat, signal_corr_array_stat))

                            num_stat += 1
                    else:
                        continue

            noise_corr_hist_run /= float(num_run)
            noise_corr_hist_stat /= float(num_stat)
            signal_corr_hist_run /= float(num_run)
            signal_corr_hist_stat /= float(num_stat)

            ax1[plot_depth_ind, j].plot(bins[:-1], noise_corr_hist_stat, 'k', label='Stat.', linewidth=2)
            ax1[plot_depth_ind, j].plot(bins[:-1], noise_corr_hist_run, 'c', label='Run', linewidth=2)

            ax2[plot_depth_ind, j].plot(bins[:-1], signal_corr_hist_stat, 'k', label='Stat.', linewidth=2)
            ax2[plot_depth_ind, j].plot(bins[:-1], signal_corr_hist_run, 'c', label='Run', linewidth=2)

            if i == 0:
                ax1[plot_depth_ind, j].set_title(str(area))
                ax2[plot_depth_ind, j].set_title(str(area))
            if j == 0:
                ax1[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))
                ax2[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))

            if (i==0) and (j==0):
                ax1[plot_depth_ind, j].legend(loc=0)
                ax2[plot_depth_ind, j].legend(loc=0)


            for ax in (ax1, ax2):
                ax[plot_depth_ind, j].set_xlim((-1., 1.))
                ax[plot_depth_ind, j].set_ylim((-1., 1.))
                ax[plot_depth_ind, j].set_xticks([-1., 0., 1.])
                ax[plot_depth_ind, j].set_yticks([-1., 0., 1.])

                ax[plot_depth_ind, j].spines['right'].set_visible(False)
                ax[plot_depth_ind, j].spines['top'].set_visible(False)
                ax[plot_depth_ind, j].xaxis.set_ticks_position('bottom')
                ax[plot_depth_ind, j].yaxis.set_ticks_position('left')


            if ttest:  # 2-sample related t-test

                plot_mean1 = np.mean(noise_corr_all_stat)
                plot_mean2 = np.mean(noise_corr_all_run)
                ax1[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes, color='k')
                ax1[plot_depth_ind, j].text(.55, .45, r'$\mu=$' + str(np.round(plot_mean2, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes, color='c')

                t, p = ttest_rel(noise_corr_all_stat, noise_corr_all_run)
                if p < 1e-5:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax1[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes)

                plot_mean1 = np.mean(signal_corr_all_stat)
                plot_mean2 = np.mean(signal_corr_all_run)
                ax2[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes, color='k')
                ax2[plot_depth_ind, j].text(.55, .45, r'$\mu=$' + str(np.round(plot_mean2, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes, color='c')

                t, p = ttest_rel(signal_corr_all_stat, signal_corr_all_run)
                if p < 1e-5:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax2[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes)


    fig1.text(0.5, 0.04, 'Noise Correlation', ha='center')
    fig2.text(0.5, 0.04, 'Signal Correlation', ha='center')


def plot_run_stat_diff_hist(cumulative=False, ttest=True):

    '''
    plot histograms of the difference in noise or signal correlation from running vs non-running trials
    if ttest, do a 2-sample t test for whether the mean of each correlation is different between the two behavioral conditions and write the results on the plot

    :param cumulative: whether to plot cumulative histograms
    :param ttest: whether to do the t-test
    :return: makes 2 plots, one for noise correlations and one for signal correlations
    '''

    corr_dir = '/allen/programs/braintv/workgroups/cortexmodels/gocker/cam_decode/noise_signal_corr_results_noBoots/'
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

    # areas = boc.get_all_targeted_structures()
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    depths = boc.get_all_imaging_depths()
    Nareas = len(areas)
    Ndepths = len(depths)

    plot_depths = [100, 200, 300, 500]

    stim = '_natural_scenes.pkl'

    fig1, ax1 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    fig2, ax2 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    bins = np.arange(-2., 2., .05)

    for j, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):

            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]

            noise_corr_hist = np.zeros(bins.shape[0] - 1)
            signal_corr_hist = np.zeros(bins.shape[0] - 1)

            noise_corr_all = []
            signal_corr_all = []

            num_run = 0
            num_stat = 0

            for i, depth in enumerate(this_depths):
                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth])

                ec_ids = [ec['id'] for ec in ecs]
                exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids)
                exp_ids = [exp['id'] for exp in exps]

                for exp_num, exp_id in enumerate(exp_ids):

                    if 'B' in exps[exp_num]['session_type']:  # session B is one with natural scenes

                        corr_path = os.path.join(corr_dir, str(exp_id) + stim)

                        if not os.path.exists(corr_path):
                            continue

                        corr_file = pickle.load(open(corr_path))

                        if ('NoiseCorrRun' in corr_file.keys()) and ('NoiseCorrStat' in corr_file.keys()):
                            noise_corr_bootstrap_list_run = corr_file['NoiseCorrRun']
                            signal_corr_bootstrap_list_run = corr_file['SignalCorrRun']
                            noise_corr_array_run = np.ma.masked_array(noise_corr_bootstrap_list_run, mask=np.isnan(noise_corr_bootstrap_list_run)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_run = np.ma.masked_array(signal_corr_bootstrap_list_run, mask=np.isnan(signal_corr_bootstrap_list_run)).reshape(-1, 1)

                            noise_corr_bootstrap_list_stat = corr_file['NoiseCorrStat']
                            signal_corr_bootstrap_list_stat = corr_file['SignalCorrStat']
                            noise_corr_array_stat = np.ma.masked_array(noise_corr_bootstrap_list_stat, mask=np.isnan(noise_corr_bootstrap_list_stat)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_stat = np.ma.masked_array(signal_corr_bootstrap_list_stat, mask=np.isnan(signal_corr_bootstrap_list_stat)).reshape(-1, 1)

                            noise_corr_hist_temp, _ = np.histogram(noise_corr_array_run - noise_corr_array_stat, normed=True, bins=bins)
                            signal_corr_hist_temp, _ = np.histogram(signal_corr_array_run - signal_corr_array_stat, normed=True, bins=bins)

                            if cumulative:
                                noise_corr_hist_temp = np.cumsum(noise_corr_hist_temp)*(bins[1]-bins[0])
                                signal_corr_hist_temp = np.cumsum(signal_corr_hist_temp)*(bins[1]-bins[0])

                            noise_corr_hist += noise_corr_hist_temp
                            signal_corr_hist += signal_corr_hist_temp


                            if (len(noise_corr_all) == 0):
                                noise_corr_all = noise_corr_array_run - noise_corr_array_stat
                            else:
                                noise_corr_all = np.vstack((noise_corr_all, noise_corr_array_run - noise_corr_array_stat))

                            if (len(signal_corr_all) == 0):
                                signal_corr_all = signal_corr_array_run - signal_corr_array_stat
                            else:
                                signal_corr_all = np.vstack((signal_corr_all, signal_corr_array_run - signal_corr_array_stat))

                            num_run += 1
                            num_stat += 1

                    else:
                        continue

            noise_corr_hist /= float(num_run)
            signal_corr_hist /= float(num_run)

            ax1[plot_depth_ind, j].plot(bins[:-1], noise_corr_hist, 'k', label='Run-Stat.', linewidth=2)
            ax2[plot_depth_ind, j].plot(bins[:-1], signal_corr_hist, 'k', label='Run-Stat.', linewidth=2)

            if i == 0:
                ax1[plot_depth_ind, j].set_title(str(area))
                ax2[plot_depth_ind, j].set_title(str(area))
            if j == 0:
                ax1[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))
                ax2[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))

            if (i==0) and (j==0):
                ax1[plot_depth_ind, j].legend(loc=0)
                ax2[plot_depth_ind, j].legend(loc=0)

            for ax in (ax1, ax2):
                ax[plot_depth_ind, j].set_xlim((-1., 1.))
                ax[plot_depth_ind, j].set_ylim((-1., 5.))
                ax[plot_depth_ind, j].set_xticks([-1., 0., 1.])
                if cumulative:
                    ax[plot_depth_ind, j].set_yticks([-1., 0., 1.])

                ax[plot_depth_ind, j].spines['right'].set_visible(False)
                ax[plot_depth_ind, j].spines['top'].set_visible(False)
                ax[plot_depth_ind, j].xaxis.set_ticks_position('bottom')
                ax[plot_depth_ind, j].yaxis.set_ticks_position('left')

                ax[plot_depth_ind, j].plot([0, 0], [-1, 10], 'k--')


            if ttest:  # 2-sample related t-test

                plot_mean1 = np.mean(noise_corr_all)
                ax1[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes, color='k')

                t, p = ttest_1samp(noise_corr_all, popmean=0.)
                if p < 1e-5:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax1[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes)

                plot_mean1 = np.mean(signal_corr_all)
                ax2[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)),
                                        transform=ax1[plot_depth_ind, j].transAxes, color='k')

                t, p = ttest_1samp(signal_corr_all, popmean=0.)
                if p < 1e-5:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax2[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes)


    fig1.text(0.5, 0.04, 'Noise Correlation', ha='center')
    fig2.text(0.5, 0.04, 'Signal Correlation', ha='center')


def plot_dilate_constrict_hist(cumulative=True, ttest=False):

    '''
    same as plot_run_stat_hist but for dilated and constricted pupils
    :param cumulative:
    :param ttest:
    :return:
    '''

    corr_dir = '/allen/programs/braintv/workgroups/cortexmodels/gocker/cam_decode/noise_signal_corr_results_noBoots/'
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

    # areas = boc.get_all_targeted_structures()
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    depths = boc.get_all_imaging_depths()
    Nareas = len(areas)
    Ndepths = len(depths)

    plot_depths = [100, 200, 300, 500]

    stim = '_natural_scenes.pkl'

    fig1, ax1 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    fig2, ax2 = plt.subplots(len(plot_depths)-1, Nareas, sharex=True, sharey=True, figsize=(10, 5))
    bins = np.arange(-2., 2., .05)

    for j, area in enumerate(areas):

        for plot_depth_ind, plot_depth in enumerate(plot_depths[:-1]):

            this_depths = [d for d in depths if (d >= plot_depth) and (d < plot_depths[plot_depth_ind+1])]

            noise_corr_hist = np.zeros(bins.shape[0] - 1)
            noise_corr_hist_Dilate = np.zeros(bins.shape[0] - 1)
            noise_corr_hist_Constrict = np.zeros(bins.shape[0] - 1)

            signal_corr_hist = np.zeros(bins.shape[0] - 1)
            signal_corr_hist_Dilate = np.zeros(bins.shape[0] - 1)
            signal_corr_hist_Constrict = np.zeros(bins.shape[0] - 1)

            noise_corr_all_Dilate = []
            noise_corr_all_Constrict = []
            signal_corr_all_Dilate = []
            signal_corr_all_Constrict = []

            num_Dilate = 0
            num_Constrict = 0

            for i, depth in enumerate(this_depths):
                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth])

                ec_ids = [ec['id'] for ec in ecs]
                exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids)
                exp_ids = [exp['id'] for exp in exps]

                for exp_num, exp_id in enumerate(exp_ids):

                    if 'B' in exps[exp_num]['session_type']:  # session B is one with natural scenes

                        corr_path = os.path.join(corr_dir, str(exp_id) + stim)

                        if not os.path.exists(corr_path):
                            continue

                        corr_file = pickle.load(open(corr_path))

                        if ('NoiseCorrDilate' in corr_file.keys()) and ('NoiseCorrConstrict' in corr_file.keys()):
                            noise_corr_bootstrap_list_Dilate = corr_file['NoiseCorrDilate']
                            signal_corr_bootstrap_list_Dilate = corr_file['SignalCorrDilate']
                            noise_corr_array_Dilate = np.ma.masked_array(noise_corr_bootstrap_list_Dilate, mask=np.isnan(noise_corr_bootstrap_list_Dilate)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_Dilate = np.ma.masked_array(signal_corr_bootstrap_list_Dilate, mask=np.isnan(signal_corr_bootstrap_list_Dilate)).reshape(-1, 1)

                            hist, _ = np.histogram(noise_corr_array_Dilate, bins, normed=True)
                            if cumulative:
                                noise_corr_hist_Dilate += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                noise_corr_hist_Dilate += hist

                            hist, _ = np.histogram(signal_corr_array_Dilate, bins, normed=True)
                            if cumulative:
                                signal_corr_hist_Dilate += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                signal_corr_hist_Dilate += hist

                            if (len(noise_corr_all_Dilate) == 0):
                                noise_corr_all_Dilate = noise_corr_array_Dilate.copy()
                            else:
                                noise_corr_all_Dilate = np.vstack((noise_corr_all_Dilate, noise_corr_array_Dilate))

                            if (len(signal_corr_all_Dilate) == 0):
                                signal_corr_all_Dilate = signal_corr_array_Dilate
                            else:
                                signal_corr_all_Dilate = np.vstack((signal_corr_all_Dilate, signal_corr_array_Dilate))

                            num_Dilate += 1

                        if ('NoiseCorrDilate' in corr_file.keys()) and ('NoiseCorrConstrict' in corr_file.keys()):
                            noise_corr_bootstrap_list_Constrict = corr_file['NoiseCorrConstrict']
                            signal_corr_bootstrap_list_Constrict = corr_file['SignalCorrConstrict']
                            noise_corr_array_Constrict = np.ma.masked_array(noise_corr_bootstrap_list_Constrict, mask=np.isnan(noise_corr_bootstrap_list_Constrict)).mean(axis=2).reshape(-1, 1) # average over stim
                            signal_corr_array_Constrict = np.ma.masked_array(signal_corr_bootstrap_list_Constrict, mask=np.isnan(signal_corr_bootstrap_list_Constrict)).reshape(-1, 1)

                            hist, _ = np.histogram(noise_corr_array_Constrict, bins, normed=True)
                            if cumulative:
                                noise_corr_hist_Constrict += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                noise_corr_hist_Constrict += hist

                            hist, _ = np.histogram(signal_corr_array_Constrict, bins, normed=True)
                            if cumulative:
                                signal_corr_hist_Constrict += np.cumsum(hist)*(bins[1]-bins[0])
                            else:
                                signal_corr_hist_Constrict += hist

                            if (len(noise_corr_all_Constrict) == 0):
                                noise_corr_all_Constrict = noise_corr_array_Constrict.copy()
                            else:
                                noise_corr_all_Constrict = np.vstack((noise_corr_all_Constrict, noise_corr_array_Constrict))

                            if (len(signal_corr_all_Constrict) == 0):
                                signal_corr_all_Constrict = signal_corr_array_Constrict
                            else:
                                signal_corr_all_Constrict = np.vstack((signal_corr_all_Constrict, signal_corr_array_Constrict))

                            num_Constrict += 1
                    else:
                        continue

            noise_corr_hist_Dilate /= float(num_Dilate)
            noise_corr_hist_Constrict /= float(num_Constrict)
            signal_corr_hist_Dilate /= float(num_Dilate)
            signal_corr_hist_Constrict /= float(num_Constrict)

            ax1[plot_depth_ind, j].plot(bins[:-1], noise_corr_hist_Constrict, 'k', label='Constrict.', linewidth=2)
            ax1[plot_depth_ind, j].plot(bins[:-1], noise_corr_hist_Dilate, 'c', label='Dilate', linewidth=2)

            ax2[plot_depth_ind, j].plot(bins[:-1], signal_corr_hist_Constrict, 'k', label='Constrict.', linewidth=2)
            ax2[plot_depth_ind, j].plot(bins[:-1], signal_corr_hist_Dilate, 'c', label='Dilate', linewidth=2)

            if i == 0:
                ax1[plot_depth_ind, j].set_title(str(area))
                ax2[plot_depth_ind, j].set_title(str(area))
            if j == 0:
                ax1[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))
                ax2[plot_depth_ind, j].set_ylabel(str(plot_depths[plot_depth_ind]))


            for ax in (ax1, ax2):
                ax[plot_depth_ind, j].set_xlim((-1., 1.))
                ax[plot_depth_ind, j].set_ylim((-1., 1.))
                ax[plot_depth_ind, j].set_xticks([-1., 0., 1.])
                ax[plot_depth_ind, j].set_yticks([-1., 0., 1.])

                ax[plot_depth_ind, j].spines['right'].set_visible(False)
                ax[plot_depth_ind, j].spines['top'].set_visible(False)
                ax[plot_depth_ind, j].xaxis.set_ticks_position('bottom')
                ax[plot_depth_ind, j].yaxis.set_ticks_position('left')


            if ttest:  # 2-sample related t-test

                plot_mean1 = np.mean(noise_corr_all_Constrict)
                plot_mean2 = np.mean(noise_corr_all_Dilate)
                ax1[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes, color='k')
                ax1[plot_depth_ind, j].text(.55, .45, r'$\mu=$' + str(np.round(plot_mean2, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes, color='c')

                t, p = ttest_rel(noise_corr_all_Constrict, noise_corr_all_Dilate)
                if p < 1e-5:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax1[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax1[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax1[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax1[plot_depth_ind, j].transAxes)

                plot_mean1 = np.mean(signal_corr_all_Constrict)
                plot_mean2 = np.mean(signal_corr_all_Dilate)
                ax2[plot_depth_ind, j].text(.55, .55, r'$\mu=$' + str(np.round(plot_mean1, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes, color='k')
                ax2[plot_depth_ind, j].text(.55, .45, r'$\mu=$' + str(np.round(plot_mean2, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes, color='c')

                t, p = ttest_rel(signal_corr_all_Constrict, signal_corr_all_Dilate)
                if p < 1e-5:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-5}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-4:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-4}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-3:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-3}$', transform=ax2[plot_depth_ind, j].transAxes)
                elif p < 1e-2:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p<10^{-2}$', transform=ax2[plot_depth_ind, j].transAxes)
                # elif p < .05:
                else:
                    ax2[plot_depth_ind, j].text(.55, .65, r'$p=$'+str(np.round(p, decimals=3)), transform=ax2[plot_depth_ind, j].transAxes)


    fig1.text(0.5, 0.04, 'Noise Correlation', ha='center')
    fig2.text(0.5, 0.04, 'Signal Correlation', ha='center')


def find_exp_negativeRNoiseSignalCorrs():

    results_loc_dict = make_results_loc_dict(stim_type=plot_stim, results_path=storage_dir, plot_key='bootstrap_RhoSignalNoiseCorrs_list')
    r_keys = [r for r in results_loc_dict.keys() if 'VISam' in r]

    mean_r_dict = dict()

    for r in r_keys:

        mean_r = []
        for exp in results_loc_dict[r]:
            mean_r.append(np.mean(exp))

        mean_r_dict[r] = mean_r



if __name__ == '__main__':

    main_missing_exps(stim=sys.argv[1])

    # main_expt(expt=sys.argv[1], filter_window=None, calc_stim=['natural_scenes', 'drifting_gratings', 'static_gratings'])
    # main_expt_bootstrap(expt=sys.argv[1], filter_window=10,
    #                     calc_stim=['natural_movie_one', 'natural_movie_two', 'natural_movie_three'], num_boots=50)



    # main_expt_bootstrap(expt=sys.argv[1], filter_window=None, calc_stim=['natural_scenes', 'drifting_gratings', 'static_gratings'], num_boots=50)
    # main_expt_bootstrap(expt=sys.argv[1], filter_window=10,
    #                     calc_stim=['natural_movie_one', 'natural_movie_two', 'natural_movie_three'], num_boots=50)
    #


    # main_missing_exps(stim='natural_scenes')
    # main_missing_exps(stim='static_gratings')
    # main_missing_exps(stim='drifting_gratings')
    # main_missing_exps(stim='natural_movie_one')


    # main_missing_exps(stim='natural_movie_one', results_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone')
    # main_missing_exps(stim='natural_movie_two', results_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone')
    # main_missing_exps(stim='natural_movie_three', results_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone')
