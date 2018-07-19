### functions for computing correlations

import os, sys, pickle, corr_functions
import numpy as np
from pandas import HDFStore
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import sklearn.mixture as mixture
from scipy.stats import ttest_rel, ttest_1samp, spearmanr
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from visual_coding_2p_analysis.l0_analysis import L0_analysis
from visual_coding_2p_analysis import core
from visual_coding_2p_analysis.decoding import label_run_stationary_dpgmm, get_tables_exp, interpolate_pupil_size

manifest_file = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file=manifest_file)

def get_response_table(dff, stim_table, stim_type, log=False, shift=0, width=None):

    Ncells = dff.shape[0]

    if stim_type == 'spontaneous':
        Nstim = 0
        for start, end in zip(stim_table['start'].values, stim_table['end'].values):
            Nstim += end-start

        response_array = np.zeros((Ncells, Nstim))

        Nstim = 0
        for start, end in zip(stim_table['start'].values, stim_table['end'].values):
            response_array[:, Nstim:Nstim+(end-start)] = dff[:, start:end]
            Nstim += end-start

    else:

        stim_lengths = stim_table.end.values - stim_table.start.values
        Nstim = len(stim_table)

        if width is None:  # use stim length - this should correspond to the SDK sweep response

            if stim_type == 'dg' or stim_type == 'drifting_gratings':
                ind_start = stim_table.start.values + shift
                ind_end = ind_start + np.amin(stim_lengths)

            elif 'nm' in stim_type or 'natural_movie' in stim_type:
                # ind_start = np.arange(start=stim_table.start.values[0], stop=stim_table.end.values[-1], step=33)  # 1 second bins
                ind_start = stim_table.start.values + shift
                ind_end = ind_start+1
            else:
                ind_start = stim_table.start.values + shift
                ind_end = ind_start + np.amin(stim_lengths)

        else:  # use pre-defined window width

            ind_start = stim_table.start.values + shift
            ind_end = ind_start + width


        ''' get response array '''
        response_array = np.zeros((Ncells, Nstim))
        for i in range(Nstim):
            response_array[:, i] = np.mean(dff[:, ind_start[i]:ind_end[i]], axis=1)

        if log:
            response_array += 1. - np.amin(response_array)
            response_array = np.log(response_array)


    return response_array.T  # stim x cells


def reshape_response_table(response_table, stim_table, stim_type, shuffle=True, min_reps=9):

    '''
    take in response table of stim x cells and reshape to cells x stim identity x repeats
    :param response_table:
    :param stim_table:
    :param stim_type:
    :return:
    '''

    if len(response_table.shape) != 2:
        raise Exception('response table for reshape_response_table() should be Nstim x Ncells')

    response_table = response_table.T # cells x stim
    Ncells, Ntrials = response_table.shape

    if shuffle:
        ind = np.random.choice(range(Ntrials), size=Ntrials, replace=False)
        response_table = response_table[:, ind]
        stim_table = stim_table.iloc[ind]

    if stim_type == 'natural_scenes' or stim_type == 'ns' or 'natural_movie' in stim_type or 'nm' in stim_type:

        stims = np.unique(stim_table['frame'])
        Nstim = len(stims)

        Nreps = []
        for i, frame in enumerate(stims):
            stim_ind = (stim_table['frame'] == frame)
            if sum(stim_ind) < min_reps:
                stim_table = stim_table.drop(stim_table.index[stim_ind])
                response_table = response_table[:, ~stim_ind]
                stims = [s for s in stims if (s != frame)]
            else:
               Nreps.append(sum(stim_ind))

        if len(Nreps) >= 1:
            Nreps = min(Nreps)

            response_array = np.zeros((Ncells, Nstim, Nreps))
            for i, stim in enumerate(stims):

                stim_ind = (stim_table['frame'] == stim)
                response_temp = response_table[:, stim_ind]
                response_temp = response_temp[:, :Nreps]
                response_array[:, i] = response_temp

            return response_array

        else:
            return None

    elif stim_type == 'static_gratings' or stim_type == 'sg':

        oris = np.unique(stim_table['orientation'])
        oris = oris[~np.isnan(oris)]

        sfs = np.unique(stim_table['spatial_frequency'])
        sfs = sfs[~np.isnan(sfs)]

        phases = np.unique(stim_table['phase'])
        phases = phases[~np.isnan(phases)]

        Nstim = len(oris)*len(sfs)*len(phases) + 1 # +1 for blank

        Nreps = []
        stim_ind = (~np.isfinite(stim_table['orientation'])) * (~np.isfinite(stim_table['spatial_frequency'])) * (~np.isfinite(stim_table['phase'])) # blank sweep
        Nreps.append(sum(stim_ind))

        for i, ori in enumerate(oris):
            for j, sf in enumerate(sfs):
                for k, phase in enumerate(phases):
                    stim_ind = (stim_table['orientation'] == ori) * (stim_table['spatial_frequency'] == sf) * (stim_table['phase'] == phase)

                    if sum(stim_ind) < min_reps:
                        stim_table = stim_table.drop(stim_table.index[stim_ind])
                        response_table = response_table[:, ~stim_ind]
                    else:
                       Nreps.append(sum(stim_ind))

        if len(Nreps) >= 1:
            Nreps = min(Nreps)
            response_array = np.zeros((Ncells, Nstim, Nreps))
            stim_ind = (~np.isfinite(stim_table['orientation'])) * (~np.isfinite(stim_table['spatial_frequency'])) * (~np.isfinite(stim_table['phase'])) # blank sweep
            response_temp = response_table[:, stim_ind]
            response_array[:, 0, :] = response_temp[:, :Nreps]

            count = 1
            for i, ori in enumerate(oris):
                for j, sf in enumerate(sfs):
                    for k, phase in enumerate(phases):
                        stim_ind = (stim_table['orientation'] == ori) * (stim_table['spatial_frequency'] == sf) * (stim_table['phase'] == phase)
                        if sum(stim_ind) >= min_reps:
                            response_temp = response_table[:, stim_ind]
                            response_array[:, count, :] = response_temp[:, :Nreps]
                            count += 1

            return response_array

        else:
            return None


    elif stim_type == 'drifting_gratings' or stim_type == 'dg':

        oris = np.unique(stim_table['orientation'])
        oris = oris[~np.isnan(oris)]

        tfs = np.unique(stim_table['temporal_frequency'])
        tfs = tfs[~np.isnan(tfs)]


        Nstim = len(oris)*len(tfs) + 1 # +1 for blank

        Nreps = []
        stim_ind = (~np.isfinite(stim_table['orientation'])) * (~np.isfinite(stim_table['temporal_frequency'])) # blank sweep
        Nreps.append(sum(stim_ind))

        for i, ori in enumerate(oris):
            for j, tf in enumerate(tfs):
                stim_ind = (stim_table['orientation'] == ori) * (stim_table['temporal_frequency'] == tf)

                if sum(stim_ind) < min_reps:
                    stim_table = stim_table.drop(stim_table.index[stim_ind])
                    response_table = response_table[:, ~stim_ind]
                else:
                    Nreps.append(sum(stim_ind))

        if len(Nreps) >= 1:
            Nreps = min(Nreps)
            response_array = np.zeros((Ncells, Nstim, Nreps))
            stim_ind = (~np.isfinite(stim_table['orientation'])) * (~np.isfinite(stim_table['temporal_frequency'])) # blank sweep
            response_temp = response_table[:, stim_ind]
            response_array[:, 0, :] = response_temp[:, :Nreps]

            count = 1
            for i, ori in enumerate(oris):
                for j, tf in enumerate(tfs):
                        stim_ind = (stim_table['orientation'] == ori) * (stim_table['temporal_frequency'] == tf)

                        if sum(stim_ind) >= min_reps:
                            response_temp = response_table[:, stim_ind]
                            response_array[:, count, :] = response_temp[:, :Nreps]
                            count += 1

            return response_array

        else:
            return None

    else:
        raise Exception('need to incorporate %s in reshape_response_table()' % stim_type)


def label_run_stationary_dpgmm(running_array, alpha_par=.2):
    '''

    :param running_array: array of running speeds
    :param alpha_par: concentration parameter for dirichlet process, .2 seems good for speeds averaged over 7 frames
    :return:
    '''

    import sklearn.mixture as mixture
    import numpy as np

    X = np.array(running_array).reshape(-1, 1)
    dpgmm = mixture.DPGMM(n_components=10, alpha=alpha_par, covariance_type='diag', n_iter=1000, tol=1e-3)
    dpgmm.fit(X)
    Y = dpgmm.predict(X)
    states = np.unique(Y)
    means = dpgmm.means_
    vars = dpgmm._get_covars()

    labels = Y.reshape(running_array.shape)

    return labels, means, vars


def rank_response_array(response_array):
    '''
    rank trials for each neuron and stimulus
    :param response_array:
    :return:
    '''

    if len(response_array.shape) == 3:
        N, Nstim, Ntrials = response_array.shape
        rank_array = np.zeros((N, Nstim, Ntrials))

        for n in range(N):
            for ns in range(Nstim):
                rank_array[n, ns] = np.argsort(np.argsort(response_array[n, ns]))

    elif len(response_array.shape) == 2:
        N, Nstim = response_array.shape
        rank_array = np.zeros((N, Nstim))

        for n in range(N):
            rank_array[n] = np.argsort(np.argsort(response_array[n]))

    return rank_array


def regress_runSpeed(response_table, run_table):

    Nstim, N = response_table.shape
    residuals = np.zeros((Nstim, N))
    run_table -= run_table.mean()

    for n in range(N):
        response = response_table[:, n]
        response -= response.mean()
        residuals[:, n] = response - lstsq(run_table, response)[0]

    return residuals


def compute_noise_signal_corrs_dimension(response_array, corr='pearson'):

    # calculate signal and noise correlations for that bootstrap sample
    Ncells, Nstim, Ntrials = response_array.shape

    response_array_mean = np.mean(response_array, axis=2)
    if corr == 'spearman':
        response_rank = rank_response_array(response_array_mean).astype(np.float32)
    else:
        response_rank = response_array_mean

    signal_cov = np.cov(response_rank)
    signal_std = np.sqrt(np.diag(signal_cov))
    signal_corr = signal_cov / np.outer(signal_std, signal_std)

    noise_corr = np.zeros((Ncells, Ncells, Nstim))
    noise_cov = np.zeros((Ncells, Ncells, Nstim))

    for n in range(Nstim):

        if corr == 'spearman':
            response_rank = rank_response_array(response_array[:, n, :]).astype(np.float32)
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


def compute_spont_corrs_dimension(response_array, corr='pearson'):

    Ncells, Ntrials = response_array.shape

    if corr == 'spearman':
        response_rank = rank_response_array(response_array)
    else:
        response_rank = response_array

    spont_cov = np.cov(response_rank)
    spont_std = np.sqrt(np.diag(spont_cov))
    spont_corr = spont_cov / np.outer(spont_std, spont_std)

    cov_eigs = np.linalg.eigvalsh(spont_cov)
    dimension_frac = (np.sum(cov_eigs) ** 2) / np.sum(cov_eigs ** 2) / float(Ncells)
    dimension = (np.sum(cov_eigs) ** 2) / np.sum(cov_eigs ** 2)


    return spont_corr, dimension, dimension_frac


def main_expt_bootstrap(expt, num_boots=50, regress_run='none', run_thresh=1., run_window=1., detect_events=True, filter_window=None, corr='pearson', split=True, calc_stim=['natural_scenes', 'static_gratings', 'drifting_gratings', 'natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'spontaneous'], save_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations_bootstrap_split/regressNone'):

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
            response_table = get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
            running_speed = get_response_table(dxcm, stim_table, stim_type=stim, width=filter_window)

            if regress_run == 'speed':
                response_table = regress_runSpeed(response_table, running_speed)
            elif regress_run == 'state':
                response_table = regress_runState(response_table, running_speed)

            response_array = reshape_response_table(response_table, stim_table, stim_type=stim)
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

            response_table = get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
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


def main_expt(expt, regress_run='none', run_thresh=1., run_window=1, filter_window=None, corr='pearson', detect_events=True, calc_stim=['natural_scenes', 'static_gratings', 'drifting_gratings', 'natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'spontaneous'], save_dir_base='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations'):

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

            response_table = get_response_table(dff=dff, stim_table=stim_table, stim_type=stim, width=filter_window)
            running_speed = get_response_table(dxcm, stim_table, stim_type=stim, width=filter_window)

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
                response_array = reshape_response_table(response_table, stim_table, stim_type=stim)
                if response_array is not None and np.prod(response_array.shape) > 0:
                    results['NoiseCorr'], results['SignalCorr'], results['RhoSignalNoiseCorrs'], results['NoiseDimension'], results['NoiseDimensionFrac'] = compute_noise_signal_corrs_dimension(response_array, corr)


                if np.sum(~ind_stationary) > 9:
                    response_array = reshape_response_table(response_table_run, stim_table[~ind_stationary], stim_type=stim, shuffle=True)
                    if response_array is not None and np.prod(response_array.shape) > 0:
                        results['NoiseCorrRun'], results['SignalCorrRun'], results['RhoSignalNoiseCorrsRun'], results['NoiseDimensionRun'], results['NoiseDimensionFracRun'] = compute_noise_signal_corrs_dimension(response_array, corr)

                if np.sum(ind_stationary) > 9:
                    response_array = reshape_response_table(response_table_stat, stim_table[ind_stationary], stim_type=stim, shuffle=True)
                    if response_array is not None and np.prod(response_array.shape) > 0:
                        results['NoiseCorrStat'], results['SignalCorrStat'], results['RhoSignalNoiseCorrsStat'], results['NoiseDimensionStat'], results['NoiseDimensionFracStat'] = compute_noise_signal_corrs_dimension(response_array, corr)

            error_file = open(savefile, 'wb')
            pickle.dump(results, error_file)
            error_file.close()
