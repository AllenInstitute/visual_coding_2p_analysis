# -*- coding: utf-8 -*-

import os, sys, time, pickle, argparse
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from visual_coding_2p_analysis import core, pawplot, strip_plot
import h5py

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    manifest_file = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache/brain_observatory_manifest.json'
    boc = BrainObservatoryCache(manifest_file=manifest_file)

elif sys.platform == 'linux2':
    try:
        # manifest_file = '/allen/aibs/mat/gkocker/BrainObservatory/boc/manifest.json'
        # manifest_file = '/allen/aibs/technology/allensdk_data/platform_boc_pre_2018_3_16/manifest.json'
        # manifest_file = core.get_manifest_path()
        manifest_file = '/allen/aibs/mat/michaelbu/platform_boc_pre_2018_3_16/manifest.json'

        boc = BrainObservatoryCache(manifest_file=manifest_file)

    except:
        manifest_file = 'boc/manifest.json'
        boc = BrainObservatoryCache(manifest_file=manifest_file)

from cam_decode_final import make_results_loc_dict, plot_results_compare_method_diff
from compute_noise_signal_corr import make_results_loc_dict as make_results_loc_dict_corrs

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

areas = boc.get_all_targeted_structures()


def example_response_table(analysis_file_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/analysis_files_pre_2018_3_29', stim='drifting_gratings'):

    exps = boc.get_ophys_experiments(targeted_structures=['VISp'], cre_lines=['Cux2-CreERT2'], imaging_depths=[175], stimuli=[stim])
    exp_id = exps[0]['id']

    if stim == 'drifting_gratings':
        short_stim = 'dg'
        analysis_stim = 'DriftingGratings'
    elif stim == 'natural_scenes':
        short_stim = 'ns'
        analysis_stim = 'NaturalScenes'

    analysis_file = h5py.File(os.path.join(analysis_file_dir, analysis_stim, str(exp_id) + '_' + short_stim + '_events_analysis.h5'))
    response = analysis_file['mean_sweep_events'].values()[3].value  # trials x neurons

    T, N = response.shape

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(response.T, cmap='gray_r', aspect=2)
    ax.set_yticks(range(0, N, int(np.floor(N/5))))
    plt.tick_params(labelsize=20)

    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Neuron', fontsize=20)
    fig.tight_layout()


def example_rho_noise_signal_correlation(stim='natural_scenes', cre_line='Slc17a7-IRES2-Cre', area='VISp', depth=175, results_path='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    exps = boc.get_ophys_experiments(targeted_structures=[area], cre_lines=[cre_line], imaging_depths=[depth], stimuli=[stim])
    exp_id = exps[0]['id']

    results_file = pickle.load(open(os.path.join(results_path, str(exp_id)+'_'+stim+'.pkl'), 'r'))

    noise_corr = results_file['NoiseCorr'].mean(axis=2)
    Ncells = noise_corr.shape[0]
    inds = np.triu_indices(n=Ncells, k=1)
    noise_corr = noise_corr[inds]

    sig_corr = results_file['SignalCorr']

    intercept = np.mean(noise_corr)
    slope = results_file['RhoSignalNoiseCorrs']
    x = np.arange(-1, 2, .001)
    y = slope*(x-np.mean(sig_corr)) + intercept

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(sig_corr, noise_corr, 'ko', markersize=1)
    ax.plot(x, y, 'k', linewidth=2)

    plt.text(.5, .3, r'$r=$'+str(np.round(slope, 2)), fontsize=20)

    ax.set_xlim((-.3, 1))
    ax.set_ylim((-.3, 1))
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([0, .5, 1])
    ax.plot([0, 0], [-.3, 1], 'k', linewidth=.5)
    ax.plot([-.3, 1], [0, 0], 'k', linewidth=.5)
    ax.set_axis_bgcolor('w')

    plt.tick_params(labelsize=20)
    ax.set_xlabel('Signal Corr.', fontsize=24)
    ax.set_ylabel('Noise Corr.', fontsize=24)
    fig.tight_layout()

    savefile = '/allen/aibs/mat/gkocker/bob_platform_plots/example_noise_signal_corr.pdf'
    fig.savefig(savefile)


    bins = np.arange(-.3, 1.1, .01)
    H, xedges, yedges = np.histogram2d(sig_corr, noise_corr, bins=bins)

    fig, ax = plt.subplots(1, figsize=(5, 5))

    ax.imshow(H.T, extent=(-.3, 1.1, 1.1, -.3), cmap='plasma')
    ax.set_xlim((-.3, 1))
    ax.set_ylim((-.3, 1))
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([0, .5, 1])
    ax.plot([0, 0], [-.3, 1], 'k', linewidth=.5)
    ax.plot([-.3, 1], [0, 0], 'k', linewidth=.5)

    plt.tick_params(labelsize=20)
    ax.set_xlabel('Signal Corr.', fontsize=24)
    ax.set_ylabel('Noise Corr.', fontsize=24)
    fig.tight_layout()

    savefile = '/allen/aibs/mat/gkocker/bob_platform_plots/example_noise_signal_corr_hist.pdf'
    fig.savefig(savefile)



def make_results_loc_array_include_keys(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False, relative=False, relative_divide=False, meta='L0events', include_keys=['Sst']):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''


    if stim_type == 'ns':
        stim_type_full = 'natural_scenes'
    elif stim_type == 'dg':
        stim_type_full = 'drifting_gratings'
    elif stim_type == 'sg':
        stim_type_full = 'static_gratings'
    elif stim_type == 'nm1':
        stim_type_full = 'natural_movie_one'
    elif stim_type == 'nm2':
        stim_type_full = 'natural_movie_two'
    elif stim_type == 'nm3':
        stim_type_full = 'natural_movie_three'

    areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = boc.get_all_imaging_depths()
    layers = [100, 200, 300, 500, 1000]
    results_array = np.zeros((len(areas), len(layers)-1, 2))

    if sys.platform == 'darwin':
        results_path = '/Volumes/Brain2017/decode_results'
    elif sys.platform == 'linux2':
        results_path = '/local1/Documents/projects/cam_analysis/decode_results'

    if type(meta) is str:
        results_path += '_'+meta

    results_loc_dict1, shuffle_loc_dict1 = make_results_loc_dict(stim_type=stim_type, stim_features=stim_features, plot_methods=plot_methods, relative=relative, error=error, results_path=results_path)
    results_keys = results_loc_dict1.keys()


    for a, area in enumerate(areas):
        if area == 'VISp':
            r_keys = [k for k in results_keys if ('VISp' in k) and ('VISpm' not in k)]
        else:
            r_keys = [k for k in results_keys if k[:len(area)] == area]

        for l, layer in enumerate(layers[:-1]):
            depth_range = [d for d in depths if (d >= layer) and (d < (layers[l+1]))]

            error_list1 = []
            shuffle_list1 = []

            r_keys1 = [r for r in r_keys if any(str(d) in r for d in depth_range) and any(k in r for k in include_keys)]

            for r in r_keys1:

                error_dict1 = results_loc_dict1[r][plot_methods[0]]
                shuffle_dict1 = shuffle_loc_dict1[r][plot_methods[0]]

                if len(error_dict1) > 0:
                    for k in range(len(error_dict1)):
                        for kk in range(len(error_dict1[k])):
                            key_temp = error_dict1[k].keys()[kk]
                            error_list1.append(error_dict1[k][key_temp].mean())
                            shuffle_list1.append(shuffle_dict1[k][key_temp].mean())

            error_list1 = np.array(error_list1)
            shuffle_list1 = np.array(shuffle_list1)
            if relative:
                if relative_divide:
                    error_list1 /= shuffle_list1
                else:
                    error_list1 -= shuffle_list1

            results_array[a, l, 0] = np.mean(error_list1)
            results_array[a, l, 1] = len(error_list1)

    return results_array


def make_results_diff_loc_array_include_keys(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors', 'ShuffledKNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False, relative=False, relative_divide=False, meta='L0events', include_keys=['Sst']):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''


    if stim_type == 'ns':
        stim_type_full = 'natural_scenes'
    elif stim_type == 'dg':
        stim_type_full = 'drifting_gratings'
    elif stim_type == 'sg':
        stim_type_full = 'static_gratings'
    elif stim_type == 'nm1':
        stim_type_full = 'natural_movie_one'
    elif stim_type == 'nm2':
        stim_type_full = 'natural_movie_two'
    elif stim_type == 'nm3':
        stim_type_full = 'natural_movie_three'

    areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = boc.get_all_imaging_depths()
    layers = [100, 200, 300, 500, 1000]
    results_array = np.zeros((len(areas), len(layers)-1, 2))

    if sys.platform == 'darwin':
        results_path = '/Volumes/Brain2017/decode_results'
    elif sys.platform == 'linux2':
        results_path = '/local1/Documents/projects/cam_analysis/decode_results'

    if type(meta) is str:
        results_path += '_'+meta

    results_loc_dict1, shuffle_loc_dict1 = make_results_loc_dict(stim_type=stim_type, stim_features=stim_features, plot_methods=plot_methods, relative=relative, error=error, results_path=results_path)
    results_keys = results_loc_dict1.keys()



    for a, area in enumerate(areas):
        if area == 'VISp':
            r_keys = [k for k in results_keys if ('VISp' in k) and ('VISpm' not in k)]
        else:
            r_keys = [k for k in results_keys if k[:len(area)] == area]

        for l, layer in enumerate(layers[:-1]):
            depth_range = [d for d in depths if (d >= layer) and (d < (layers[l+1]))]

            error_list1, shuffle_list1, error_list2, shuffle_list2 = [], [], [], []

            r_keys1 = [r for r in r_keys if any(str(d) in r for d in depth_range) and any(k in r for k in include_keys)]

            for r in r_keys1:

                error_dict1 = results_loc_dict1[r][plot_methods[0]]
                shuffle_dict1 = shuffle_loc_dict1[r][plot_methods[0]]

                error_dict2 = results_loc_dict1[r][plot_methods[1]]
                shuffle_dict2 = shuffle_loc_dict1[r][plot_methods[1]]

                if len(error_dict1) > 0:
                    for k in range(len(error_dict1)):
                        for kk in range(len(error_dict1[k])):
                            key_temp = error_dict1[k].keys()[kk]
                            error_list1.append(error_dict1[k][key_temp].mean())
                            shuffle_list1.append(shuffle_dict1[k][key_temp].mean())

                            key_temp = error_dict2[k].keys()[kk]
                            error_list2.append(error_dict2[k][key_temp].mean())
                            shuffle_list2.append(shuffle_dict2[k][key_temp].mean())

            error_list1, error_list2, shuffle_list1, shuffle_list2 = np.array(error_list1), np.array(error_list2), np.array(shuffle_list1), np.array(shuffle_list2)

            if relative:
                if relative_divide:
                    error_list1 /= shuffle_list2
                    error_list2 /= shuffle_list2
                else:
                    error_list1 -= shuffle_list1
                    error_list2 -= shuffle_list2

            if relative_divide:
                results_array[a, l, 0] = np.mean(error_list1 / error_list2)
            else:
                results_array[a, l, 0] = np.mean(error_list1 - error_list2)

            results_array[a, l, 1] = len(error_list1)

    return results_array



def make_results_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False, relative=False, relative_divide=False, meta='L0events', exclude_keys=[]):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''


    if stim_type == 'ns':
        stim_type_full = 'natural_scenes'
    elif stim_type == 'dg':
        stim_type_full = 'drifting_gratings'
    elif stim_type == 'sg':
        stim_type_full = 'static_gratings'
    elif stim_type == 'nm1':
        stim_type_full = 'natural_movie_one'
    elif stim_type == 'nm2':
        stim_type_full = 'natural_movie_two'
    elif stim_type == 'nm3':
        stim_type_full = 'natural_movie_three'

    areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = boc.get_all_imaging_depths()
    layers = [100, 200, 300, 500, 1000]
    results_array = np.zeros((len(areas), len(layers)-1, 2))

    if sys.platform == 'darwin':
        results_path = '/Volumes/Brain2017/decode_results'
    elif sys.platform == 'linux2':
        results_path = '/local1/Documents/projects/cam_analysis/decode_results'

    if type(meta) is str:
        results_path += '_'+meta

    results_loc_dict1, shuffle_loc_dict1 = make_results_loc_dict(stim_type=stim_type, stim_features=stim_features, plot_methods=plot_methods, relative=relative, error=error, results_path=results_path)
    results_keys = results_loc_dict1.keys()


    for a, area in enumerate(areas):
        if area == 'VISp':
            r_keys = [k for k in results_keys if ('VISp' in k) and ('VISpm' not in k)]
        else:
            r_keys = [k for k in results_keys if k[:len(area)] == area]

        for l, layer in enumerate(layers[:-1]):
            depth_range = [d for d in depths if (d >= layer) and (d < (layers[l+1]))]

            error_list1 = []
            shuffle_list1 = []

            r_keys1 = [r for r in r_keys if any(str(d) in r for d in depth_range)]

            if layer == 100: # layer 2/3
                r_keys1 = [r for r in r_keys1 if ~any(cre in r for cre in ('Scnn1a', 'Rorb', 'Nr5a1', 'Ntsr1'))]

            elif layer == 200: # layer 4
                r_keys1 = r_keys1 + [r for r in r_keys1 if any(cre in r for cre in ('Scnn1a', 'Rorb', 'Nr5a1'))]
                r_keys1 = [r for r in r_keys1 if 'Ntsr1' not in r]

            elif layer == 300:
                r_keys1 = [r for r in r_keys1 if ~any(cre in r for cre in ('Scnn1a', 'Rorb', 'Nr5a1', 'Ntsr1'))]

            elif layer == 500:
                r_keys1 = [r for r in r_keys1 if ~any(cre in r for cre in ('Scnn1a', 'Rorb', 'Nr5a1'))]
                r_keys1 = r_keys1 + [r for r in r_keys if 'Ntsr1' in r]

            r_keys1 = [r for r in r_keys1 if not any(e in r for e in exclude_keys)]

            for r in r_keys1:

                error_dict1 = results_loc_dict1[r][plot_methods[0]]
                shuffle_dict1 = shuffle_loc_dict1[r][plot_methods[0]]

                if len(error_dict1) > 0:
                    for k in range(len(error_dict1)):
                        for kk in range(len(error_dict1[k])):
                            key_temp = error_dict1[k].keys()[kk]
                            error_list1.append(error_dict1[k][key_temp].mean())
                            shuffle_list1.append(shuffle_dict1[k][key_temp].mean())

            error_list1 = np.array(error_list1)
            shuffle_list1 = np.array(shuffle_list1)
            if relative:
                if relative_divide:
                    error_list1 /= shuffle_list1
                else:
                    error_list1 -= shuffle_list1

            results_array[a, l, 0] = np.mean(error_list1[np.isfinite(error_list1)])
            results_array[a, l, 1] = len(error_list1)

    return results_array


def make_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors', 'ShuffledKNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False, relative=False, relative_divide=False, meta='L0events', exclude_keys=[]):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''


    results1 = make_results_loc_array(stim_type=stim_type, stim_features=stim_features, plot_methods=[plot_methods[0]], meta=meta, relative=relative, relative_divide=relative_divide)
    results2 = make_results_loc_array(stim_type=stim_type, stim_features=stim_features, plot_methods=[plot_methods[1]], meta=meta, relative=relative, relative_divide=relative_divide)
    results = results1.copy()
    results[:, :, 0] -= results2[:, :, 0]

    return results


def make_canonical_results_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', relative=False, relative_divide=False, meta='L0events', plot_key='decode_performance'):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''


    if stim_type == 'ns':
        stim_type_full = 'natural_scenes'
    elif stim_type == 'dg':
        stim_type_full = 'drifting_gratings'
    elif stim_type == 'sg':
        stim_type_full = 'static_gratings'
    elif stim_type == 'nm1':
        stim_type_full = 'natural_movie_one'
    elif stim_type == 'nm2':
        stim_type_full = 'natural_movie_two'
    elif stim_type == 'nm3':
        stim_type_full = 'natural_movie_three'

    areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = boc.get_all_imaging_depths()
    layers = [100, 200, 300, 500, 1000]
    results_array = np.zeros((len(areas), len(layers)-1, 2))

    if sys.platform == 'darwin':
        results_path = '/Volumes/Brain2017/decode_results'
    elif sys.platform == 'linux2':
        results_path = '/local1/Documents/projects/cam_analysis/decode_results'

    if type(meta) is str:
        results_path += '_'+meta

    results_df = make_results_dataframe(stim_type=stim_type, stim_features=stim_features, plot_methods=plot_methods, results_path=results_path)

    for a, area in enumerate(areas):
        area_results = results_df[results_df.area == area]

        layer_results = area_results[area_results.cre_depth == ('Cux2-CreERT2', 100)]
        results_tmp = layer_results[plot_key].values
        results_tmp = np.ma.MaskedArray(results_tmp, mask=~np.isfinite(results_tmp))

        results_array[a, 0, 0] = np.ma.mean(results_tmp)
        results_array[a, 0, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Rorb-IRES2-Cre', 200)]
        results_tmp = layer_results[plot_key].values
        results_tmp = np.ma.MaskedArray(results_tmp, mask=~np.isfinite(results_tmp))

        results_array[a, 1, 0] = np.ma.mean(results_tmp)
        results_array[a, 1, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Rbp4-Cre_KL100', 300)]
        results_tmp = layer_results[plot_key].values
        results_tmp = np.ma.MaskedArray(results_tmp, mask=~np.isfinite(results_tmp))

        results_array[a, 2, 0] = np.ma.mean(results_tmp)
        results_array[a, 2, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Ntsr1-Cre_GN220', 500)]
        results_tmp = layer_results[plot_key].values
        results_tmp = np.ma.MaskedArray(results_tmp, mask=~np.isfinite(results_tmp))

        results_array[a, 3, 0] = np.ma.mean(results_tmp)
        results_array[a, 3, 1] = len(layer_results)

    return results_array


def make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors', 'ShuffledKNeighbors'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False, relative=False, relative_divide=False, meta='L0events', exclude_keys=[]):
    '''

    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param results_path:
    :param error:
    :param mean_over_features:
    :param relative:
    :return: areas x layers x 2 array of median decoding result and number of experiments for each area/layer
    '''

    results1 = make_canonical_results_loc_array(stim_type=stim_type, stim_features=stim_features, plot_methods=[plot_methods[0]], meta=meta, relative=relative, relative_divide=relative_divide)
    results2 = make_canonical_results_loc_array(stim_type=stim_type, stim_features=stim_features, plot_methods=[plot_methods[1]], meta=meta, relative=relative, relative_divide=relative_divide)
    results = results1.copy()
    results[:, :, 0] -= results2[:, :, 0]

    return results


def make_canonical_results_loc_array_corrs(stim_type='ns', results_path='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone', compute_missing=False, plot_key='RhoSignalNoiseCorrs'):

    areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = boc.get_all_imaging_depths()
    layers = [100, 200, 300, 500, 1000]
    results_array = np.zeros((len(areas), len(layers)-1, 2))

    results_df = make_results_dataframe_corrs(stim_type=stim_type, results_path=results_path, plot_key=plot_key)

    for a, area in enumerate(areas):
        area_results = results_df[results_df.area == area]

        layer_results = area_results[area_results.cre_depth == ('Cux2-CreERT2', 100)]
        results_array[a, 0, 0] = np.nanmean(layer_results[plot_key].values)
        results_array[a, 0, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Rorb-IRES2-Cre', 200)]
        results_array[a, 1, 0] = np.nanmean(layer_results[plot_key].values)
        results_array[a, 1, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Rbp4-Cre_KL100', 300)]
        results_array[a, 2, 0] = np.nanmean(layer_results[plot_key].values)
        results_array[a, 2, 1] = len(layer_results)

        layer_results = area_results[area_results.cre_depth == ('Ntsr1-Cre_GN220', 500)]
        results_array[a, 3, 0] = np.nanmean(layer_results[plot_key].values)
        results_array[a, 3, 1] = len(layer_results)

    return results_array


def make_results_dataframe_corrs(stim_type='ns', plot_key='RhoSignalNoiseCorrs', df_file_head='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/plot_dataframes_corrs_decoding/results_dataframe_corrs', results_path='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''


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

    file_suffix = '_' + long_stim + '.pkl'

    try:
        results_df = pickle.load(open(df_file_head+plot_key+file_suffix))

    except:
        results_df = pd.DataFrame(columns=(plot_key, 'area', 'cre_depth'))

        exps = boc.get_ophys_experiments(stimuli=[long_stim])

        for n, exp in enumerate(exps):

            exp_id = exp['id']

            try:
                results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + file_suffix)))

            except:
                print('no results for exp '+str(exp_id))
                continue

            results_keys = results_file.keys()
            data_keys = []

            if plot_key in results_file.keys():

                tmp = pd.DataFrame(columns=(plot_key, 'area', 'cre_depth'), index=range(1))
                if plot_key in ['NoiseCorr', 'NoiseCorrStat', 'NoiseCorrRun', 'NoiseCorrDilate', 'NoiseCorrConstrict']:
                    tmp[plot_key] = np.nanmean(np.triu(results_file[plot_key].mean(axis=2), k=1 ))
                elif plot_key in ['SignalCorr', 'SignalCorrStat','SignalCorrRun', 'SignalCorrDilate', 'SignalCorrConstrict']:
                    tmp[plot_key] = np.nanmean(results_file[plot_key])
                else:
                    tmp[plot_key] = results_file[plot_key]

                im_depth = exp['imaging_depth']
                cre_line = exp['cre_line']

                if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                    depth = 200
                elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                    depth = 300
                elif cre_line == 'Ntsr1-Cre_GN220':
                    depth = 500
                else:
                    if im_depth >= 100 and im_depth < 200:
                        depth = 100
                    elif (im_depth >= 200 and im_depth < 300):
                        depth = 200
                    elif (im_depth >= 300 and im_depth < 500) :
                        depth = 300
                    elif im_depth >= 500:
                        depth = 500

                tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
                tmp['area'] = exp['targeted_structure']

                results_df = results_df.append(tmp, ignore_index=True)

            else:
                continue

        pickle.dump(results_df, file=open(df_file_head+plot_key+file_suffix, 'wb'))

    return results_df


def make_results_dataframe_signal_dim(stim_type='ns', plot_key='SignalCorr', df_file_head='/local1/Documents/projects/cam_analysis/results_dataframe_corrs', results_path='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/noise_signal_correlations/regressNone'):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''


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

    file_suffix = '_' + long_stim + '.pkl'

    results_df = pd.DataFrame(columns=(plot_key, 'area', 'cre_depth'))

    exps = boc.get_ophys_experiments(stimuli=[long_stim])

    for n, exp in enumerate(exps):

        exp_id = exp['id']

        try:
            results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + file_suffix)))

        except:
            print('no results for exp '+str(exp_id))
            continue

        results_keys = results_file.keys()
        data_keys = []

        if plot_key in results_file.keys():

            sig_corr_flat = results_file[plot_key]



            tmp = pd.DataFrame(columns=(plot_key, 'area', 'cre_depth'), index=range(1))
            tmp[plot_key] = results_file[plot_key]

            im_depth = exp['imaging_depth']
            cre_line = exp['cre_line']

            if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                depth = 200
            elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                depth = 300
            elif cre_line == 'Ntsr1-Cre_GN220':
                depth = 500
            else:
                if im_depth >= 100 and im_depth < 200:
                    depth = 100
                elif (im_depth >= 200 and im_depth < 300):
                    depth = 200
                elif (im_depth >= 300 and im_depth < 500) :
                    depth = 300
                elif im_depth >= 500:
                    depth = 500

            tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
            tmp['area'] = exp['targeted_structure']

            results_df = results_df.append(tmp, ignore_index=True)

        else:
            continue

    return results_df


def make_results_dataframe(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], df_file_head='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/plot_dataframes_corrs_decoding/results_dataframe_decode', results_path='/local1/Documents/projects/cam_analysis/decode_results_L0events', error='test', relative=True, relative_divide=True):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''

    if stim_type == 'ns':
        long_stim = 'natural_scenes'
    elif stim_type == 'dg':
        long_stim = 'drifting_gratings'
    elif stim_type == 'sg':
        long_stim = 'static_gratings'

    file_suffix = '_' + long_stim + '_' + plot_methods[0] + '.pkl'

    try:
        results_df = pickle.load(open(df_file_head+file_suffix))

    except:

        results_df = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'))

        exps = boc.get_ophys_experiments(stimuli=[long_stim])
        exp_ids = [exp['id'] for exp in exps]

        for n, exp in enumerate(exps):

            exp_id = exp['id']

            try:
                if error == 'test':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_error.pkl')))

                elif error == 'train':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_error.pkl')))

                elif error == 'test_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_dist.pkl')))

                elif error == 'train_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_dist.pkl')))

                elif error == 'num_neighbors':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_num_neighbors.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_num_neighbors.pkl')))

                else:
                    raise Exception('pick train or test error')

            except:
                print('no results for exp '+str(exp_id))
                continue

            results_keys = results_file.keys()
            data_keys = []

            for stim_feature in stim_features:
                for plot_method in plot_methods:
                    data_keys.append(stim_type + '_' + stim_feature + '_' + plot_method)

            data_keys = [d for d in data_keys if d in results_keys]

            for data_key in data_keys:
                tmp = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'), index=range(1))

                if relative:
                    if relative_divide:
                        tmp['decode_performance'] = np.mean(results_file[data_key]) / np.mean(shuffle_file[data_key])
                    else:
                        tmp['decode_performance'] = np.mean(results_file[data_key]) - np.mean(shuffle_file[data_key])
                else:
                    tmp['decode_performance'] = np.mean(results_file[data_key])


                im_depth = exp['imaging_depth']
                cre_line = exp['cre_line']

                if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                    depth = 200
                elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                    depth = 300
                elif cre_line == 'Ntsr1-Cre_GN220':
                    depth = 500
                else:
                    if im_depth >= 100 and im_depth < 200:
                        depth = 100
                    elif (im_depth >= 200 and im_depth < 300):
                        depth = 200
                    elif (im_depth >= 300 and im_depth < 500) :
                        depth = 300
                    elif im_depth >= 500:
                        depth = 500

                tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
                tmp['area'] = exp['targeted_structure']

                results_df = results_df.append(tmp, ignore_index=True)

        pickle.dump(results_df, open(df_file_head+file_suffix, 'wb'))

    return results_df


def make_results_dataframe_numNeurons(stim_type='dg', stim_features=['orientation'], plot_methods=['diagLDA'], df_file_head='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/plot_dataframes_corrs_decoding/results_dataframe_decode_numNeurons', results_path='/local1/Documents/projects/cam_analysis/decode_results_L0eventsNumNeurons', error='test', relative=True, relative_divide=True):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''

    if stim_type == 'ns':
        long_stim = 'natural_scenes'
    elif stim_type == 'dg':
        long_stim = 'drifting_gratings'
    elif stim_type == 'sg':
        long_stim = 'static_gratings'

    file_suffix = '_' + long_stim + '_' + plot_methods[0] + '.pkl'

    try:
        results_df = pickle.load(open(df_file_head+file_suffix))

    except:

        results_df = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'))
        exps = boc.get_ophys_experiments(stimuli=[long_stim])

        for n, exp in enumerate(exps):

            exp_id = exp['id']

            try:
                if error == 'test':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_error.pkl')))

                elif error == 'train':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_error.pkl')))

                elif error == 'test_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_dist.pkl')))

                elif error == 'train_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_dist.pkl')))

                elif error == 'num_neighbors':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_num_neighbors.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_num_neighbors.pkl')))

                else:
                    raise Exception('pick train or test error')

            except:
                print('no results for exp '+str(exp_id))
                continue

            results_keys = results_file.keys()
            data_keys = []

            for stim_feature in stim_features:
                for plot_method in plot_methods:
                    data_keys.append(stim_type + '_' + stim_feature + '_' + plot_method)

            data_keys = [d for d in data_keys if d in results_keys]

            for data_key in data_keys:

                results_tmp = results_file[data_key]
                Nrange = results_tmp.shape[0]
                tmp = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'), index=range(Nrange))

                if relative:
                    if relative_divide:
                        tmp['decode_performance'] = np.mean(results_file[data_key], axis=(1, 2)) / np.mean(shuffle_file[data_key], axis=(1, 2))
                    else:
                        tmp['decode_performance'] = np.mean(results_file[data_key], axis=(1, 2)) - np.mean(shuffle_file[data_key], axis=(1, 2))
                else:
                    tmp['decode_performance'] = np.mean(results_file[data_key], axis=(1, 2))


                im_depth = exp['imaging_depth']
                cre_line = exp['cre_line']

                if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                    depth = 200
                elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                    depth = 300
                elif cre_line == 'Ntsr1-Cre_GN220':
                    depth = 500
                else:
                    if im_depth >= 100 and im_depth < 200:
                        depth = 100
                    elif (im_depth >= 200 and im_depth < 300):
                        depth = 200
                    elif (im_depth >= 300 and im_depth < 500) :
                        depth = 300
                    elif im_depth >= 500:
                        depth = 500

                tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
                tmp['area'] = exp['targeted_structure']

                results_df = results_df.append(tmp, ignore_index=True)

        pickle.dump(results_df, open(df_file_head+file_suffix, 'wb'))

    return results_df


def make_confusion_dataframe(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], df_file_head='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/plot_dataframes_corrs_decoding/results_dataframe_decode', results_path='/local1/Documents/projects/cam_analysis/decode_results_L0events', error='test', relative=True, relative_divide=True):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''

    if stim_type == 'ns':
        long_stim = 'natural_scenes'
    elif stim_type == 'dg':
        long_stim = 'drifting_gratings'
    elif stim_type == 'sg':
        long_stim = 'static_gratings'

    file_suffix = '_' + long_stim + '_' + plot_methods[0] + '.pkl'

    try:
        results_df = pickle.load(open(df_file_head+file_suffix))

    except:

        results_df = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'))

        exps = boc.get_ophys_experiments(stimuli=[long_stim])
        exp_ids = [exp['id'] for exp in exps]

        for n, exp in enumerate(exps):

            exp_id = exp['id']

            try:
                if error == 'test':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_confusion.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_error.pkl')))

                elif error == 'train':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_confusion.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_error.pkl')))

                else:
                    raise Exception('pick train or test error')

            except:
                print('no results for exp '+str(exp_id))
                continue

            results_keys = results_file.keys()
            data_keys = []

            for stim_feature in stim_features:
                for plot_method in plot_methods:
                    data_keys.append(stim_type + '_' + stim_feature + '_' + plot_method)

            data_keys = [d for d in data_keys if d in results_keys]

            for data_key in data_keys:
                tmp = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'), index=range(1))

                if relative:
                    if relative_divide:
                        tmp['decode_performance'] = np.mean(results_file[data_key]) / np.mean(shuffle_file[data_key])
                    else:
                        tmp['decode_performance'] = np.mean(results_file[data_key]) - np.mean(shuffle_file[data_key])
                else:
                    tmp['decode_performance'] = np.mean(results_file[data_key])


                im_depth = exp['imaging_depth']
                cre_line = exp['cre_line']

                if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                    depth = 200
                elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                    depth = 300
                elif cre_line == 'Ntsr1-Cre_GN220':
                    depth = 500
                else:
                    if im_depth >= 100 and im_depth < 200:
                        depth = 100
                    elif (im_depth >= 200 and im_depth < 300):
                        depth = 200
                    elif (im_depth >= 300 and im_depth < 500) :
                        depth = 300
                    elif im_depth >= 500:
                        depth = 500

                tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
                tmp['area'] = exp['targeted_structure']

                results_df = results_df.append(tmp, ignore_index=True)

        pickle.dump(results_df, open(df_file_head+file_suffix, 'wb'))

    return results_df




def make_results_dataframe_compare_method(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors','ShuffledKNeighbors'], df_file_head='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/plot_dataframes_corrs_decoding/results_dataframe_decode', results_path='/local1/Documents/projects/cam_analysis/decode_results_L0events', error='test', mean_over_features=False, relative=True, relative_divide=True):

    '''
    make dataframe for metric_plot
    :param stim_type:
    :param stim_features:
    :param plot_methods:
    :param meta:
    :param relative:
    :param exclude_keys:
    :return:
    '''

    if stim_type == 'ns':
        long_stim = 'natural_scenes'
    elif stim_type == 'dg':
        long_stim = 'drifting_gratings'
    elif stim_type == 'sg':
        long_stim = 'static_gratings'

    file_suffix = '_' + long_stim + '_' + plot_methods[0] +'-'+ plot_methods[1] + '.pkl'

    try:
        results_df = pickle.load(open(df_file_head+file_suffix))

    except:

        results_df = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'))

        exps = boc.get_ophys_experiments(stimuli=[long_stim])
        exp_ids = [exp['id'] for exp in exps]

        for n, exp in enumerate(exps):

            exp_id = exp['id']

            try:
                if error == 'test':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_error.pkl')))

                elif error == 'train':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_error.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_error.pkl')))

                elif error == 'test_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_test_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_dist.pkl')))

                elif error == 'train_dist':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_train_dist.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_train_dist.pkl')))

                elif error == 'num_neighbors':
                    results_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_num_neighbors.pkl')))
                    shuffle_file = pickle.load(open(os.path.join(results_path, str(exp_id) + '_shuffle_num_neighbors.pkl')))

                else:
                    raise Exception('pick train or test error')

            except:
                print('no results for exp '+str(exp_id))
                continue

            results_keys = results_file.keys()
            data_keys1 = []
            data_keys2 = []

            for stim_feature in stim_features:
                data_keys1.append(stim_type + '_' + stim_feature + '_' + plot_methods[0])
                data_keys2.append(stim_type + '_' + stim_feature + '_' + plot_methods[1])

            for d, data_key1 in enumerate(data_keys1):
                data_key2 = data_keys2[d]

                if data_key1 in results_keys and data_key2 in results_keys:

                    tmp = pd.DataFrame(columns=('decode_performance', 'area', 'cre_depth'), index=range(1))

                    if relative:
                        if relative_divide:
                            performance_tmp1 = np.mean(results_file[data_key1]) / np.mean(shuffle_file[data_key1])
                            performance_tmp2 = np.mean(results_file[data_key2]) / np.mean(shuffle_file[data_key2])
                        else:
                            performance_tmp1 = np.mean(results_file[data_key1]) - np.mean(shuffle_file[data_key1])
                            performance_tmp2 = np.mean(results_file[data_key2]) - np.mean(shuffle_file[data_key2])
                    else:
                        performance_tmp1 = np.mean(results_file[data_key1])
                        performance_tmp2 = np.mean(results_file[data_key2])

                    tmp['decode_performance'] = performance_tmp1 - performance_tmp2

                    im_depth = exp['imaging_depth']
                    cre_line = exp['cre_line']

                    if cre_line in ['Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                        depth = 200
                    elif cre_line in ['Rbp4-Cre_KL100', 'Fezf2-CreER', 'Tlx3-Cre_PL56']:
                        depth = 300
                    elif cre_line == 'Ntsr1-Cre_GN220':
                        depth = 500
                    else:
                        if im_depth >= 100 and im_depth < 200:
                            depth = 100
                        elif (im_depth >= 200 and im_depth < 300):
                            depth = 200
                        elif (im_depth >= 300 and im_depth < 500):
                            depth = 300
                        elif im_depth >= 500:
                            depth = 500

                    tmp['cre_depth'].loc[0] = (exp['cre_line'], depth)
                    tmp['area'] = exp['targeted_structure']

                    results_df = results_df.append(tmp, ignore_index=True)

                else:
                    print('no results for exp ' + str(exp_id))
                    continue

        pickle.dump(results_df, open(df_file_head+file_suffix, 'wb'))


    return results_df




def main():


    # decoding plots

    results = make_canonical_results_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_KNN', scale=.015)

    # results = make_canonical_results_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors'], meta='L0events', relative=True, relative_divide=True)
    # pawplot.make_pawplot_population(results, filename='dg_ori_KNN_overChance', scale=.015)

    results = make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighborsRun','KNeighborsStat'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_KNNRun-KNNStat', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)

    results = make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['ShuffledKNeighborsRun','ShuffledKNeighborsStat'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_ShuffledKNNRun-ShuffledKNNStat', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0., clim=(-5, 5))


    results = make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighbors','ShuffledKNeighbors'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_KNN-ShuffledKNN', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)

    results = make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighborsRun','ShuffledKNeighborsRun'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_KNNRun-ShuffledKNNRun', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)

    results = make_canonical_results_diff_loc_array(stim_type='dg', stim_features=['orientation'], plot_methods=['KNeighborsStat','ShuffledKNeighborsStat'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='dg_ori_KNNStat-ShuffledKNNStat', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)



    results = make_canonical_results_loc_array(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighbors'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='ns_frame_KNN', scale=.015)

    results = make_canonical_results_diff_loc_array(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighborsRun','KNeighborsStat'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='ns_frame_KNNRun-KNNStat', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)

    results = make_canonical_results_diff_loc_array(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighbors','ShuffledKNeighbors'], meta='L0events', relative=True, relative_divide=True)
    pawplot.make_pawplot_population(results, filename='ns_frame_KNN-ShuffledKNN', scale=.015, symmetric_cmap=True, cmap='PuOr_r', cmid=0.)






    # strip plots
    areas = boc.get_all_targeted_structures()

    decode_df = make_results_dataframe()
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance (/ chance)', figname='dg_ori_KNN_cre_strip', bar=True, xlim=(0, 10), Nticks=5)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(plot_methods=['KNeighborsRun','KNeighborsStat'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Run - Stat (/ chance)', figname='dg_ori_KnnRun-KNNStat_cre_strip', bar=True, xlim=(-35, 18), Nticks=5)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(plot_methods=['KNeighborsDilate','KNeighborsConstrict'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Dilated - Constricted (/ chance)', figname='dg_ori_KnnDilate-KNNConstrict_cre_strip', bar=True, xlim=(-8, 16), Nticks=5)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(plot_methods=['KNeighbors','ShuffledKNeighbors'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Full - Shuffled Trials (/ chance)', figname='dg_ori_Knn-ShuffledKNN_cre_strip', bar=True, xlim=(-4, 3), Nticks=5)
    plt.close('all')


    decode_df = make_results_dataframe_compare_method(plot_methods=['KNeighborsStat','ShuffledKNeighborsStat'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Full - Shuffled Trials (/ chance)', figname='dg_ori_KnnStat-ShuffledKnnStat_cre_strip', bar=True, Nticks=5)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(plot_methods=['ShuffledKNeighborsRun','ShuffledKNeighborsStat'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Run - Stat (/ chance)', figname='dg_ori_ShuffledKnnRun-ShuffledKnnStat_cre_strip', bar=True, Nticks=5)
    plt.close('all')





    decode_df = make_results_dataframe(stim_type='ns', stim_features=['frame'])
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance (/ chance)', figname='ns_frame_KNN_cre_strip', bar=True)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighborsRun','KNeighborsStat'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Run - Stat (/ chance)', figname='ns_frame_KnnRun-KNNStat_cre_strip', bar=True)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighborsDilate','KNeighborsConstrict'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Dilated - Constricted (/ chance)', figname='ns_frame_KnnDilate-KNNConstrict_cre_strip', bar=True)
    plt.close('all')

    decode_df = make_results_dataframe_compare_method(stim_type='ns', stim_features=['frame'], plot_methods=['KNeighbors','ShuffledKNeighbors'], relative=True, relative_divide=True)
    for area in areas:
        strip_plot.plot_strip_plot(data_input=decode_df, area=area, plot_key='decode_performance', x_label='Decoding Performance \n Full - Shuffled Trials (/ chance)', figname='ns_frame_Knn-ShuffledKNN_cre_strip', bar=True)
    plt.close('all')




    # noise vs signal correlations

    results = make_canonical_results_loc_array_corrs(compute_missing=False, plot_key='RhoSignalNoiseCorrs')
    pawplot.make_pawplot_population(results, filename='ns_rho_noise_signal_corrs', scale=.015)

    results = make_canonical_results_loc_array_corrs(compute_missing=False, plot_key='DimensionFrac')
    pawplot.make_pawplot_population(results, filename='ns_dim', scale=.015)

    results = make_canonical_results_loc_array_corrs(compute_missing=False, plot_key='RhoSignalNoiseCorrsStat')
    pawplot.make_pawplot_population(results, filename='ns_rho_noise_signal_corrs_stat', scale=.015)

    results = make_canonical_results_loc_array_corrs(compute_missing=False, plot_key='DimensionFracStat')
    pawplot.make_pawplot_population(results, filename='ns_dim_stat', scale=.015)



    results = make_canonical_results_loc_array_corrs(stim_type='dg', compute_missing=False)
    pawplot.make_pawplot_population(results, filename='dg_rho_noise_signal_corrs', scale=.015, clim=(0, .5))

    results = make_canonical_results_loc_array_corrs(stim_type='dg', compute_missing=False, plot_key='DimensionFrac')
    pawplot.make_pawplot_population(results, filename='dg_dim', scale=.015)

    plt.close('all')



    results = make_canonical_results_loc_array_corrs(stim_type='ns', compute_missing=False, plot_key='NoiseCorr')
    pawplot.make_pawplot_population(results, filename='ns_noise_corr', scale=.015, clim=(0, .14))

    corr_df = make_results_dataframe_corrs(stim_type='ns', plot_key='NoiseCorr')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='NoiseCorr', x_label='Mean Noise Corr.', figname='ns_noise_corr_cre_strip', bar=True, xlim=(0, .6), Nticks=3)
    plt.close('all')

    results = make_canonical_results_loc_array_corrs(stim_type='ns', compute_missing=False, plot_key='NoiseCorrStat')
    pawplot.make_pawplot_population(results, filename='ns_noise_corr_stat', scale=.015, clim=(0, .2))

    corr_df = make_results_dataframe_corrs(stim_type='ns', plot_key='NoiseCorrStat')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='NoiseCorrStat', x_label='Mean Noise Corr.', figname='ns_noise_corr_stat_cre_strip', bar=True, xlim=(-1, 1))
    plt.close('all')



    results = make_canonical_results_loc_array_corrs(stim_type='ns', compute_missing=False, plot_key='SignalCorr')
    pawplot.make_pawplot_population(results, filename='dg_signal_corr', scale=.015, clim=(0, .5))

    corr_df = make_results_dataframe_corrs(stim_type='ns', plot_key='SignalCorr')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='SignalCorr', x_label='Mean Signal Corr.', figname='ns_signal_corr_cre_strip', bar=True, xlim=(-1, 1))
    plt.close('all')





    corr_df = make_results_dataframe_corrs(stim_type='ns')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='RhoSignalNoiseCorrs', x_label='Corr. Signal vs Noise Corrs', figname='ns_rho_noise_signal_corrs_cre_strip', bar=True, xlim=(-1, 1))
    plt.close('all')

    dim_df = make_results_dataframe_corrs(stim_type='ns', plot_key='DimensionFrac')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=dim_df, area=area, plot_key='DimensionFrac', x_label='Dimensionality / Num. Neurons', figname='ns_dim_cre_strip', bar=True, xlim=(0, 1))
    plt.close('all')


    corr_df = make_results_dataframe_corrs(stim_type='ns', plot_key='RhoSignalNoiseCorrsStat')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='RhoSignalNoiseCorrsStat', x_label='Corr. Signal vs Noise Corrs', figname='ns_rho_noise_signal_corrs_stat_cre_strip', bar=True, xlim=(-1, 1))
    plt.close('all')

    dim_df = make_results_dataframe_corrs(stim_type='ns', plot_key='DimensionFracStat')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=dim_df, area=area, plot_key='DimensionFracStat', x_label='Dimensionality / Num. Neurons', figname='ns_dim_stat_cre_strip', bar=True, xlim=(0, 1))
    plt.close('all')


    corr_df = make_results_dataframe_corrs(stim_type='dg')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=corr_df, area=area, plot_key='RhoSignalNoiseCorrs', x_label='Corr. Signal vs Noise Corrs', figname='dg_rho_noise_signal_corrs_cre_strip', bar=True)
    plt.close('all')

    dim_df = make_results_dataframe_corrs(stim_type='dg', plot_key='DimensionFrac')
    for area in areas:
        strip_plot.plot_strip_plot(data_input=dim_df, area=area, plot_key='DimensionFrac', x_label='Dimensionality / Num. Neurons', figname='dg_dim_cre_strip', bar=True)
    plt.close('all')





if __name__=='__main__':
    main()