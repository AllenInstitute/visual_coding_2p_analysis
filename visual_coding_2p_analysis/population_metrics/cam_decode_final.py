# -*- coding: utf-8 -*-

import os, sys, time, pickle, argparse
from visual_coding_2p_analysis import core
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    manifest_file = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache/brain_observatory_manifest.json'

elif sys.platform == 'linux2':
    # manifest_file = '/allen/aibs/mat/gkocker/BrainObservatory/boc/manifest.json'
    manifest_file = core.get_manifest_path()

boc = BrainObservatoryCache(manifest_file=manifest_file)

from visual_coding_2p_analysis.l0_analysis import L0_analysis

import numpy as np
import matplotlib.pyplot as plt
import gaussClassifier as gc
import knnClassifier as knnC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.mixture as mixture
import h5py

def circdist(ori1, ori2):
    '''
    calculate shortest distance around a circle between two orientations
    '''

    ori_diff = np.abs(np.fmod(ori1 - ori2, 360.))
    if ori_diff >= 180:
        ori_diff = 360. - ori_diff

    return ori_diff


def calc_distance_error(predictions, labels, stim_class, stim_category, stim_template=None):

    error_val = 0.
    if (stim_class == 'ns') or (stim_class == 'natural_scenes'):

        if stim_template is None:
            raise Exception('need stim template for natural scenes')

        for i, n in enumerate(predictions):
            error_val += np.sqrt(np.mean((stim_template[n] - stim_template[labels[i]])**2))
        error_val /= float(len(predictions))

    elif ('nm' in stim_class) or ('natural_movie' in stim_class):
        error_val = np.sqrt(np.mean( (predictions - labels)**2)) # rmse in frames

    elif (stim_class == 'sg') or (stim_class == 'static_gratings'):
        if (stim_category == 'orientation') or (stim_category == 'spatial_frequency'):
            count = 1
            for i, n in enumerate(predictions):
                if (n != '-1') and (labels[i] != '-1'): # discard blank
                    error_val += np.abs(float(n) - float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'all': # orientation, spatial frequency in that order for each stim

            count = 1
            for i, stim_predict in enumerate(predictions):
                if (stim_predict != '-1') and (labels[i] != '-1'): # discard blank

                    stim_predict = stim_predict[1:-1].split(' ')
                    stim_predict = [x for x in stim_predict if x != '']

                    stim_predict0 = float(stim_predict[0])
                    stim_predict1 = float(stim_predict[1])

                    label = labels[i][1:-1].split(' ')
                    label = [x for x in label if x != '']

                    label0 = float(label[0])
                    label1 = float(label[1])

                    error_val += np.sqrt( (stim_predict0 - label0)**2 + (stim_predict1 - label1)**2 )
                    count += 1

            error_val /= float(count)

    elif (stim_class == 'dg') or (stim_class == 'drifting_gratings'):

        if stim_category == 'orientation':
            count = 1
            for i, n in enumerate(predictions):
                if (n != '-1') and (labels[i] != '-1'): # discard blank
                    error_val += circdist(float(n), float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'temporal_frequency':
            count = 1
            for i, n in enumerate(predictions):
                if (n != -1) and (labels[i] != -1): # discard blank
                    error_val += np.abs(float(n) - float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'all':

            count = 1
            for i, stim_predict in enumerate(predictions):
                if (stim_predict != '-1') and (labels[i] != '-1'): # discard blank

                    stim_predict = stim_predict[1:-1].split(' ')
                    stim_predict = [x for x in stim_predict if x != '']

                    stim_predict0 = float(stim_predict[0])
                    stim_predict1 = float(stim_predict[1])

                    label = labels[i][1:-1].split(' ')
                    label = [x for x in label if x != '']

                    label0 = float(label[0])
                    label1 = float(label[1])

                    error_val += np.sqrt( (stim_predict0 - label0)**2 + (stim_predict1 - label1)**2 )
                    count += 1

            error_val /= float(count)

    else:
        raise Exception('No relative error defined for stim class'+str(stim_class))

    return error_val


def get_response_table(dff, stim_table, stim, log=False, shift=0, width=None):

    stim_lengths = stim_table.end.values - stim_table.start.values
    Nstim = len(stim_table)
    Ncells = dff.shape[0]

    if width is None:  # use stim length - this should correspond to the SDK sweep response

        if stim == 'ns' or stim == 'natural_scenes':
            ind_start = shift
            ind_end = ind_start + np.amin(stim_lengths) + 7
        else:
            min_length = max([1, np.amin(stim_lengths)])
            ind_start = shift
            ind_end = ind_start + min_length

    else: # use pre-defined window width

        ind_start = shift
        ind_end = ind_start + width


    ''' get response array '''
    response_array = np.zeros((Ncells, Nstim))
    for i in range(Nstim):
        response_array[:, i] += np.mean(dff[:, stim_table.start.values[i]+ind_start:stim_table.start.values[i]+ind_end], axis=1)

    if log:
        response_array += 1. - np.amin(response_array)
        response_array = np.log(response_array)


    return response_array.T  # stim x cells


def get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, nan_ind=None, log=False, shift=0, width=14):
    '''
    get array of responses and running speeds throughout experiment
    '''


    N, T = dff.shape

    if master_stim_table is not None:
        start = master_stim_table['start'][0]
    else:
        start = 1000

    T -= start

    width = int(width)
    Nstim = np.floor(np.float(T) / np.float(width)).astype('int')

    response_array = np.zeros((Nstim, N))
    running_array = np.zeros((Nstim))

    dxcm = np.ma.masked_array(dxcm, mask=np.isnan(dxcm)).reshape(-1, 1)

    for i in range(Nstim):
        response_array[i] = np.mean(dff[:, start+(i*width):start+(i+1)*width], axis=1)
        running_array[i] = np.ma.mean(dxcm[start+(i*width):start+(i+1)*width])

    if pupil_size is not None and nan_ind is not None:

        # pupil_size = np.ma.masked_array(pupil_size, mask=np.isnan(pupil_size))
        pupil_array = np.zeros((Nstim))
        bad_ind = []

        for i in range(Nstim):
            # pupil_array[i] = np.ma.mean(pupil_size[start+(i*width):start+(i+1)*width])
            ind = range(start+i*width, start+(i+1)*width)
            if not np.all([j in nan_ind for j in ind]):
                pupil_array[i] = np.mean(pupil_size[ind])
            else:
                bad_ind.append(i)

        response_array = np.delete(response_array, bad_ind, 0)
        running_array = np.delete(running_array, bad_ind, 0)
        pupil_array = np.delete(pupil_array, bad_ind, 0)

        return response_array, running_array, pupil_array

    else:
        return response_array, running_array, None


def get_running_table(dxcm, stim_table, stim, shift=0, width=None):

    stim_lengths = stim_table.end.values - stim_table.start.values
    Nstim = len(stim_table)

    dxcm = np.ma.masked_array(dxcm, mask=np.isnan(dxcm))

    if width is None:  # use stim length - this should correspond to the SDK sweep response

        if stim == 'ns' or stim == 'natural_scenes':
            ind_start = shift
            ind_end = ind_start + np.amin(stim_lengths) + 7
        else:
            min_length = max([1, np.amin(stim_lengths)])
            ind_start = shift
            ind_end = ind_start + min_length

    else: # use pre-defined window width
        ind_start = shift
        ind_end = ind_start + width

    running_array = np.zeros((Nstim))

    for i in range(Nstim):
        start = stim_table.start.values[i] + ind_start
        end = stim_table.start.values[i] + ind_end
        running_array[i] = np.ma.mean(dxcm[start:end])

    return running_array


def interpolate_pupil_size(pupil_size):

    nan_mask = np.isnan(pupil_size)
    nan_ind = np.where(nan_mask)[0]
    good_ind = np.where(~nan_mask)[0]

    pupil_size[nan_ind] = np.interp(nan_ind, good_ind, pupil_size[good_ind])

    return pupil_size, nan_ind


def label_run_stationary_dpgmm(running_array, alpha_par=.2):
    '''

    :param running_array: array of running speeds
    :param alpha_par: concentration parameter for dirichlet process, default good for speeds averaged over 7 frames
    :return:
    '''



    good_ind = np.isfinite(running_array)
    # X = np.array(running_array[good_ind]).reshape(-1, 1)
    X = np.array(running_array[good_ind]).reshape(-1, 1)

    dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=alpha_par, covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)
    dpgmm.fit(X)
    Y = dpgmm.predict(X)
    means = dpgmm.means_
    vars = dpgmm.covariances_

    labels = -1*np.ones(running_array.shape, dtype=np.int)
    labels[good_ind] = Y
    # labels = Y.reshape(running_array.shape)

    return labels, means, vars


def nested_cross_val_LDA_factors(response, stims, num_folds=5, plot=False, shrinkage='ledoit-wolf'):
    ''' nested cross-validation for number of factors to use for LDA with predicted covariances '''

    Nstim, N = response.shape
    num_factors_range = range(1, 10)

    lda_train_scores = np.zeros((len(num_factors_range), num_trials, num_folds))
    lda_validation_scores = np.zeros((len(num_factors_range), num_trials, num_folds))

    # best_model_num_factors = np.zeros((num_trials, num_folds))
    # lda_best_model_test_scores = np.zeros((num_trials, num_folds))
    # lda_best_model_train_scores = np.zeros((num_trials, num_folds))

    # for i in range(num_trials):
        # skf_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=i)

    skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # outer_fold = 0
    # for selection, test in skf_outer.split(response, stims):

    train_scores_temp = np.zeros((len(num_factors_range), num_folds))
    validation_scores_temp = np.zeros((len(num_factors_range), num_folds))

    for n, num_fact in enumerate(num_factors_range):

        inner_fold = 0
        for train, validation in skf_inner.split(response, stims):
            rTrain = response[train]
            cTrain = stims[train]
            rValid = response[validation]
            cValid = stims[validation]

            lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0, shrinkage=shrinkage)
            lda.fit(rTrain)
            train_predictions = lda.predict(rTrain)
            train_scores_temp[n, inner_fold] = np.sum((train_predictions == cTrain)) / float(len(train))

            lda.fit(rValid)
            validation_predictions = lda.predict(rValid)
            validation_scores_temp[n, inner_fold] = np.sum((validation_predictions == cValid)) / float(
                len(validation))

            inner_fold += 1

        lda_train_scores[n] = train_scores_temp[n].mean()
        lda_validation_scores[n] = validation_scores_temp[n].mean()

    mean_validation_scores = validation_scores_temp.mean(axis=1)
    ind_best = np.where(mean_validation_scores == max(mean_validation_scores))[0][0]
    # best_model_num_factors[i, outer_fold] = num_factors_range[ind_best]
    best_model_num_factors = num_factors_range[ind_best]

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[1].errorbar(num_factors_range, lda_validation_scores.mean(axis=2), yerr=lda_validation_scores.std(axis=2)/float(num_folds), label='validation')
        ax[0].errorbar(num_factors_range, lda_train_scores.mean(axis=2), yerr=lda_train_scores.std(axis=2)/float(num_folds), label='train')
        for i in range(num_folds):
            ax[0].plot(num_factors_range, lda_train_scores[:, 0, i], 'k', alpha=0.2)
            ax[1].plot(num_factors_range, lda_validation_scores[:, 0, i], 'k', alpha=0.2)

        # ax.legend(loc=0)
        # ax.set_xlabel('Number of factors')
        # ax.set_ylabel('Accuracy')
        ax[0].set_title('Train')
        ax[1].set_title('Validation')
        fig.text(.5, .05, 'Number of factors', ha='center')
        fig.text(.05, .5, 'Accuracy', va='center', rotation='vertical')


    return best_model_num_factors


def nested_cross_val_KNeighbors(response, stims, num_folds=3, plot=False):
    ''' nested cross-validation for number of factors to use for LDA with predicted covariances '''

    # skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True)
    skf_inner = KFold(n_splits=num_folds, shuffle=True)

    Nstim, N = response.shape
    # num_neighbors_range = range(5, min(100, Nstim*(num_folds-1)).astype('int'), 20)

    max_num_neighbors_fold = []
    for train, validation in skf_inner.split(response, stims):
        max_num_neighbors_fold.append(len(train))

    max_num_neighbors_fold = np.unique(max_num_neighbors_fold)

    # num_neighbors_range = range(5, min(max_num_neighbors_fold), 20)
    # num_neighbors_range = np.round(np.logspace(0., 2., num=8)).astype('int')
    num_neighbors_range = [1] + range(5, 100, 10)

    if max(num_neighbors_range) > min(max_num_neighbors_fold):
        ind = np.where(num_neighbors_range > min(max_num_neighbors_fold))[0][0]
        num_neighbors_range = num_neighbors_range[:ind]

    # print 'testing ' + str(len(num_neighbors_range)) + ' possible num_neighbors'
    train_scores = np.zeros((len(num_neighbors_range), num_folds))
    validation_scores = np.zeros((len(num_neighbors_range), num_folds))

    for n, num in enumerate(num_neighbors_range):

        # knn = KNeighborsClassifier(n_neighbors=num, weights='uniform', algorithm='ball_tree')

        inner_fold = 0
        for train, validation in skf_inner.split(response, stims):
            rTrain = response[train]
            cTrain = stims[train]
            rValid = response[validation]
            cValid = stims[validation]

            knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num)
            knn.fit(rTrain, rTrain)
            train_predictions = knn.predict()

            knn = knnC.KNN(rTrain, cTrain, rValid, num_neighbors=num)
            knn.fit(rValid, rTrain)
            validation_predictions = knn.predict()

            train_scores[n, inner_fold] = np.sum((train_predictions == cTrain)) / float(len(train))
            validation_scores[n, inner_fold] = np.sum((validation_predictions == cValid)) / float(
                len(validation))

            inner_fold += 1


    mean_validation_scores = validation_scores.mean(axis=1)
    ind_best = np.where(mean_validation_scores == max(mean_validation_scores))[0][0]
    best_num_neighbors = num_neighbors_range[ind_best]

    return best_num_neighbors


def make_results_loc_dict(stim_type='ns', stim_features=['frame'], plot_methods=['NaiveBayes', 'LDA'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', mean_over_features=False):

    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    error_dict = dict()
    shuffle_dict = dict()

    for area in areas:
        for depth in depths:
            for cre_line in cre_lines:

                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth], cre_lines=[cre_line], simple=False)
                if len(ecs) == 0:
                    print('no experiments for '+area+str(depth)+cre_line)
                else:
                    ec_ids = [ec['id'] for ec in ecs]
                    exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids, simple=False)
                    exp_ids = [exp['id'] for exp in exps]

                    error_dict[area+str(depth)+cre_line] = dict()
                    shuffle_dict[area+str(depth)+cre_line] = dict()

                    for method in plot_methods:
                        if mean_over_features:
                            error_dict[area+str(depth)+cre_line][method] = [0 for x in range(len(exp_ids))]
                            shuffle_dict[area+str(depth)+cre_line][method] = [0 for x in range(len(exp_ids))]

                        else:
                            error_dict[area+str(depth)+cre_line][method] = []
                            shuffle_dict[area + str(depth) + cre_line][method] = []

                    for x, exp in enumerate(exp_ids):

                        try:
                            if error == 'test':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_test_error.pkl')))
                                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_error.pkl')))

                            elif error == 'train':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_train_error.pkl')))
                                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_train_error.pkl')))

                            elif error == 'test_dist':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_test_dist.pkl')))
                                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_dist.pkl')))

                            elif error == 'train_dist':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_train_dist.pkl')))
                                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_train_dist.pkl')))

                            elif error == 'num_neighbors':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_num_neighbors.pkl')))
                                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_num_neighbors.pkl')))

                            else:
                                raise Exception('pick train or test error')

                            results_keys_all = results_file.keys()
                            results_keys = []
                            # filter by stim type
                            for r in results_keys_all:

                                break_inds = [i for i, xx in enumerate(r) if xx == '_']
                                if r[:break_inds[0]] == stim_type:
                                    results_keys.append(r)

                            # results_keys = [r for r in results_keys if r[:2] == stim_type]

                            for r in results_keys:

                                break_inds = [i for i, xx in enumerate(r) if xx == '_']
                                r_method = r[break_inds[-1]+1:]
                                r_feature = r[break_inds[0]+1:break_inds[-1]]

                                if r_method in plot_methods and r_feature in stim_features:

                                    if 'Run' in r: # only do experiments that have both running and stationary
                                        ind = r.index('Run')
                                        if not r[:ind] + 'Stat' in results_keys:
                                            continue
                                    elif ('Stat' in r) and ('State' not in r):
                                        ind = r.index('Stat')
                                        if not r[:ind] + 'Run' in results_keys:
                                            continue

                                    if 'Constrict' in r: # only do experiments that have both running and stationary
                                        ind = r.index('Constrict')
                                        if not r[:ind] + 'Dilate' in results_keys:
                                            continue
                                    elif ('Dilate' in r):
                                        ind = r.index('Dilate')
                                        if not r[:ind] + 'Constrict' in results_keys:
                                            continue



                                    error_store = dict()
                                    error_store[r_feature] = results_file[r] #np.mean(results_file[r])
                                    shuffle_store = dict()
                                    shuffle_store[r_feature] = shuffle_file[r] #np.mean(shuffle_file[r])

                                    error_dict[area + str(depth) + cre_line][r_method].append( error_store)
                                    shuffle_dict[area + str(depth) + cre_line][r_method].append( shuffle_store)

                        except:
                            print('no results for exp '+str(exp))
                            continue

    return error_dict, shuffle_dict


def make_confusion_loc_dict(stim_type='ns', stim_features=['frame'], plot_methods=['NaiveBayes', 'LDA'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test'):

    areas = boc.get_all_targeted_structures()
    depths = boc.get_all_imaging_depths()
    cre_lines = boc.get_all_cre_lines()

    confusion_dict = dict()

    for area in areas:
        for depth in depths:
            for cre_line in cre_lines:

                ecs = boc.get_experiment_containers(targeted_structures=[area], imaging_depths=[depth], cre_lines=[cre_line], simple=False)
                if len(ecs) == 0:
                    print 'no experiments for '+area+str(depth)+cre_line
                else:
                    ec_ids = [ec['id'] for ec in ecs]
                    exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids, simple=False)
                    exp_ids = [exp['id'] for exp in exps]

                    confusion_dict[area+str(depth)+cre_line] = dict()

                    for method in plot_methods:
                        confusion_dict[area+str(depth)+cre_line][method] = []

                    for x, exp in enumerate(exp_ids):

                        try:
                            if error == 'test':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_test_confusion.pkl')))

                            elif error == 'train':
                                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_train_confusion.pkl')))

                            else:
                                raise Exception('pick train or test error')

                            results_keys = results_file.keys()
                            results_keys = [r for r in results_keys if r[:2] == stim_type]

                            # for method in plot_methods:
                            for r in results_keys:

                                break_inds = [i for i, xx in enumerate(r) if xx == '_']
                                r_method = r[break_inds[-1]+1:]
                                r_feature = r[break_inds[0]+1:break_inds[-1]]

                                if r_method in plot_methods and r_feature in stim_features:

                                    if 'Run' in r: # only do experiments that have both running and stationary
                                        ind = r.index('Run')
                                        if not r[:ind] + 'Stat' in results_keys:
                                            continue
                                    elif 'Stat' in r:
                                        ind = r.index('Stat')
                                        if not r[:ind] + 'Run' in results_keys:
                                            continue

                                    error_store = dict()
                                    error_store[r_feature] = results_file[r] #np.mean(results_file[r])
                                    confusion_dict[area + str(depth) + cre_line][r_method].append( error_store)

                        except:
                            print 'no results for exp '+str(exp)
                            continue

    return confusion_dict


def make_results_expID_dict(stim_type='ns', stim_features=['frame'], plot_methods=['NaiveBayes', 'LDA'], results_path='/local1/Documents/projects/cam_analysis/decode_results', error='test', relative=False):

    error_dict = dict()
    shuffle_dict = dict()

    exps = boc.get_ophys_experiments()
    exp_ids = [exp['id'] for exp in exps]


    for x, exp in enumerate(exp_ids):

        try:
            if error == 'test':
                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_test_error.pkl')))
                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_error.pkl')))

            elif error == 'train':
                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_train_error.pkl')))
                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_train_error.pkl')))

            elif error == 'test_dist':
                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_test_dist.pkl')))
                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_dist.pkl')))

            elif error == 'train_dist':
                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_train_dist.pkl')))
                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_train_dist.pkl')))

            elif error == 'num_neighbors':
                results_file = pickle.load(open(os.path.join(results_path, str(exp)+'_num_neighbors.pkl')))
                shuffle_file = pickle.load(open(os.path.join(results_path, str(exp)+'_shuffle_num_neighbors.pkl')))

            else:
                raise Exception('pick train or test error')

            results_keys_all = results_file.keys()
            results_keys = []
            # filter by stim type
            for r in results_keys_all:

                break_inds = [i for i, xx in enumerate(r) if xx == '_']
                if r[:break_inds[0]] == stim_type:
                    results_keys.append(r)

            # results_keys = [r for r in results_keys if r[:2] == stim_type]

            for r in results_keys:

                break_inds = [i for i, xx in enumerate(r) if xx == '_']
                r_method = r[break_inds[-1]+1:]
                r_feature = r[break_inds[0]+1:break_inds[-1]]

                if r_method in plot_methods and r_feature in stim_features:

                    if 'Run' in r: # only do experiments that have both running and stationary
                        ind = r.index('Run')
                        if not r[:ind] + 'Stat' in results_keys:
                            continue
                    elif ('Stat' in r) and ('State' not in r):
                        ind = r.index('Stat')
                        if not r[:ind] + 'Run' in results_keys:
                            continue

                    if 'Constrict' in r: # only do experiments that have both running and stationary
                        ind = r.index('Constrict')
                        if not r[:ind] + 'Dilate' in results_keys:
                            continue
                    elif ('Dilate' in r):
                        ind = r.index('Dilate')
                        if not r[:ind] + 'Constrict' in results_keys:
                            continue

                    if relative:
                        error_dict[str(exp)] = results_file[r].mean() / shuffle_file[r].mean()
                        shuffle_dict[str(exp)] = shuffle_file[r]

                    else:
                        error_dict[str(exp)] = results_file[r]
                        shuffle_dict[str(exp)] = shuffle_file[r]

        except:
            print('no results for exp '+str(exp))
            continue

    return error_dict, shuffle_dict


def shuffle_trials_within_class(response, classes):
    '''
    shuffle trials across neurons within the same class to decorrelate them
    :param response: trials x neurons
    :param classes: trials
    :return:
    '''

    T, N = response.shape
    response_new = np.zeros((T, N))

    for i, c in enumerate(np.unique(classes)):

        ind = (classes == c)
        for n in range(N):
            response_new[ind, n] = np.random.permutation(response[ind, n])

    return response_new


def decode_drifting_grating_cross(response_full, stim_table_full, stim_template, method='KNeighbors', num_folds=5, standardize=False):

    '''
    decode drifting gratings, ori for each tf and vice versa
    :param response:
    :param stim_table:
    :param method:
    :return:
    '''


    stim_class = 'dg'
    stim_categories = ['orientation','temporal_frequency']

    skf = KFold(n_splits=num_folds, shuffle=False)

    test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
        train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict = dict(), dict(), dict(), \
                                                                                                dict(), dict(), dict(), \
                                                                                                dict(), dict()


    for stim_category in stim_categories:
        if stim_category == 'orientation':
            other_category = 'temporal_frequency'
        else:
            other_category = 'orientation'

        other_stims = np.array(stim_table_full[other_category])
        other_stims = np.unique(other_stims[np.isfinite(other_stims)])
        Nother = len(other_stims)

        test_scores = np.zeros((Nother, num_folds))
        train_scores = np.zeros((Nother, num_folds))
        train_shuffle_scores = np.zeros((Nother, num_folds))
        test_shuffle_scores = np.zeros((Nother, num_folds))

        test_scores_dist = np.zeros((Nother, num_folds))
        train_scores_dist = np.zeros((Nother, num_folds))
        train_shuffle_scores_dist = np.zeros((Nother, num_folds))
        test_shuffle_scores_dist = np.zeros((Nother, num_folds))

        for n_other, other_stim in enumerate(other_stims):

            other_stim_ind = (stim_table_full[other_category].values == other_stim)

            # stim_table = stim_table_full[other_stim_ind]
            response_calc = response_full[other_stim_ind]
            stims_calc = stim_table_full[stim_category][other_stim_ind].values
            stims_shuffle_calc = np.random.permutation(stims_calc)


            # skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0
            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                if 'Shuffled' in method:
                    rTest = shuffle_trials_within_class(rTest, cTest)

                if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or (
                    'ShuffledLDA' in method):
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif 'bootstrapLDA' in method:

                    lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate',
                                'LDAFactorConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate',
                                'NaiveBayesConstrict']:
                    nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
                    nb.fit(rTrain)
                    train_predictions = nb.predict(rTrain)

                    nb.fit(rTest)
                    test_predictions = nb.predict(rTest)

                elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
                    gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'GDAFactor':

                    gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'LDABestFactor':

                    num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds, plot=False,
                                                            shrinkage=None)
                    lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)
                    num_factors_shuffle[fold] = num_fact

                elif 'KNeighbors' in method:

                    num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds, plot=False)
                    # num_neighb = len(np.unique(cTrain))
                    # num_neighb = 1
                    num_neighbors[fold] = num_neighb

                    knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
                    knn.fit(rTrain, rTrain)
                    train_predictions = knn.predict()

                    knn = knnC.KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
                    knn.fit(rTest, rTrain)
                    test_predictions = knn.predict()

                train_scores_dist[n_other, fold] = calc_distance_error(predictions=train_predictions, labels=cTrain,
                                                              stim_class=stim_class, stim_category=stim_category,
                                                              stim_template=stim_template)
                test_scores_dist[n_other, fold] = calc_distance_error(predictions=test_predictions, labels=cTest,
                                                             stim_class=stim_class, stim_category=stim_category,
                                                             stim_template=stim_template)

                train_scores[n_other, fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[n_other, fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                if 'Shuffled' in method:
                    rTest = shuffle_trials_within_class(rTest, cTest)

                if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or (
                    'ShuffledLDA' in method):
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif 'bootstrapLDA' in method:

                    lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate',
                                'LDAFactorConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate',
                                'NaiveBayesConstrict']:
                    nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
                    nb.fit(rTrain)
                    train_predictions = nb.predict(rTrain)

                    nb.fit(rTest)
                    test_predictions = nb.predict(rTest)

                elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
                    gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'GDAFactor':

                    gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'LDABestFactor':

                    num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds, plot=False,
                                                            shrinkage=None)
                    lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)
                    num_factors_shuffle[fold] = num_fact

                elif 'KNeighbors' in method:

                    num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds, plot=False)
                    # num_neighb = len(np.unique(cTrain))
                    # num_neighb = 1
                    num_neighbors[fold] = num_neighb

                    knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
                    knn.fit(rTrain, rTrain)
                    train_predictions = knn.predict()

                    knn = knnC.KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
                    knn.fit(rTest, rTrain)
                    test_predictions = knn.predict()

                train_shuffle_scores_dist[n_other, fold] = calc_distance_error(predictions=train_predictions, labels=cTrain,
                                                                      stim_class=stim_class,
                                                                      stim_category=stim_category,
                                                                      stim_template=stim_template)
                test_shuffle_scores_dist[n_other, fold] = calc_distance_error(predictions=test_predictions, labels=cTest,
                                                                     stim_class=stim_class,
                                                                     stim_category=stim_category,
                                                                     stim_template=stim_template)

                train_shuffle_scores[n_other, fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[n_other, fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

        test_scores_dict[stim_category] = test_scores
        train_scores_dict[stim_category] = train_scores
        test_shuffle_scores_dict[stim_category] = test_shuffle_scores
        train_shuffle_scores_dict[stim_category] = train_shuffle_scores

        test_scores_dist_dict[stim_category] = test_scores_dist
        train_scores_dist_dict[stim_category] = train_scores_dist
        test_shuffle_scores_dist_dict[stim_category] = test_shuffle_scores_dist
        train_shuffle_scores_dist_dict[stim_category] = train_shuffle_scores_dist


    return test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
        train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict


def decode_static_grating_cross(response_full, stim_table_full, stim_template, method='KNeighbors', num_folds=5, standardize=False):

    '''
    decode drifting gratings, ori for each tf and vice versa
    :param response:
    :param stim_table:
    :param method:
    :return:
    '''


    stim_class = 'sg'
    stim_categories = ['orientation','spatial_frequency']

    skf = KFold(n_splits=num_folds, shuffle=False)

    test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
        train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict = dict(), dict(), dict(), \
                                                                                                dict(), dict(), dict(), \
                                                                                                dict(), dict()


    for stim_category in stim_categories:
        if stim_category == 'orientation':
            other_category = 'spatial_frequency'
        else:
            other_category = 'orientation'

        other_stims = np.array(stim_table_full[other_category])
        other_stims = np.unique(other_stims[np.isfinite(other_stims)])
        Nother = len(other_stims)

        test_scores = np.zeros((Nother, num_folds))
        train_scores = np.zeros((Nother, num_folds))
        train_shuffle_scores = np.zeros((Nother, num_folds))
        test_shuffle_scores = np.zeros((Nother, num_folds))

        test_scores_dist = np.zeros((Nother, num_folds))
        train_scores_dist = np.zeros((Nother, num_folds))
        train_shuffle_scores_dist = np.zeros((Nother, num_folds))
        test_shuffle_scores_dist = np.zeros((Nother, num_folds))

        for n_other, other_stim in enumerate(other_stims):

            other_stim_ind = (stim_table_full[other_category].values == other_stim)

            # stim_table = stim_table_full[other_stim_ind]
            response_calc = response_full[other_stim_ind]
            stims_calc = stim_table_full[stim_category][other_stim_ind].values
            stims_shuffle_calc = np.random.permutation(stims_calc)


            # skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0
            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                if 'Shuffled' in method:
                    rTest = shuffle_trials_within_class(rTest, cTest)

                if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or (
                    'ShuffledLDA' in method):
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif 'bootstrapLDA' in method:

                    lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate',
                                'LDAFactorConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate',
                                'NaiveBayesConstrict']:
                    nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
                    nb.fit(rTrain)
                    train_predictions = nb.predict(rTrain)

                    nb.fit(rTest)
                    test_predictions = nb.predict(rTest)

                elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
                    gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'GDAFactor':

                    gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'LDABestFactor':

                    num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds, plot=False,
                                                            shrinkage=None)
                    lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)
                    num_factors_shuffle[fold] = num_fact

                elif 'KNeighbors' in method:

                    num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds, plot=False)
                    # num_neighb = len(np.unique(cTrain))
                    # num_neighb = 1
                    num_neighbors[fold] = num_neighb

                    knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
                    knn.fit(rTrain, rTrain)
                    train_predictions = knn.predict()

                    knn = knnC.KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
                    knn.fit(rTest, rTrain)
                    test_predictions = knn.predict()

                train_scores_dist[n_other, fold] = calc_distance_error(predictions=train_predictions, labels=cTrain,
                                                              stim_class=stim_class, stim_category=stim_category,
                                                              stim_template=stim_template)
                test_scores_dist[n_other, fold] = calc_distance_error(predictions=test_predictions, labels=cTest,
                                                             stim_class=stim_class, stim_category=stim_category,
                                                             stim_template=stim_template)

                train_scores[n_other, fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[n_other, fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                if 'Shuffled' in method:
                    rTest = shuffle_trials_within_class(rTest, cTest)

                if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or (
                    'ShuffledLDA' in method):
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif 'bootstrapLDA' in method:

                    lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate',
                                'LDAFactorConstrict']:
                    lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)

                elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate',
                                'NaiveBayesConstrict']:
                    nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
                    nb.fit(rTrain)
                    train_predictions = nb.predict(rTrain)

                    nb.fit(rTest)
                    test_predictions = nb.predict(rTest)

                elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
                    gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'GDAFactor':

                    gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
                    gda.fit(rTrain)
                    train_predictions = gda.predict(rTrain)

                    gda.fit(rTest)
                    test_predictions = gda.predict(rTest)

                elif method == 'LDABestFactor':

                    num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds, plot=False,
                                                            shrinkage=None)
                    lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
                    lda.fit(rTrain)
                    train_predictions = lda.predict(rTrain)

                    lda.fit(rTest)
                    test_predictions = lda.predict(rTest)
                    num_factors_shuffle[fold] = num_fact

                elif 'KNeighbors' in method:

                    num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds, plot=False)
                    # num_neighb = len(np.unique(cTrain))
                    # num_neighb = 1
                    num_neighbors[fold] = num_neighb

                    knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
                    knn.fit(rTrain, rTrain)
                    train_predictions = knn.predict()

                    knn = knnC.KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
                    knn.fit(rTest, rTrain)
                    test_predictions = knn.predict()

                train_shuffle_scores_dist[n_other, fold] = calc_distance_error(predictions=train_predictions, labels=cTrain,
                                                                      stim_class=stim_class,
                                                                      stim_category=stim_category,
                                                                      stim_template=stim_template)
                test_shuffle_scores_dist[n_other, fold] = calc_distance_error(predictions=test_predictions, labels=cTest,
                                                                     stim_class=stim_class,
                                                                     stim_category=stim_category,
                                                                     stim_template=stim_template)

                train_shuffle_scores[n_other, fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[n_other, fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

        test_scores_dict[stim_category] = test_scores
        train_scores_dict[stim_category] = train_scores
        test_shuffle_scores_dict[stim_category] = test_shuffle_scores
        train_shuffle_scores_dict[stim_category] = train_shuffle_scores

        test_scores_dist_dict[stim_category] = test_scores_dist
        train_scores_dist_dict[stim_category] = train_scores_dist
        test_shuffle_scores_dist_dict[stim_category] = test_shuffle_scores_dist
        train_shuffle_scores_dist_dict[stim_category] = train_shuffle_scores_dist


    return test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
        train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict


def compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner=2):

    if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or ('ShuffledLDA' in method):
        lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif 'bootstrapLDA' in method:

        lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
        lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate', 'LDAFactorConstrict']:
        lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate', 'NaiveBayesConstrict']:
        nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
        nb.fit(rTrain)
        train_predictions = nb.predict(rTrain)

        nb.fit(rTest)
        test_predictions = nb.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
        gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
        gda.fit(rTrain)
        train_predictions = gda.predict(rTrain)

        gda.fit(rTest)
        test_predictions = gda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method == 'GDAFactor':

        gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
        gda.fit(rTrain)
        train_predictions = gda.predict(rTrain)

        gda.fit(rTest)
        test_predictions = gda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method == 'LDABestFactor':

        num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds_inner, plot=False, shrinkage=None)
        lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, num_fact

    elif 'KNeighbors' in method:

        num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds_inner, plot=False)

        knn = knnC.KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
        knn.fit(rTrain, rTrain)
        train_predictions = knn.predict()

        knn = knnC.KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
        knn.fit(rTest, rTrain)
        test_predictions = knn.predict()

        return train_predictions, test_predictions, num_neighb

    elif 'RandomForest' in method:

        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(rTrain, cTrain)

        train_predictions = rfc.predict(rTrain)
        test_predictions = rfc.predict(rTest)

        return train_predictions, test_predictions, None


def run_decoding_missing_expts(standardize=True, detect_events=False, subsampleBehavior=True, flat_class=True, save_dir='/local1/Documents/projects/cam_analysis/decode_results_L0events'):

    results_files = os.listdir(save_dir)
    exps = boc.get_ophys_experiments(stimuli=['natural_scenes', 'static_gratings', 'drifting_gratings'])
    exp_ids = [exp['id'] for exp in exps]

    missing_exps = [exp_id for exp_id in exp_ids if str(exp_id)+'_train_error.pkl' not in results_files]

    for n, expt in enumerate(missing_exps):
        print('expt '+str(n)+'/'+str(len(missing_exps)))
        run_decoding_expt(expt, save_dir=save_dir)


def run_decoding_expt(expt, standardize=True, subsampleBehavior=True, flat_class=True, detect_events=True, save_dir='/local1/Documents/projects/cam_analysis/decode_results_L0events', analysis_file_dir='/allen/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/event_analysis_files_2018_09_25'):

    '''
    :return:
    '''

    # print expt
    expt = int(expt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # decode_methods = ['diagLDA', 'LDA', 'ShuffledLDA']
    # decode_methods = ['KNeighbors']
    # suffixes=['Run','Stat']
    # decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'RandomForest', 'ShuffledKNeighbors', 'ShuffledLDA','ShuffledRandomForest'] # need to check memory scaling of random forests for decoding movies
    decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'KNeighbors', 'ShuffledKNeighbors', 'ShuffledLDA'] # need to check memory scaling of random forests for decoding movies

    # decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'ShuffledKNeighbors', 'ShuffledLDA']
    suffixes = ['', 'Stat', 'Run', 'Dilate', 'Constrict']

    methods = []
    for method in decode_methods:
        for suffix in suffixes:
            methods.append(method+suffix)

    stim_class_dict = {'A': ['dg'], 'B': ['sg','ns'], 'C': [], 'C2': []} # not decoding movies
    # stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}
    stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}


    run_thresh = 1.
    min_stim_repeats = 5 # must be at least num_folds * num_folds of inner cross-val for stratified k-fold; for k-fold will have at least 2*min_repeats points
    num_folds = 5
    num_folds_inner = 2

    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)

    try:
        expt_stimuli = data_set.list_stimuli()
    except:
        raise Exception('no data for exp '+str(expt)+', skipping')
        # continue

    if 'drifting_gratings' in expt_stimuli:
        session_type = 'A'
    elif 'static_gratings' in expt_stimuli:
        session_type = 'B'
    elif 'locally_sparse_noise' in expt_stimuli:
        session_type = 'C'
    elif 'locally_sparse_noise_8deg' in expt_stimuli:
        session_type = 'C2'
    else:
        raise Exception('Session type not defined')

    if session_type == 'A':
        run_window = 60
    else:
        run_window = 15

    test_dict = {}
    train_dict = {}
    test_shuffle_dict = {}
    train_shuffle_dict = {}

    test_dist_dict = {}
    train_dist_dict = {}
    test_shuffle_dist_dict = {}
    train_shuffle_dist_dict = {}

    num_factors_dict = {}
    num_factors_shuffle_dict = {}
    num_neighbors_dict = {}
    num_neighbors_shuffle_dict = {}
    test_confusion_dict = {}
    train_confusion_dict = {}

    _, dff = data_set.get_dff_traces()
    dxcm, _ = data_set.get_running_speed()

    N, T = dff.shape
    try:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
    except:
        print('no pupil size information')
        pupil_size = None

    if detect_events:

        l0 = L0_analysis(data_set)
        dff = l0.get_events()

        # event_file = os.path.join(event_dir, 'expt_'+str(expt)+'_events.npy')
        # dff = np.load(event_file)
        #
        # noise_stds = np.load(os.path.join(event_dir, 'expt_'+str(expt)+'_NoiseStd.npy'))[:, None]
        # dff *= noise_stds * 10. # rescale to units of df/f

    print('decoding running/stationary')
    methods_calc = decode_methods
    try:
        master_stim_table = data_set.get_stimulus_table('master')
    except:
        print('no good master stim table')
        master_stim_table = None

    if pupil_size is not None:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
        response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind, width=run_window)
    else:
        print('no pupil size information')
        response, running_speed, _ = get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, width=run_window)
        pupil_array = None
        methods = [m for m in methods if ('Dilate' not in m) and ('Constrict' not in m) ]


    # fit mixture model to running speeds to classify running / stationary
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
    stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

    Nstationary_labels = len(stationary_label)
    stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
                        stationary_vars[i] is min(stationary_vars)]

    # running stuff
    if Nstationary_labels != 0:
        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == stationary_label).astype('float')  # thing to decode!
        stim_types = np.unique(stims_calc)
    else:
        stim_types = [-10]

    if len(stim_types) > 1:
        if subsampleBehavior and flat_class:
            NperStat = sum(stims_calc == 1.)
            NperRun = len(stims_calc) - NperStat
            Nsweeps = min(NperStat, NperRun)

            RunSweeps = np.random.choice(np.where(stims_calc == 0.)[0], size=Nsweeps)
            StatSweeps = np.random.choice(np.where(stims_calc == 1.)[0], size=Nsweeps)
            SweepInds = np.concatenate((RunSweeps, StatSweeps), axis=0)
            stims_calc = stims_calc[SweepInds]
            response = response[SweepInds]

        stims_shuffle_calc = np.random.permutation(stims_calc)

        if len(response) >= num_folds:
            for method in methods_calc:

                print(method)

                skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

                if 'Shuffled' in method:
                    response_calc = shuffle_trials_within_class(response, stims_calc)
                else:
                    response_calc = response.copy()

                test_scores = np.zeros((num_folds))
                train_scores = np.zeros((num_folds))
                train_shuffle_scores = np.zeros((num_folds))
                test_shuffle_scores = np.zeros((num_folds))

                # test_confusion = np.zeros((len(stim_types), len(stim_types)))
                # train_confusion = np.zeros((len(stim_types), len(stim_types)))

                if 'LDABestFactor' in method:
                    num_factors = np.zeros((num_folds))
                    num_factors_shuffle = np.zeros((num_folds))

                if 'KNeighbors' in method:
                    num_neighbors = np.zeros((num_folds))
                    num_neighbors_shuffle = np.zeros((num_folds))

                fold = 0
                scaler = StandardScaler()
                for train, test in skf.split(response_calc, stims_calc):

                    rTrain = response_calc[train]
                    rTest = response_calc[test]
                    cTrain = stims_calc[train]
                    cTest = stims_calc[test]

                    if standardize:
                        scaler.fit(rTrain)
                        rTrain = scaler.transform(rTrain)
                        rTest = scaler.transform(rTest)

                    train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                    if method == 'KNeighbors':
                        num_neighbors[fold] = param
                    elif method == 'LDABestFactor':
                        num_factors[fold] = param

                    train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                    test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                    fold += 1

                    # if len(np.unique(cTrain)) < len(stim_types)
                    #     # get labels for train stims, expand temp confusion to full size
                    #     temp_confusion1 = confusion_matrix(cTrain, train_predictions)
                    #     cTrain_labels = np.unique(cTrain)
                    #     temp_confusion2 = np.zeros((len(stim_types), len(stim_types)))
                    #     for ss, stim_temp in enumerate(stim_types):
                    #         ind = np.where()
                    #
                    # else:

                    # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                    # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

                fold = 0
                for train, test in skf.split(response_calc, stims_shuffle_calc):

                    rTrain = response_calc[train]
                    rTest = response_calc[test]
                    cTrain = stims_shuffle_calc[train]
                    cTest = stims_shuffle_calc[test]

                    if standardize:
                        scaler.fit(rTrain)
                        rTrain = scaler.transform(rTrain)
                        rTest = scaler.transform(rTest)

                    train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                    if method == 'KNeighbors':
                        num_neighbors[fold] = param
                    elif method == 'LDABestFactor':
                        num_factors[fold] = param

                    train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                    test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                    fold += 1

                test_dict['runState' + '_' + method] = test_scores
                train_dict['runState' + '_' + method] = train_scores
                test_shuffle_dict['runState' + '_' + method] = test_shuffle_scores
                train_shuffle_dict['runState' + '_' + method] = train_shuffle_scores

                # test_confusion_dict[stim_class + '_' + stim_category + '_' + method] = test_confusion
                # train_confusion_dict[stim_class + '_' + stim_category + '_' + method] = train_confusion

                if method == 'LDABestFactor':
                    num_factors_dict['runState'] = num_factors
                    num_factors_shuffle_dict['runState'] = num_factors_shuffle

                elif 'KNeighbors' in method:
                    num_neighbors_dict['runState' + '_' + method] = num_neighbors
                    num_neighbors_shuffle_dict['runState' + '_' + method] = num_neighbors_shuffle

    # decode pupil diameter - wide / narrow
    if (pupil_array is not None):
        print('decoding pupil wide / narrow')

        response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind, width=run_window) # get full response array again

        pupil_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.2,
                                                    covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)

        good_ind = np.isfinite(pupil_array)
        X = pupil_array[good_ind].reshape(-1, 1)

        pupil_dpgmm.fit(X)
        Y = pupil_dpgmm.predict(X)
        means = pupil_dpgmm.means_
        vars = pupil_dpgmm.covariances_

        labels = -1 * np.ones(pupil_array.shape, dtype=np.int)
        labels[good_ind] = Y

        dilate_states = np.unique(Y)
        means = [means[i][0] for i in dilate_states]

        constrict_label = dilate_states[np.where(means == np.amin(means))]

        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == constrict_label).astype('float')  # thing to decode!
        stims_shuffle_calc = np.random.permutation(stims_calc)

        stim_types = np.unique(stims_calc)

        if subsampleBehavior and flat_class:
            NperConstrict = sum(stims_calc == 1.)
            NperDilate = len(stims_calc) - NperConstrict
            Nsweeps = min(NperConstrict, NperDilate)

            DilateSweeps = np.random.choice(np.where(stims_calc == 0.)[0], size=Nsweeps)
            ConstrictSweeps = np.random.choice(np.where(stims_calc == 1.)[0], size=Nsweeps)
            SweepInds = np.concatenate((DilateSweeps, ConstrictSweeps), axis=0)
            stims_calc = stims_calc[SweepInds]
            response = response[SweepInds]

        stims_shuffle_calc = np.random.permutation(stims_calc)

    if (len(stim_types) > 1) and (len(response) >= num_folds):
        for method in methods_calc:

            print(method)

            skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'Shuffled' in method:
                response_calc = shuffle_trials_within_class(response_calc, stims_calc)
            else:
                response_calc = response.copy()

            test_scores = np.zeros((num_folds))
            train_scores = np.zeros((num_folds))
            train_shuffle_scores = np.zeros((num_folds))
            test_shuffle_scores = np.zeros((num_folds))

            # test_confusion = np.zeros((len(stim_types), len(stim_types)))
            # train_confusion = np.zeros((len(stim_types), len(stim_types)))

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0

            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

                # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                fold += 1

            test_dict['pupil' + '_' + method] = test_scores
            train_dict['pupil' + '_' + method] = train_scores
            test_shuffle_dict['pupil' + '_' + method] = test_shuffle_scores
            train_shuffle_dict['pupil' + '_' + method] = train_shuffle_scores

            if method == 'LDABestFactor':
                num_factors_dict['pupil'] = num_factors
                num_factors_shuffle_dict['pupil'] = num_factors_shuffle

            elif 'KNeighbors' in method:
                num_neighbors_dict['pupil' + '_' + method] = num_neighbors
                num_neighbors_shuffle_dict['pupil' + '_' + method] = num_neighbors_shuffle

    print('decoding visual stimulus')
    stim_class_list = stim_class_dict[session_type]
    if len(stim_class_list) > 0:         # decode visual stimulus
        for stim_class in stim_class_list:
            if stim_class == 'ns':
                stim_table = data_set.get_stimulus_table('natural_scenes')
                stim_template = data_set.get_stimulus_template('natural_scenes')
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'NaturalScenes', str(expt)+'_ns_events_analysis.h5'))

            elif stim_class == 'dg':
                stim_table = data_set.get_stimulus_table('drifting_gratings')
                stim_template = None
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'DriftingGratings', str(expt)+'_dg_events_analysis.h5'))

            elif stim_class == 'sg':
                stim_table = data_set.get_stimulus_table('static_gratings')
                stim_template = None
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'StaticGratings', str(expt)+'_sg_events_analysis.h5'))

            # elif stim_class == 'nm1':
            #     stim_table = data_set.get_stimulus_table('natural_movie_one')
            #     stim_template = data_set.get_stimulus_template('natural_movie_one')
            #
            # elif stim_class == 'nm2':
            #     stim_table = data_set.get_stimulus_table('natural_movie_two')
            #     stim_template = data_set.get_stimulus_template('natural_movie_two')
            #
            # elif stim_class == 'nm3':
            #     stim_table = data_set.get_stimulus_table('natural_movie_three')
            #     stim_template = data_set.get_stimulus_template('natural_movie_three')

            response = analysis_file['mean_sweep_events'].values()[3].value # trials x neurons

            if 'nm' in stim_class:
                # response = get_response_table(dff, stim_table, stim=stim_class, log=False, width=6)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class, width=6)
            else:
                # response = get_response_table(dff, stim_table, stim=stim_class, log=False)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class)

            labels = run_dpgmm.predict(running_speed.reshape(-1, 1))
            ind_stationary = (labels == stationary_label)
            ind_run = ~ind_stationary

            try:
                pupil_t, pupil_size = data_set.get_pupil_size()
                pupil_size, nan_ind = interpolate_pupil_size(pupil_size)

                if 'nm' in stim_class:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class, width=6)
                else:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class)
                labels = pupil_dpgmm.predict(pupil_array.reshape(-1, 1))
                ind_constrict = (labels == constrict_label)

            except:
                print('no pupil size information')
                pupil_size = None

            ''' set up stim labels '''
            decode_list = list(stim_category_dict[stim_class])
            if len(decode_list) > 0:  print("\tStimulus class:  ", stim_class)
            if len(decode_list) > 1:  decode_list.append('all')

            for stim_category in decode_list:
                print("\t\tStimulus Category:  ", stim_category)
                if (stim_category != 'all') and (stim_category != 'run'):
                    stims = np.array(stim_table[stim_category])
                elif stim_category == 'run':
                    stims = ind_stationary.astype('int')
                else:
                    stim_temp_array = [np.array(stim_table[stim_temp]) for stim_temp in decode_list[:-1]]
                    stims = np.vstack(stim_temp_array).T
                    stims = np.array([str(s_str) for s_str in stims])

                if stim_class == 'dg':

                    if stim_category == 'temporal_frequency':
                        stims = np.ceil(stims*100)
                    blank_mask = np.array(stim_table['blank_sweep'].astype('bool'))
                    stims[blank_mask] = -1

                elif stim_class == 'sg':

                    if stim_category == 'spatial_frequency':
                        stims = np.ceil(stims*100)

                    if stim_category != 'all':
                        blank_mask = ~np.isfinite(stims)
                        stims[blank_mask] = -1
                    else:
                        blank_mask = np.zeros(stims.shape).astype('bool')
                        for kk in range(len(stims)):
                            if 'nan' in stims[kk]: blank_mask[kk] = True
                        stims[blank_mask] = -1


                ''' set up response / stims for running '''

                stims_shuffle = np.random.permutation(stims)
                stim_types = np.unique(stims)


                if Nstationary_labels > 0:
                    response_stationary = response[ind_stationary, :]
                    response_run = response[ind_run, :]

                    stims_stationary = stims[ind_stationary]
                    stims_run = stims[ind_run]

                    stim_types_stat = np.unique(stims_stationary)
                    stim_types_run = np.unique(stims_run)

                    Nperstim_stat = [sum(stims_stationary == c) for c in stim_types_stat]
                    Nperstim_run = [sum(stims_run == c) for c in stim_types_run]


                    ''' for running / stationary, only use stims that have >= min_stim_repeats repetitions '''
                    methods_calc = methods[:] # slice to copy

                    # if len(Nperstim_run) > 1:
                    #     stims_run_good = [stim for i, stim in enumerate(stim_types_run) if Nperstim_run[i] >= min_stim_repeats]
                    #     Nperstim_run = [i for i in Nperstim_run if i >= min_stim_repeats]
                    #
                    # if len(Nperstim_stat) > 1:
                    #     stims_stat_good = [stim for i, stim in enumerate(stim_types_stat) if Nperstim_stat[i] >= min_stim_repeats]
                    #     Nperstim_stat = [i for i in Nperstim_stat if i >= min_stim_repeats]

                    if (len(Nperstim_run) > 1) and (len(Nperstim_stat) > 1):
                        stims_run_good = [stim for i, stim in enumerate(stim_types_run) if (Nperstim_run[i] >= min_stim_repeats)]
                        stims_stat_good = [stim for i, stim in enumerate(stim_types_stat) if (Nperstim_stat[i] >= min_stim_repeats)]

                        stims_run_good = [stim for i, stim in enumerate(stims_run_good) if stim in stims_stat_good]
                        stims_stat_good = stims_run_good

                    if (len(Nperstim_run) <= 1) or (len(Nperstim_stat) <= 1) or (len(stims_run_good) < 2) or (len(stims_stat_good) < 2):
                        print('not enough stims during running and stationary')
                        methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]
                        stims_run_good = []
                        stims_stat_good = []

                    if (len(stims_run_good) > 0) and (len(stims_stat_good) > 0):
                        if subsampleBehavior: # use same total number of trials per stimulus in each condition

                            if flat_class:
                                stim_types_run = np.unique(stims_run_good)  # already know has to be in stims_stat good

                                stims_stat_mask = [False for x in range(len(stims_stationary))]
                                stims_run_mask = [False for x in range(len(stims_run))]

                                Nperstim_list = []
                                for stim in stim_types_run:
                                    Nperstim_list.append(min([sum(stims_stationary == stim), sum(stims_run == stim)]))

                                Nperstim = min(Nperstim_list)

                                for stim in stim_types_run:
                                    stim_ind_stat = np.random.choice(np.where(stims_stationary == stim)[0], size=Nperstim,
                                                                     replace=False)
                                    stim_ind_run = np.random.choice(np.where(stims_run == stim)[0], size=Nperstim,
                                                                    replace=False)

                                    for i in stim_ind_stat: stims_stat_mask[i] = True
                                    for i in stim_ind_run: stims_run_mask[i] = True


                            else:
                                stim_types_run = np.unique(stims_run_good) # already know has to be in stims_stat good

                                stims_stat_mask = [False for x in range(len(stims_stationary))]
                                stims_run_mask = [False for x in range(len(stims_run))]

                                for stim in stim_types_run:

                                    Nperstim = min([sum(stims_stationary == stim), sum(stims_run == stim)])

                                    stim_ind_stat = np.random.choice(np.where(stims_stationary == stim)[0], size=Nperstim, replace=False)
                                    stim_ind_run = np.random.choice(np.where(stims_run == stim)[0], size=Nperstim, replace=False)

                                    for i in stim_ind_stat: stims_stat_mask[i] = True
                                    for i in stim_ind_run: stims_run_mask[i] = True

                        else:
                            stims_stat_mask = [False for x in range(len(stims_stationary))]
                            for i, stim in enumerate(stims_stationary):
                                if stim in stims_stat_good:
                                    stims_stat_mask[i] = True

                            stims_run_mask = [False for x in range(len(stims_run))]
                            for i, stim in enumerate(stims_run):
                                if stim in stims_run_good:
                                    stims_run_mask[i] = True

                        stims_stat_mask = np.array(stims_stat_mask).astype('bool')
                        stims_run_mask = np.array(stims_run_mask).astype('bool')
                        stims_stationary = stims_stationary[stims_stat_mask]
                        response_stationary = response_stationary[stims_stat_mask, :]

                        stims_run = stims_run[stims_run_mask]
                        response_run = response_run[stims_run_mask, :]

                        stims_stationary_shuffle = np.random.permutation(stims_stationary)
                        stims_run_shuffle = np.random.permutation(stims_run)

                    else:
                        methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]
                else:
                    methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]


                ''' set up response / stims for pupil narrow / wide '''
                if pupil_size is not None:

                    response_constrict = response[ind_constrict, :]
                    response_dilate = response[~ind_constrict, :]

                    stims_shuffle = np.random.permutation(stims)

                    stims_constrict = stims[ind_constrict]
                    stims_dilate = stims[~ind_constrict]

                    stim_types = np.unique(stims)
                    stim_types_constrict = np.unique(stims_constrict)
                    stim_types_dilate = np.unique(stims_dilate)

                    Nperstim_constrict = [sum(stims_constrict == c) for c in stim_types_constrict]
                    Nperstim_dilate = [sum(stims_dilate == c) for c in stim_types_dilate]

                    ''' for dilated/constricted, only use stims that have >= min_stim_repeats repetitions '''

                    if (len(Nperstim_dilate) > 1) and (len(Nperstim_constrict) > 1):
                        stims_dilate_good = [stim for i, stim in enumerate(stim_types_dilate) if
                                          (Nperstim_dilate[i] >= min_stim_repeats)]
                        stims_constrict_good = [stim for i, stim in enumerate(stim_types_constrict) if
                                           (Nperstim_constrict[i] >= min_stim_repeats)]

                        stims_dilate_good = [stim for i, stim in enumerate(stims_dilate_good) if stim in stims_constrict_good]
                        stims_constrict_good = stims_dilate_good


                    if (len(Nperstim_dilate) <= 1) or (len(Nperstim_constrict) <= 1) or (len(stims_dilate_good) < 2) or (len(stims_constrict_good) < 2):

                        print('not enough stims during dilated and constricted pupils')
                        methods_calc = [m for m in methods_calc if ('Constrict' not in m) and ('Dilate' not in m)]
                        stims_dilate_good = []
                        stims_constrict_good = []

                    if (len(stims_dilate_good) > 0) and (len(stims_constrict_good) > 0):
                        if subsampleBehavior:  # use same total number of trials per stimulus in each condition

                            if flat_class:
                                if flat_class:
                                    stim_types_dilate = np.unique(stims_dilate_good)  # already know has to be in stims_constrict good

                                    stims_constrict_mask = [False for x in range(len(stims_constrict))]
                                    stims_dilate_mask = [False for x in range(len(stims_dilate))]

                                    Nperstim_list = []
                                    for stim in stim_types_dilate:
                                        Nperstim_list.append(min([sum(stims_constrict == stim), sum(stims_dilate == stim)]))

                                    Nperstim = min(Nperstim_list)

                                    for stim in stim_types_dilate:
                                        stim_ind_constrict = np.random.choice(np.where(stims_constrict == stim)[0],
                                                                         size=Nperstim,
                                                                         replace=False)
                                        stim_ind_dilate = np.random.choice(np.where(stims_dilate == stim)[0], size=Nperstim,
                                                                        replace=False)

                                        for i in stim_ind_constrict: stims_constrict_mask[i] = True
                                        for i in stim_ind_dilate: stims_dilate_mask[i] = True



                            else:
                                stim_types_dilate = np.unique(stims_dilate_good)  # already know has to be in stims_constrict good

                                stims_constrict_mask = [False for x in range(len(stims_constrict))]
                                stims_dilate_mask = [False for x in range(len(stims_dilate))]

                                for stim in stim_types_dilate:

                                    Nperstim = min([sum(stims_constrict == stim), sum(stims_dilate == stim)])

                                    stim_ind_constrict = np.random.choice(np.where(stims_constrict == stim)[0], size=Nperstim,
                                                                     replace=False)
                                    stim_ind_dilate = np.random.choice(np.where(stims_dilate == stim)[0], size=Nperstim,
                                                                    replace=False)

                                    for i in stim_ind_constrict: stims_constrict_mask[i] = True
                                    for i in stim_ind_dilate: stims_dilate_mask[i] = True


                        else:
                            stims_constrict_mask = [False for x in range(len(stims_constrict))]
                            for i, stim in enumerate(stims_constrict):
                                if stim in stims_constrict_good:
                                    stims_constrict_mask[i] = True

                            stims_dilate_mask = [False for x in range(len(stims_dilate))]
                            for i, stim in enumerate(stims_dilate):
                                if stim in stims_dilate_good:
                                    stims_dilate_mask[i] = True

                        stims_constrict_mask = np.array(stims_constrict_mask).astype('bool')
                        stims_dilate_mask = np.array(stims_dilate_mask).astype('bool')
                        stims_constrict = stims_constrict[stims_constrict_mask]
                        response_constrict = response_constrict[stims_constrict_mask, :]

                        stims_dilate = stims_dilate[stims_dilate_mask]
                        response_dilate = response_dilate[stims_dilate_mask, :]

                        stims_constrict_shuffle = np.random.permutation(stims_constrict)
                        stims_dilate_shuffle = np.random.permutation(stims_dilate)

                else:
                    methods_calc = [m for m in methods_calc if ('Constrict' not in m) and ('Dilate' not in m)]


                for method in methods_calc:

                    print(method)

                    if 'Run' in method:
                        response_calc = response_run.copy()
                        stims_calc = stims_run.copy()
                        stims_shuffle_calc = stims_run_shuffle.copy()
                    elif 'Stat' in method:
                        response_calc = response_stationary.copy()
                        stims_calc = stims_stationary.copy()
                        stims_shuffle_calc = stims_stationary_shuffle.copy()
                    elif 'Dilate' in method:
                        response_calc = response_dilate.copy()
                        stims_calc = stims_dilate.copy()
                        stims_shuffle_calc = stims_dilate_shuffle.copy()
                    elif 'Constrict' in method:
                        response_calc = response_constrict.copy()
                        stims_calc = stims_constrict.copy()
                        stims_shuffle_calc = stims_constrict_shuffle.copy()
                    else:
                        response_calc = response.copy()
                        stims_calc = stims.copy()
                        stims_shuffle_calc = stims_shuffle.copy()

                    # skf = StratifiedKFold(n_splits=num_folds, shuffle=False)
                    skf = KFold(n_splits=num_folds, shuffle=False)

                    test_scores = np.zeros((num_folds))
                    train_scores = np.zeros((num_folds))
                    train_shuffle_scores = np.zeros((num_folds))
                    test_shuffle_scores = np.zeros((num_folds))

                    test_scores_dist = np.zeros((num_folds))
                    train_scores_dist = np.zeros((num_folds))
                    train_shuffle_scores_dist = np.zeros((num_folds))
                    test_shuffle_scores_dist = np.zeros((num_folds))

                    test_confusion = np.zeros((len(stim_types), len(stim_types)))
                    train_confusion = np.zeros((len(stim_types), len(stim_types)))

                    if 'LDABestFactor' in method:
                        num_factors = np.zeros((num_folds))
                        num_factors_shuffle = np.zeros((num_folds))

                    if 'KNeighbors' in method:
                        num_neighbors = np.zeros((num_folds))
                        num_neighbors_shuffle = np.zeros((num_folds))

                    fold = 0
                    scaler = StandardScaler()
                    for train, test in skf.split(response_calc, stims_calc):

                        rTrain = response_calc[train]
                        rTest = response_calc[test]
                        cTrain = stims_calc[train]
                        cTest = stims_calc[test]

                        if standardize:
                            scaler.fit(rTrain)
                            rTrain = scaler.transform(rTrain)
                            rTest = scaler.transform(rTest)

                        if 'Shuffled' in method:
                            rTest = shuffle_trials_within_class(rTest, cTest)

                        train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                        if method == 'KNeighbors':
                            num_neighbors[fold] = param
                        elif method == 'LDABestFactor':
                            num_factors[fold] = param

                        train_scores_dist[fold] = calc_distance_error(predictions=train_predictions, labels=cTrain, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)
                        test_scores_dist[fold] = calc_distance_error(predictions=test_predictions, labels=cTest, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)

                        train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                        test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                        fold += 1

                        train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                        test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)


                    fold = 0
                    for train, test in skf.split(response_calc, stims_shuffle_calc):

                        rTrain = response_calc[train]
                        rTest = response_calc[test]
                        cTrain = stims_shuffle_calc[train]
                        cTest = stims_shuffle_calc[test]

                        if standardize:
                            scaler.fit(rTrain)
                            rTrain = scaler.transform(rTrain)
                            rTest = scaler.transform(rTest)

                        if 'Shuffled' in method:
                            rTest = shuffle_trials_within_class(rTest, cTest)

                        train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                        if method == 'KNeighbors':
                            num_neighbors_shuffle[fold] = param
                        elif method == 'LDABestFactor':
                            num_factors_shuffle[fold] = param

                        train_shuffle_scores_dist[fold] = calc_distance_error(predictions=train_predictions, labels=cTrain, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)
                        test_shuffle_scores_dist[fold] = calc_distance_error(predictions=test_predictions, labels=cTest, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)

                        train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                        test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                        fold += 1

                    test_dict[stim_class + '_' + stim_category + '_' + method] = test_scores
                    train_dict[stim_class + '_' + stim_category + '_' + method] = train_scores
                    test_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores
                    train_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores

                    test_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_scores_dist
                    train_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_scores_dist
                    test_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores_dist
                    train_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores_dist

                    test_confusion_dict[stim_class + '_' + stim_category + '_' + method] = test_confusion
                    train_confusion_dict[stim_class + '_' + stim_category + '_' + method] = train_confusion

                    if method == 'LDABestFactor':
                        num_factors_dict[stim_class + '_' + stim_category] = num_factors
                        num_factors_shuffle_dict[stim_class + '_' + stim_category] = num_factors_shuffle

                    elif 'KNeighbors' in method:
                        num_neighbors_dict[stim_class + '_' + stim_category + '_' + method] = num_neighbors
                        num_neighbors_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = num_neighbors_shuffle

                    # print 'train score: ' + str(np.mean(train_scores)) + ' (+/- ' +str(np.std(train_scores)*2) + ')'
                    # print 'test score: ' + str(np.mean(test_scores)) + ' (+/- ' +str(np.std(test_scores)*2) + ')'
                    # print 'shuffle score: ' + str(np.mean(test_shuffle_scores)) + ' (+/- ' +str(np.std(test_shuffle_scores)*2) + ')' #   # decode visual stimulus


    savefile = os.path.join(save_dir, str(expt)+'_test_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_confusion_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_confusion_dict, error_file)
    error_file.close()

    # print 'session type: '+str(session_type)+' , elapsed time: '+str(time.time()-start_time)


def run_decoding_gratings_expt(expt, standardize=True, detect_events=True, subsampleBehavior=True, flat_class=True):

    '''
    :return:
    '''

    # print expt
    expt = int(expt)
    # save_dir = '/local1/Documents/projects/cam_analysis/decode_results'

    if detect_events:
        # save_dir = '/allen/aibs/mat/gkocker/cam_decode/decode_results_L0events'
        save_dir = '/home/gabeo/decode_results_gratingsL0events'
    else:
        # save_dir = '/allen/aibs/mat/gkocker/cam_decode/decode_results'
        save_dir = '/home/gabeo/decode_results_gratings'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # decode_methods = ['diagLDA', 'LDA', 'ShuffledLDA']
    # decode_methods = ['KNeighbors']
    # suffixes=['Run','Stat']
    decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'RandomForest', 'ShuffledKNeighbors', 'ShuffledLDA','ShuffledRandomForest'] # need to check memory scaling of random forests for decoding movies
    # decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'ShuffledKNeighbors', 'ShuffledLDA']
    suffixes = ['', 'Stat', 'Run', 'Dilate', 'Constrict']

    methods = []
    for method in decode_methods:
        for suffix in suffixes:
            methods.append(method+suffix)

    stim_class_dict = {'A': ('dg',), 'B': ('sg',), 'C': (), 'C2': ()}
    stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}

    # stim_class_dict = {'A': ('dg'), 'B': ('sg','ns'), 'C': (), 'C2': ()}
    # stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}

    run_thresh = 1.
    min_stim_repeats = 5 # must be at least num_folds * num_folds of inner cross-val for stratified k-fold; for k-fold will have at least 2*min_repeats points
    num_folds = 5
    num_folds_inner = 2

    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)

    try:
        expt_stimuli = data_set.list_stimuli()
    except:
        raise Exception('no data for exp '+str(expt)+', skipping')
        # continue

    if 'drifting_gratings' in expt_stimuli:
        session_type = 'A'
    elif 'static_gratings' in expt_stimuli:
        session_type = 'B'
    elif 'locally_sparse_noise' in expt_stimuli:
        session_type = 'C'
    elif 'locally_sparse_noise_8deg' in expt_stimuli:
        session_type = 'C2'
    else:
        raise Exception('Session type not defined')

    if session_type == 'B':
        run_window = 60
    else:
        run_window = 15

    test_dict = {}
    train_dict = {}
    test_shuffle_dict = {}
    train_shuffle_dict = {}

    test_dist_dict = {}
    train_dist_dict = {}
    test_shuffle_dist_dict = {}
    train_shuffle_dist_dict = {}

    num_factors_dict = {}
    num_factors_shuffle_dict = {}
    num_neighbors_dict = {}
    num_neighbors_shuffle_dict = {}
    test_confusion_dict = {}
    train_confusion_dict = {}

    _, dff = data_set.get_dff_traces()
    dxcm, _ = data_set.get_running_speed()
    try:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
    except:
        print('no pupil size information')
        pupil_size = None

    if detect_events:
        l0a = L0_analysis(data_set, manifest_file=manifest_file)
        dff = l0a.get_events()

    print('decoding running/stationary')
    methods_calc = decode_methods
    try:
        master_stim_table = data_set.get_stimulus_table('master')
    except:
        print('no good master stim table')
        master_stim_table = None

    if pupil_size is not None:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
        response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind, width=run_window)
    else:
        print('no pupil size information')
        response, running_speed, _ = get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, width=run_window)
        pupil_array = None
        methods = [m for m in methods if ('Dilate' not in m) and ('Constrict' not in m) ]


#    fit mixture model to running speeds to classify running / stationary
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
    stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

    Nstationary_labels = len(stationary_label)
    stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
                        stationary_vars[i] is min(stationary_vars)]

    if Nstationary_labels != 0:
        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == stationary_label).astype('float')  # thing to decode!
        stims_shuffle_calc = np.random.permutation(stims_calc)
        stim_types = np.unique(stims_calc)

        for method in methods_calc:

            print(method)

            skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'Shuffled' in method:
                response_calc = shuffle_trials_within_class(response_calc, stims_calc)
            else:
                response_calc = response.copy()

            test_scores = np.zeros((num_folds))
            train_scores = np.zeros((num_folds))
            train_shuffle_scores = np.zeros((num_folds))
            test_shuffle_scores = np.zeros((num_folds))

            # test_confusion = np.zeros((len(stim_types), len(stim_types)))
            # train_confusion = np.zeros((len(stim_types), len(stim_types)))

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0
            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                fold += 1

                # if len(np.unique(cTrain)) < len(stim_types)
                #     # get labels for train stims, expand temp confusion to full size
                #     temp_confusion1 = confusion_matrix(cTrain, train_predictions)
                #     cTrain_labels = np.unique(cTrain)
                #     temp_confusion2 = np.zeros((len(stim_types), len(stim_types)))
                #     for ss, stim_temp in enumerate(stim_types):
                #         ind = np.where()
                #
                # else:

                # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

            test_dict['runState' + '_' + method] = test_scores
            train_dict['runState' + '_' + method] = train_scores
            test_shuffle_dict['runState' + '_' + method] = test_shuffle_scores
            train_shuffle_dict['runState' + '_' + method] = train_shuffle_scores

            # test_confusion_dict[stim_class + '_' + stim_category + '_' + method] = test_confusion
            # train_confusion_dict[stim_class + '_' + stim_category + '_' + method] = train_confusion

            if method == 'LDABestFactor':
                num_factors_dict['runState'] = num_factors
                num_factors_shuffle_dict['runState'] = num_factors_shuffle

            elif 'KNeighbors' in method:
                num_neighbors_dict['runState' + '_' + method] = num_neighbors
                num_neighbors_shuffle_dict['runState' + '_' + method] = num_neighbors_shuffle

    # decode pupil diameter - wide / narrow
    if (pupil_array is not None):
        print('decoding pupil wide / narrow')
        pupil_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.2,
                                                    covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)

        good_ind = np.isfinite(pupil_array)
        X = pupil_array[good_ind].reshape(-1, 1)

        pupil_dpgmm.fit(X)
        Y = pupil_dpgmm.predict(X)
        means = pupil_dpgmm.means_
        vars = pupil_dpgmm.covariances_

        labels = -1 * np.ones(pupil_array.shape, dtype=np.int)
        labels[good_ind] = Y

        dilate_states = np.unique(Y)
        means = [means[i][0] for i in dilate_states]

        constrict_label = dilate_states[np.where(means == np.amin(means))]

        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == stationary_label).astype('float')  # thing to decode!
        stims_shuffle_calc = np.random.permutation(stims_calc)

        stims_calc = (labels == stationary_label).astype('float')  # thing to decode!
        stims_shuffle_calc = np.random.permutation(stims_calc)
        stim_types = np.unique(stims_calc)

        for method in methods_calc:

            print(method)

            skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'Shuffled' in method:
                response_calc = shuffle_trials_within_class(response_calc, stims_calc)
            else:
                response_calc = response.copy()

            test_scores = np.zeros((num_folds))
            train_scores = np.zeros((num_folds))
            train_shuffle_scores = np.zeros((num_folds))
            test_shuffle_scores = np.zeros((num_folds))

            # test_confusion = np.zeros((len(stim_types), len(stim_types)))
            # train_confusion = np.zeros((len(stim_types), len(stim_types)))

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0

            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

                # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                fold += 1

            test_dict['pupil' + '_' + method] = test_scores
            train_dict['pupil' + '_' + method] = train_scores
            test_shuffle_dict['pupil' + '_' + method] = test_shuffle_scores
            train_shuffle_dict['pupil' + '_' + method] = train_shuffle_scores

            if method == 'LDABestFactor':
                num_factors_dict['pupil'] = num_factors
                num_factors_shuffle_dict['pupil'] = num_factors_shuffle

            elif 'KNeighbors' in method:
                num_neighbors_dict['pupil' + '_' + method] = num_neighbors
                num_neighbors_shuffle_dict['pupil' + '_' + method] = num_neighbors_shuffle

    print('decoding visual stimulus')
    stim_class_list = stim_class_dict[session_type]
    if len(stim_class_list) > 0:         # decode visual stimulus
        for stim_class in stim_class_list:
            if stim_class == 'dg':
                stim_table = data_set.get_stimulus_table('drifting_gratings')
                stim_template = None
            elif stim_class == 'sg':
                stim_table = data_set.get_stimulus_table('static_gratings')
                stim_template = None

            if 'nm' in stim_class:
                response = get_response_table(dff, stim_table, stim=stim_class, log=False, width=6)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class, width=6)
            else:
                response = get_response_table(dff, stim_table, stim=stim_class, log=False)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class)

            labels = run_dpgmm.predict(running_speed.reshape(-1, 1))
            ind_stationary = (labels == stationary_label)
            ind_run = ~ind_stationary

            try:
                pupil_t, pupil_size = data_set.get_pupil_size()
                pupil_size, nan_ind = interpolate_pupil_size(pupil_size)

                if 'nm' in stim_class:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class, width=6)
                else:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class)
                labels = pupil_dpgmm.predict(pupil_array.reshape(-1, 1))
                ind_constrict = (labels == constrict_label)

            except:
                print('no pupil size information')
                pupil_size = None

            ''' set up stim labels '''
            decode_list = list(stim_category_dict[stim_class])
            if len(decode_list) > 0:  print("\tStimulus class:  ", stim_class)
            if len(decode_list) > 1:  decode_list.append('all')

            if 'Stat' in method:
                response_calc = response[ind_stationary]
                stim_table_calc = stim_table[ind_stationary]

            elif 'Run' in method:
                response_calc = response[ind_run]
                stim_table_calc = stim_table[ind_run]

            elif 'Dilate' in method:
                response_calc = response[~ind_constrict]
                stim_table_calc = stim_table[~ind_constrict]

            elif 'Constrict' in method:
                response_calc = response[ind_constrict]
                stim_table_calc = stim_table[ind_constrict]

            else:
                response_calc = response
                stim_table_calc = stim_table

            if stim_class == 'dg':

                test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
                train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict = decode_drifting_grating_cross(response_calc, stim_table_calc, stim_template=None, method=method)

            elif stim_class == 'sg':
            #
                test_scores_dict, train_scores_dict, test_shuffle_scores_dict, train_shuffle_scores_dict, test_scores_dist_dict, \
                train_scores_dist_dict, test_shuffle_scores_dist_dict, train_shuffle_scores_dist_dict = decode_static_grating_cross(response_calc, stim_table_calc, stim_template=None, method=method)

            for stim_category in test_scores_dict.keys():
                test_dict[stim_class + '_' + stim_category + '_' + method] = test_scores_dict[stim_category]
                train_dict[stim_class + '_' + stim_category + '_' + method] = train_scores_dict[stim_category]
                test_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores_dict[stim_category]
                train_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores_dict[stim_category]

                test_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_scores_dist_dict[stim_category]
                train_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_scores_dist_dict[stim_category]
                test_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores_dist_dict[stim_category]
                train_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores_dist_dict[stim_category]


    savefile = os.path.join(save_dir, str(expt)+'_test_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_confusion_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_confusion_dict, error_file)
    error_file.close()

    # print 'session type: '+str(session_type)+' , elapsed time: '+str(time.time()-start_time)


if __name__== '__main__':

    # run_decoding_numNeurons_expt(expt=sys.argv[1])
    # run_decoding_across_expts()
    run_decoding_expt(expt=sys.argv[1], save_dir='/allen/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/decode_results')
    #
    # run_decoding_missing_expts()
