#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:06:41 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
import core


# class event_analysis(object):
#     def __init__(self, *args, **kwargs):
#         for k,v in kwargs.iteritems():
#             setattr(self, k, v)
#         self.session_id = session_id
#         save_path_head = core.get_save_path()
#         self.save_path = os.path.join(save_path_head, 'NaturalMovies')
#         self.l0_events = core.get_L0_events(self.session_id)
#         self.stim_table_2, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_movie_two')
#         self.stim_table_1c, _, _ = core.get_stim_table(self.session_id, 'natural_movie_one')
#
#
# class NaturalMoviesC(event_analysis):
#     def __init__(self, *args, **kwargs):
#         super(NaturalMoviesC, self).__init__(*args, **kwargs)

class NaturalMoviesC:
    def __init__(self, session_id):
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'NaturalMovies')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table_2, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_movie_two')
        self.stim_table_1c, _, _ = core.get_stim_table(self.session_id, 'natural_movie_one')

        self.response_events_2, self.response_trials_2 = self.get_stimulus_response()
        self.response_events_1c, self.response_trials_1c = self.get_stimulus_response_one()
        self.peak = self.get_peak()
        self.save_data()

    def get_stimulus_response(self):
        '''Calculates the mean response to the movie

Parameters
----------

Returns
-------
response events: mean response, s.e.m., and number of responsive trials for each movie frame
        '''
        numframes=900
        response_events = np.empty((numframes, self.numbercells,3))
        response_trials = np.empty((numframes, self.numbercells, 10))
        for i in range(numframes):
            starts = self.stim_table_2[self.stim_table_2.frame==i].start
            response_events[i,:,0] = self.l0_events[:,starts].mean(axis=1)
            response_events[i,:,1] = self.l0_events[:,starts].std(axis=1)/np.sqrt(10)
            response_events[i,:,2] = self.l0_events[:,starts].astype(bool).sum(axis=1)
            response_trials[i,:,:] = self.l0_events[:,starts]
        return response_events, response_trials

    def get_stimulus_response_one(self):
        '''Calculates the mean response to the movie

Parameters
----------

Returns
-------
response events: mean response, s.e.m., and number of responsive trials for each movie frame
        '''
        numframes=900
        response_events = np.empty((numframes, self.numbercells,3))
        response_trials = np.empty((numframes, self.numbercells, 10))
        for i in range(numframes):
            starts = self.stim_table_1c[self.stim_table_1c.frame==i].start
            response_events[i,:,0] = self.l0_events[:,starts].mean(axis=1)
            response_events[i,:,1] = self.l0_events[:,starts].std(axis=1)/np.sqrt(10)
            response_events[i,:,2] = self.l0_events[:,starts].astype(bool).sum(axis=1)
            response_trials[i,:,:] = self.l0_events[:,starts]
        return response_events, response_trials

    def get_reliability(self, nc):
        '''computes trial-to-trial reliability of cell to the movie

Parameters
----------
cell index

Returns
-------
reliability metric
        '''
        subset = self.response_trials_2[:,nc,:]
        corr_matrix = np.empty((10,10))
        for i in range(10):
            for j in range(10):
                r,p = st.pearsonr(subset[:,i], subset[:,j])
                corr_matrix[i,j] = r

        inds = np.triu_indices(10, k=1)
        upper_2 = corr_matrix[inds[0],inds[1]]

        subset = self.response_trials_1c[:,nc,:]
        corr_matrix = np.empty((10,10))
        for i in range(10):
            for j in range(10):
                r,p = st.pearsonr(subset[:,i], subset[:,j])
                corr_matrix[i,j] = r

        inds = np.triu_indices(10, k=1)
        upper_1c = corr_matrix[inds[0],inds[1]]

        return np.nanmean(upper_2), np.nanmean(upper_1c)

    def get_peak(self):
        '''creates a table of metrics for each cell

Parameters
----------

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','lifetime_sparseness_nm2','responsive_nm2', 'reliability_nm2', 'peak_frame_nm2', 'lifetime_sparseness_nm1c','responsive_nm1c', 'reliability_nm1c', 'peak_frame_nm1c'), index=range(self.numbercells))
        peak['cell_specimen_id'] = self.specimen_ids
        ltsparseness_2 = ((1-(1/900.)*((np.power(self.response_events_2[:,:,0].sum(axis=0),2))/
                (np.power(self.response_events_2[:,:,0],2).sum(axis=0))))/(1-(1/900.)))
        ltsparseness_1c = ((1-(1/900.)*((np.power(self.response_events_1c[:,:,0].sum(axis=0),2))/
                (np.power(self.response_events_1c[:,:,0],2).sum(axis=0))))/(1-(1/900.)))
        peak['lifetime_sparseness_nm2'] = ltsparseness_2
        peak['lifetime_sparseness_nm1c'] = ltsparseness_1c
        peak['responsive_nm2'] = False
        peak['responsive_nm1c'] = False

        for nc in range(self.numbercells):
            peak.reliability_nm2.iloc[nc], peak.reliability_nm1c.iloc[nc] = self.get_reliability(nc)
            peak.peak_frame_nm2.iloc[nc] = self.response_events_2[:,nc,0].argmax()
            peak.peak_frame_nm1c.iloc[nc] = self.response_events_1c[:,nc,0].argmax()
            if self.response_events_2[self.response_events_2[:,nc,0].argmax(),nc,2]>2:
                peak.responsive_nm2.iloc[nc] = True
            if self.response_events_1c[self.response_events_1c[:,nc,0].argmax(),nc,2]>2:
                peak.responsive_nm1c.iloc[nc] = True

        return peak

    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_nm_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events_2', data=self.response_events)
        dset1 = f.create_dataset('response_events_1c', data=self.response_events)
        dset2 = f.create_dataset('response_trials_2', data=self.response_trials)
        dset3 = f.create_dataset('response_trials_1c', data=self.response_trials)

        f.close()

if __name__=='__main__':
#    session_id = 502667200
#    nm = NaturalMoviesC(session_id=session_id)

    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    fail=[]
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file = manifest_path)
    exp = pd.DataFrame(boc.get_ophys_experiments())
    exp_dg = exp[exp.session_type=='three_session_C'].id.values
    for a in exp_dg:
        try:
            session_id = a
            nm = NaturalMoviesC(session_id=session_id)
        except:
            fail.append(a)
