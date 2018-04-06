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
#         self.stim_table_1b, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_movie_one')
#
#
# class NaturalMoviesB(event_analysis):
#     def __init__(self, *args, **kwargs):
#         super(NaturalMoviesB, self).__init__(*args, **kwargs)

class NaturalMoviesB:
    def __init__(self, session_id):
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'NaturalMovies')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table_1b, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_movie_one')

        self.response_events_1b, self.response_trials_1b = self.get_stimulus_response_one()
        self.peak = self.get_peak()
        self.save_data()

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
            starts = self.stim_table_1b[self.stim_table_1b.frame==i].start
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
        subset = self.response_trials_1b[:,nc,:]
        corr_matrix = np.empty((10,10))
        for i in range(10):
            for j in range(10):
                r,p = st.pearsonr(subset[:,i], subset[:,j])
                corr_matrix[i,j] = r

        inds = np.triu_indices(10, k=1)
        upper_1b = corr_matrix[inds[0],inds[1]]

        return np.nanmean(upper_1b)

    def get_peak(self):
        '''creates a table of metrics for each cell

Parameters
----------

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','lifetime_sparseness_nm1b','responsive_nm1b', 'reliability_nm1b', 'peak_frame_nm1b'), index=range(self.numbercells))
        peak['cell_specimen_id'] = self.specimen_ids
        ltsparseness_1b = ((1-(1/900.)*((np.power(self.response_events_1b[:,:,0].sum(axis=0),2))/
                (np.power(self.response_events_1b[:,:,0],2).sum(axis=0))))/(1-(1/900.)))
        peak['lifetime_sparseness_nm1b'] = ltsparseness_1b
        peak['responsive_nm1b'] = False

        for nc in range(self.numbercells):
            peak.reliability_nm1b.iloc[nc] = self.get_reliability(nc)
            peak.peak_frame_nm1b.iloc[nc] = self.response_events_1b[:,nc,0].argmax()
            if self.response_events_1b[self.response_events_1b[:,nc,0].argmax(),nc,2]>2:
                peak.responsive_nm1b.iloc[nc] = True

        return peak

    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_nm_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events_1b', data=self.response_events_1b)
        dset1 = f.create_dataset('response_trials_1b', data=self.response_trials_1b)
        f.close()


if __name__=='__main__':
#    session_id = 511458874
#    nm = NaturalMoviesB(session_id=session_id)

    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    fail=[]
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file = manifest_path)
    exp = pd.DataFrame(boc.get_ophys_experiments())
    exp_dg = exp[exp.session_type=='three_session_B'].id.values
    for a in exp_dg:
        try:
            session_id = a
            nm = NaturalMoviesB(session_id=session_id)
        except:
            fail.append(a)
