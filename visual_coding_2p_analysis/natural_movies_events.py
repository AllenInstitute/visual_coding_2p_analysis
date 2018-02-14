# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:21:16 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
import core


class event_analysis(object):
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'Natural Movies')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_movie_three')
    
        
class NaturalMovies(event_analysis):    
    def __init__(self, *args, **kwargs):
        super(NaturalMovies, self).__init__(*args, **kwargs) 
        self.response_events, self.response_trials = self.get_stimulus_response()
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
        response_events = np.empty((3600, self.numbercells,3))
        response_trials = np.empty((3600, self.numbercells, 10))
        for i in range(3600):
            starts = self.stim_table[self.stim_table.frame==i].start
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
        subset = self.response_trials[:,nc,:]
        corr_matrix = np.empty((10,10))
        for i in range(10):
            for j in range(10):
                r,p = st.pearsonr(subset[:,i], subset[:,j])
                corr_matrix[i,j] = r
                
        inds = np.triu_indices(10, k=1)
        upper = corr_matrix[inds[0],inds[1]]
        return np.nanmean(upper)

    def get_peak(self):
        '''creates a table of metrics for each cell

Parameters
----------

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','lifetime_sparseness_nm3','responsive_nm3', 'reliability_nm3'), index=range(self.numbercells))
        peak['cell_specimen_id'] = self.specimen_ids
        ltsparseness = ((1-(1/3600.)*((np.power(self.response_events[:,:,0].sum(axis=0),2))/
                (np.power(self.response_events[:,:,0],2).sum(axis=0))))/(1-(1/3600.)))
        peak['lifetime_sparseness_nm3'] = ltsparseness
        peak['responsive_nm3'] = False
        peak['responsive_nm3'][np.where(self.response_events[:,:,2].max(axis=0)>2)[0]] = True        
        for nc in range(self.numbercells):
            peak.reliability_nm3.iloc[nc] = self.get_reliability(nc)
        return peak
    
    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_nm_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events', data=self.response_events)
        dset2 = f.create_dataset('response_trials', data=self.response_trials)
        f.close()

if __name__=='__main__':
    session_id = 511595995
    nm = NaturalMovies(session_id=session_id)
    