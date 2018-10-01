#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
import core

def do_sweep_mean(x):
    return x[28:35].mean()

def do_sweep_mean_shifted(x):
    return x[30:40].mean()

class LocallySparseNoise:
    def __init__(self, session_id):
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'LocallySparseNoise')
        self.l0_events = core.get_L0_events(self.session_id)
        #TODO: enable lsn, lsn4, lsn8
        lsn_name = 'locally_sparse_noise'
        
        self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'locally_sparse_noise')
        self.LSN = core.get_stimulus_template('locally_sparse_noise')
        self.stim_table_sp,_,_ = core.get_stim_table(self.session_id, 'spontaneous')
        self.dxcm = core.get_running_speed(self.session_id)
        
        self.sweep_events, self.mean_sweep_events, self.sweep_p_values, self.running_speed, self.response_events, self.response_trials = self.get_stimulus_response()
        self.peak = self.get_peak()
        self.save_data()

    def get_stimulus_response(self):
        '''calculates the response to each stimulus trial. Calculates the mean response to each stimulus condition.

Parameters
----------

Returns
-------
sweep events: full trial for each trial
mean sweep events: mean response for each trial
response events: mean response, s.e.m., and number of responsive trials for each stimulus condition
response trials:

        '''
        sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        running_speed = pd.DataFrame(index=self.stim_table.index.values, columns=('running_speed','null'))
        for index,row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_events[str(nc)][index] = self.l0_events[nc, int(row.start)-28:int(row.start)+35]
            running_speed.running_speed[index] = self.dxcm[int(row.start):int(row.start)+7].mean()
        mean_sweep_events = sweep_events.map(do_sweep_mean_shifted)

        #make spontaneous p_values
        shuffled_responses = np.empty((self.numbercells, 10000,10))
        idx = np.random.choice(range(self.stim_table_sp.start, self.stim_table_sp.end), 10000)
        for i in range(10):
            shuffled_responses[:,:,i] = self.l0_events[:,idx+i]
        shuffled_mean = shuffled_responses.mean(axis=2)
        sweep_p_values = pd.DataFrame(index = self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for nc in range(self.numbercells):
            subset = mean_sweep_events[str(nc)].values
            null_dist_mat = np.tile(shuffled_mean[nc,:], reps=(len(subset),1))
            actual_is_less = subset.reshape(len(subset),1) <= null_dist_mat
            p_values = np.mean(actual_is_less, axis=1)
            sweep_p_values[str(nc)] = p_values

        response_events = np.empty((119,self.numbercells,3))
        response_trials = np.empty((119,self.numbercells,50))
        response_trials[:] = np.NaN

        for im in range(-1,118):
            subset = mean_sweep_events[self.stim_table.frame==im]
            subset_p = sweep_p_values[self.stim_table.frame==im]
            response_events[im+1,:,0] = subset.mean(axis=0)
            response_events[im+1,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
            response_events[im+1,:,2] = subset_p[subset_p<0.05].count().values
            response_trials[im+1,:,:subset.shape[0]] = subset.values.T
        return sweep_events, mean_sweep_events, sweep_p_values, running_speed, response_events, response_trials


    def get_receptive_field(self):
        #TODO: lsn, lsn4, lsn8
        print "Calculating mean responses"
        receptive_field = np.empty((16, 28, self.numbercells, 2))
#        def ptest(x):
#            return len(np.where(x<(0.05/(8*5)))[0])
        for xp in range(16):
            for yp in range(28):
                on_frame = np.where(self.LSN[:,xp,yp]==255)[0]
                off_frame = np.where(self.LSN[:,xp,yp]==0)[0]
                subset_on = self.mean_sweep_response[self.stim_table.Frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[self.stim_table.Frame.isin(off_frame)]
                receptive_field[xp,yp,:,0] = subset_on.mean(axis=0)
                receptive_field[xp,yp,:,1] = subset_off.mean(axis=0)
        return receptive_field