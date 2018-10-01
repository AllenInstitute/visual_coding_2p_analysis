#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
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
        self.stim_table_sp, _, _ = core.get_stim_table(self.session_id, 'spontaneous')
        self.dxcm = core.get_running_speed(self.session_id)
        try:
            lsn_name = 'locally_sparse_noise'
            self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, lsn_name)
            self.LSN = core.get_stimulus_template(self.session_id, lsn_name)
            self.sweep_events, self.mean_sweep_events, self.sweep_p_values, self.running_speed, self.response_events_on, self.response_events_off = self.get_stimulus_response(self.LSN)
            self.peak = self.get_peak()
        except:
            lsn_name = 'locally_sparse_noise_4deg'
            self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, lsn_name)
            self.LSN_4deg = core.get_stimulus_template(self.session_id, lsn_name)
            self.sweep_events_4deg, self.mean_sweep_events_4deg, self.sweep_p_values_4deg, self.running_speed_4deg, self.response_events_on_4deg, self.response_events_off_4deg = self.get_stimulus_response(self.LSN_4deg)
            self.peak = self.get_peak()
            
            lsn_name = 'locally_sparse_noise_8deg'
            self.stim_table, _, _ = core.get_stim_table(self.session_id, lsn_name)
            self.LSN_8deg = core.get_stimulus_template(self.session_id, lsn_name)
            self.sweep_events_8deg, self.mean_sweep_events_8deg, self.sweep_p_values_8deg, self.running_speed_8deg, self.response_events_on_8deg, self.response_events_off_8deg = self.get_stimulus_response(self.LSN_8deg)
            self.peak = self.get_peak()
        
        self.save_data()

    def get_stimulus_response(self, LSN):
        '''calculates the response to each stimulus trial. Calculates the mean response to each stimulus condition.

Returns
-------
sweep events: full trial for each trial
mean sweep events: mean response for each trial
sweep p values: p value of each trial compared measured relative to distribution of spontaneous activity
running speed: mean running speed per trial
response events_on: mean response, s.e.m., and number of responsive trials for each white square
response events_off: mean response, s.e.m., and number of responsive trials for each black square


        '''
        sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        running_speed = pd.DataFrame(index=self.stim_table.index.values, columns=('running_speed','null'))
        for index,row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_events[str(nc)][index] = self.l0_events[nc, int(row.start)-28:int(row.start)+35]
            running_speed.running_speed[index] = self.dxcm[int(row.start):int(row.start)+7].mean()

        mean_sweep_events = sweep_events.applymap(do_sweep_mean_shifted)

        #make spontaneous p_values
        shuffled_responses = np.empty((self.numbercells, 10000,10))
        idx = np.random.choice(range(self.stim_table_sp.start[0], self.stim_table_sp.end[0]), 10000)
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
        
        x_shape = LSN.shape[1]
        y_shape = LSN.shape[2]
        response_events_on = np.empty((x_shape, y_shape, self.numbercells, 3))
        response_events_off = np.empty((x_shape, y_shape, self.numbercells, 3))
        for xp in range(x_shape):
            for yp in range(y_shape):
                on_frame = np.where(self.LSN[:,xp,yp]==255)[0]
                off_frame = np.where(self.LSN[:,xp,yp]==0)[0]
                subset_on = mean_sweep_events[self.stim_table.frame.isin(on_frame)]
                subset_on_p = sweep_p_values[self.stim_table.frame.isin(on_frame)]
                subset_off = mean_sweep_events[self.stim_table.frame.isin(off_frame)]
                subset_off_p = sweep_p_values[self.stim_table.frame.isin(off_frame)]
                response_events_on[xp,yp,:,0] = subset_on.mean(axis=0)
                response_events_on[xp,yp,:,1] = subset_on.std(axis=0)/np.sqrt(len(subset_on))
                response_events_on[xp,yp,:,2] = subset_on_p[subset_on_p<0.05].count().values/float(len(subset_on_p))
                response_events_off[xp,yp,:,0] = subset_off.mean(axis=0)
                response_events_off[xp,yp,:,1] = subset_off.std(axis=0)/np.sqrt(len(subset_off))
                response_events_off[xp,yp,:,2] = subset_off_p[subset_off_p<0.05].count().values/float(len(subset_off_p))
        return sweep_events, mean_sweep_events, sweep_p_values, running_speed, response_events_on, response_events_off
    
    def get_peak(self):
        '''creates a table of metrics for each cell

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','rf_on','rf_off','num_on_pixels','num_off_pixels','on_pixels','off_pixels'), index=range(self.numbercells))
        peak['rf_on'] = False
        peak['rf_off'] = False
        peak.cell_specimen_id = self.specimen_ids
        on_rfs = np.where(self.response_events_on[:,:,:,2]>0.25)[2]
        off_rfs = np.where(self.response_events_off[:,:,:,2]>0.25)[2]
        peak.rf_on.loc[on_rfs] = True 
        peak.rf_off.loc[off_rfs] = True 
        for nc in on_rfs:
            peak.num_on_pixels.loc[nc] = len(np.where(self.response_events_on[:,:,nc,2]>0.25)[0])
            peak.on_pixels.loc[nc] = np.where(self.response_events_on[:,:,nc,2]>0.25)
        for nc in off_rfs:
            peak.num_off_pixels.loc[nc] = len(np.where(self.response_events_off[:,:,nc,2]>0.25)[0])
            peak.off_pixels.loc[nc] = np.where(self.response_events_off[:,:,nc,2]>0.25)    
        return peak

    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_lsn_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['sweep_events'] = self.sweep_events
        store['mean_sweep_events'] = self.mean_sweep_events
        store['sweep_p_values'] = self.sweep_p_values
        store['running_speed'] = self.running_speed
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events_on', data=self.response_events_on)
        dset1 = f.create_dataset('response_events_off', data=self.response_events_off)
        f.close()
        
if __name__=='__main__':
    session_id = 569611979
    lsn = LocallySparseNoise(session_id=session_id)