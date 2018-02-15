# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:53:07 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
from scipy.optimize import curve_fit
import core

def do_sweep_mean(x):
    return x[30:].mean()

def do_intersweep_mean(x):
    return x[:30].mean()

class event_analysis(object):
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'Drifting Gratings')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'drifting_gratings')

        
class DriftingGratings(event_analysis):    
    def __init__(self, *args, **kwargs):
        super(DriftingGratings, self).__init__(*args, **kwargs)                   
        self.orivals = range(0,360,45)
        self.tfvals = [0,1,2,4,8,15]
        self.sweep_events, self.mean_sweep_events, self.response_events, self.response_trials = self.get_stimulus_response()
        self.peak = self.get_peak()
        self.save_data()
    
    def get_stimulus_response(self):
        '''calculates the response to each stimulus trial. Calculates the mean response to each stimulus condition.

Returns
-------
sweep events: full trial for each trial
mean sweep events: mean response for each trial
response events: mean response, s.e.m., and number of responsive trials for each stimulus condition
response trials:         
        
        '''
        print "Computing responses"
        #make sweep_response with events
        sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for ind,row_stim in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_events[str(nc)][ind] = self.l0_events[nc, int(row_stim.start)-30:int(row_stim.start)+60]
        mean_sweep_events = sweep_events.applymap(do_sweep_mean)
    
        #make response array with events
        response_events = np.empty((8,6,self.numbercells,3))
        response_trials = np.empty((8,6,self.numbercells,15))
        response_trials[:] = np.NaN
        subset = mean_sweep_events[np.isnan(self.stim_table.orientation)]
        response_events[0,0,:,0] = subset.mean(axis=0)
        response_events[0,0,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
        response_events[0,0,:,2] = subset[subset>0].count().values

        for oi, ori in enumerate(self.orivals):
            for ti, tf in enumerate(self.tfvals[1:]):
                subset = mean_sweep_events[(self.stim_table.orientation==ori)&(self.stim_table.temporal_frequency==tf)]
                response_events[oi,ti+1,:,0] = subset.mean(axis=0)
                response_events[oi,ti+1,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
                response_events[oi,ti+1,:,2] = subset[subset>0].count().values
                response_trials[oi,ti+1,:,:subset.shape[0]] = subset.values.T
        subset = mean_sweep_events[np.isnan(self.stim_table.orientation)]
        response_events[0,0,:,0] = subset.mean(axis=0)
        response_events[0,0,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
        response_events[0,0,:,2] = subset[subset>0].count().values
        #what to do about blank sweeps for the response_trials?
        return sweep_events, mean_sweep_events, response_events, response_trials
    
    def get_lifetime_sparseness(self):
        '''computes lifetime sparseness of responses for all cells

Returns
-------
lifetime sparseness
        '''
        response = self.response_events[:,1:,:,0].reshape(40, self.numbercells)
        return ((1-(1/40.)*((np.power(response.sum(axis=0),2))/(np.power(response,2).sum(axis=0))))/(1-(1/40.)))
    
    def get_osi(self, pref_tf, nc):
        '''computes orientation and direction selectivity (cv) for cell

Parameters
----------
preferred temporal frequency
cell index

Returns
-------
orientation selectivity
direction selectivity
        '''
        orivals_rad = np.deg2rad(self.orivals)
        tuning = self.response_events[:, pref_tf+1, nc, 0]
        tuning = np.where(tuning>0, tuning, 0)
        CV_top_os = np.empty((8), dtype=np.complex128)
        CV_top_ds = np.empty((8), dtype=np.complex128)
        for i in range(8):
            CV_top_os[i] = (tuning[i]*np.exp(1j*2*orivals_rad[i]))
            CV_top_ds[i] = (tuning[i]*np.exp(1j*orivals_rad[i]))
        osi = np.abs(CV_top_os.sum())/tuning.sum()
        dsi = np.abs(CV_top_ds.sum())/tuning.sum()
        return osi, dsi

    def get_reliability(self, pref_ori, pref_tf, nc):
        '''computes trial-to-trial reliability of cell at its preferred condition

Parameters
----------
preferred orientation
preferred temporal frequency
cell index

Returns
-------
reliability metric
        '''
        subset = self.sweep_events[(self.stim_table.temporal_frequency==self.tfvals[pref_tf+1])
                                         &(self.stim_table.orientation==self.orivals[pref_ori])]         
        corr_matrix = np.empty((len(subset),len(subset)))
        for i in range(len(subset)):
            for j in range(len(subset)):
                r,p = st.pearsonr(subset[str(nc)].iloc[i][30:], subset[str(nc)].iloc[j][30:])
                corr_matrix[i,j] = r
                
        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0],inds[1]]
        return np.nanmean(upper)
                    #TODO: why are some reliability values NaN?
    
    def get_tfdi(self, pref_ori, nc):
        '''computes temporal frequency discrimination index for cell

Parameters
----------
preferred orientation
cell index

Returns
-------
tf discrimination index
        '''
        tf_tuning = self.response_events[pref_ori,1:,nc,0]
        trials = self.mean_sweep_events[(self.stim_table.orientation==self.orivals[pref_ori])][str(nc)].values
        SSE_part = np.sqrt(np.sum((trials-trials.mean())**2)/(len(trials)-5))
        return (np.ptp(tf_tuning))/(np.ptp(tf_tuning) + 2*SSE_part)

    def fit_tf_tuning(self, pref_ori, pref_tf, nc):
        '''performs gaussian or exponential fit on the temporal frequency tuning curve at preferred orientation.

Parameters
----------
preferred orientation
preferred temporal frequency
cell index

Returns
-------
index for the preferred tf from the curve fit
prefered tf from the curve fit
low cutoff tf from the curve fit
high cutoff tf from the curve fit
        '''
        tf_tuning = self.response_events[pref_ori,1:,nc,0]
        fit_tf_ind = np.NaN
        fit_tf = np.NaN
        tf_low_cutoff = np.NaN
        tf_high_cutoff = np.NaN
        if pref_tf in range(1,4):
            try:
                popt, pcov = curve_fit(core.gauss_function, range(5), tf_tuning, p0=[np.amax(tf_tuning), pref_tf, 1.], maxfev=2000)
                tf_prediction = core.gauss_function(np.arange(0., 4.1, 0.1), *popt)             
                fit_tf_ind = popt[1]
                fit_tf = np.power(2,popt[1])
                low_cut_ind = np.abs(tf_prediction-(tf_prediction.max()/2.))[:tf_prediction.argmax()].argmin()
                high_cut_ind = np.abs(tf_prediction-(tf_prediction.max()/2.))[tf_prediction.argmax():].argmin()+tf_prediction.argmax()                         
                if low_cut_ind>0:
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    tf_low_cutoff = np.power(2,low_cutoff)
                elif high_cut_ind<49:
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    tf_high_cutoff = np.power(2,high_cutoff)
            except:
                pass 
        else:
            fit_tf_ind = pref_tf
            fit_tf = self.tfvals[pref_tf+1]
            try:
                popt, pcov = curve_fit(core.exp_function, range(5), tf_tuning, p0=[np.amax(tf_tuning), 2., np.amin(tf_tuning)], maxfev=2000)
                tf_prediction = core.exp_function(np.arange(0., 4.1, 0.1), *popt)
                if pref_tf==0:
                    high_cut_ind = np.abs(tf_prediction-(tf_prediction.max()/2.))[tf_prediction.argmax():].argmin()+tf_prediction.argmax()
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    tf_high_cutoff = np.power(2,high_cutoff)
                else:
                    low_cut_ind = np.abs(tf_prediction-(tf_prediction.max()/2.))[:tf_prediction.argmax()].argmin()
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    tf_low_cutoff = np.power(2,low_cutoff)                
            except:
                pass
        return fit_tf_ind, fit_tf, tf_low_cutoff, tf_high_cutoff



    def get_peak(self):
        '''creates a table of metrics for each cell

Returns
-------
peak dataframe
        '''
        print "Computing metrics"
        peak = pd.DataFrame(columns=('cell_specimen_id','pref_ori_dg','pref_tf_dg','num_pref_trials_dg','responsive_dg',
                                     'g_osi_dg','g_dsi_dg','tfdi_dg','reliability_dg','lifetime_sparseness_dg', 
                                     'fit_tf_dg','fit_tf_ind_dg','tf_low_cutoff_dg','tf_high_cutoff_dg'), index=range(self.numbercells))
        
        peak['lifetime_sparseness_dg'] = self.get_lifetime_sparseness()
        for nc in range(self.numbercells):
            pref_ori = np.where(self.response_events[:,1:,nc,0]==self.response_events[:,1:,nc,0].max())[0][0]
            pref_tf = np.where(self.response_events[:,1:,nc,0]==self.response_events[:,1:,nc,0].max())[1][0]
            peak.cell_specimen_id.iloc[nc] = self.specimen_ids[nc]
            peak.pref_ori_dg.iloc[nc] = self.orivals[pref_ori]
            peak.pref_tf_dg.iloc[nc] = self.tfvals[pref_tf+1]
    
            #responsive
            peak.num_pref_trials_dg.iloc[nc] = self.response_events[pref_ori, pref_tf+1, nc, 2]
            if self.response_events[pref_ori, pref_tf+1, nc, 2]>3:
                peak.responsive_dg.iloc[nc] = True
            else:
                peak.responsive_dg.iloc[nc] = False
            peak.g_osi_dg.iloc[nc], peak.g_dsi_dg.iloc[nc] = self.get_osi(pref_tf, nc)
            peak.reliability_dg.iloc[nc] = self.get_reliability(pref_ori, pref_tf, nc)
            peak.tfdi_dg.iloc[nc] = self.get_tfdi(pref_ori, nc)
            if self.response_events[pref_ori, pref_tf+1,nc,2]>3:
                peak.fit_tf_ind_dg.iloc[nc], peak.fit_tf_dg.iloc[nc], peak.tf_low_cutoff_dg.iloc[nc], peak.tf_high_cutoff_dg.iloc[nc] = self.fit_tf_tuning(pref_ori, pref_tf, nc)
            
        return peak
    
    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_dg_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['sweep_events'] = self.sweep_events
        store['mean_sweep_events'] = self.mean_sweep_events
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        data = f['response_events']       # load the data
        data[...] = self.response_events
        data1 = f['response_trials']
        data1[...] = self.response_trials
#        dset = f.create_dataset('response_events', data=self.response_events)
#        dset1 = f.create_dataset('response_trials', data=self.response_trials)
        f.close()


if __name__=='__main__':
    session_id = 511595995
    dg = DriftingGratings(session_id=session_id)
    
