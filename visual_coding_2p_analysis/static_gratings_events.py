# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:39:03 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
from scipy.optimize import curve_fit
import core
import sweep_events_shuffle

def do_sweep_mean(x):
    return x[28:35].mean()

def do_sweep_mean_shifted(x):
    return x[30:40].mean()

class event_analysis(object):
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'SG')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'static_gratings')
        self.stim_table_sp,_,_ = core.get_stim_table(self.session_id, 'spontaneous')
        
class StaticGratings(event_analysis):    
    def __init__(self, *args, **kwargs):
        super(StaticGratings, self).__init__(*args, **kwargs)                   
        self.orivals = range(0,180,30)
        self.sfvals = [0,0.02,0.04,0.08,0.16,0.32]
        self.phasevals = [0,0.25,0.5,0.75]
        self.sweep_events, self.mean_sweep_events, self.sweep_p_values, self.response_events, self.response_trials = self.get_stimulus_response()
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
        sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for index,row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_events[str(nc)][index] = self.l0_events[nc, int(row.start)-28:int(row.start)+35]
#        mean_sweep_events = sweep_events.applymap(do_sweep_mean) 
        mean_sweep_events = sweep_events.applymap(do_sweep_mean_shifted) 
        
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
        
        #make trial p_values
#        sweep_p_values = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
#        for nc in range(self.numbercells):
#            test = np.empty((len(self.stim_table), 7))
#            for i in range(len(self.stim_table)):
#                test[i,:] = sweep_events[str(nc)][i][28:35]
#            sweep_p_values[str(nc)] = sweep_events_shuffle.trial_response_significance(test)
        
        response_events = np.empty((6,6,4,self.numbercells,3))
        response_trials = np.empty((6,6,4,self.numbercells,50))
        response_trials[:] = np.NaN
        for oi, ori in enumerate(self.orivals):
            for si, sf in enumerate(self.sfvals[1:]):
                for phi, phase in enumerate(self.phasevals):
                    subset = mean_sweep_events[(self.stim_table.orientation==ori)&(self.stim_table.spatial_frequency==sf)&(self.stim_table.phase==phase)]
                    subset_p = sweep_p_values[(self.stim_table.orientation==ori)&(self.stim_table.spatial_frequency==sf)&(self.stim_table.phase==phase)]
                    response_events[oi,si+1,phi,:,0] = subset.mean(axis=0)
                    response_events[oi,si+1,phi,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
#                    response_events[oi,si+1,phi,:,2] = subset[subset>0].count().values
                    response_events[oi,si+1,phi,:,2] = subset_p[subset_p<0.05].count().values
                    response_trials[oi,si+1,phi,:,:subset.shape[0]] = subset.values.T        
        subset = mean_sweep_events[np.isnan(self.stim_table.orientation)]
        subset_p = sweep_p_values[np.isnan(self.stim_table.orientation)]
        response_events[0,0,0,:,0] = subset.mean(axis=0)
        response_events[0,0,0,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
        response_events[0,0,0,:,2] = subset_p[subset_p<0.05].count().values
        return sweep_events, mean_sweep_events, sweep_p_values, response_events, response_trials

    def get_lifetime_sparseness(self):
        '''computes lifetime sparseness of responses for all cells

Returns
-------
lifetime sparseness
        '''
        response = self.response_events[:,1:,:,:,0].reshape(120, self.numbercells)
        return ((1-(1/120.)*((np.power(response.sum(axis=0),2))/(np.power(response,2).sum(axis=0))))/(1-(1/120.)))

    
    def get_osi(self, pref_sf, pref_phase, nc):
        '''computes orientation selectivity (cv) for cell

Parameters
----------
preferred spatial frequency
preferred phase
cell index

Returns
-------
orientation selectivity
        '''
        orivals_rad = np.deg2rad(self.orivals)
        tuning = self.response_events[:, pref_sf+1, pref_phase, nc, 0]
        CV_top_os = np.empty((6), dtype=np.complex128)
        for i in range(6):
            CV_top_os[i] = (tuning[i]*np.exp(1j*2*orivals_rad[i]))
        return np.abs(CV_top_os.sum())/tuning.sum()
    
    def get_reliability(self, pref_ori, pref_sf, pref_phase, nc):
        '''computes trial-to-trial reliability of cell at its preferred condition

Parameters
----------
preferred orientation
preferred spatial frequency
preferred phase
cell index

Returns
-------
reliability metric
        '''
        subset = self.sweep_events[(self.stim_table.spatial_frequency==self.sfvals[pref_sf])
                                     &(self.stim_table.orientation==self.orivals[pref_ori])&(self.stim_table.phase==self.phasevals[pref_phase])]         
        corr_matrix = np.empty((len(subset),len(subset)))
        for i in range(len(subset)):
            for j in range(len(subset)):
                r,p = st.pearsonr(subset[str(nc)].iloc[i][28:35], subset[str(nc)].iloc[j][28:35])
                corr_matrix[i,j] = r
                
        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0],inds[1]]
        return np.nanmean(upper)
        
    def get_sfdi(self, pref_ori, pref_phase, nc):
        '''computes spatial frequency discrimination index for cell

Parameters
----------
preferred orientation
preferred phase
cell index

Returns
-------
sf discrimination index
        '''
        sf_tuning = self.response_events[pref_ori,1:,pref_phase,nc,0]
        trials = self.mean_sweep_events[(self.stim_table.orientation==self.orivals[pref_ori])&(self.stim_table.phase==self.phasevals[pref_phase])][str(nc)].values
        SSE_part = np.sqrt(np.sum((trials-trials.mean())**2)/(len(trials)-5))
        return (np.ptp(sf_tuning))/(np.ptp(sf_tuning) + 2*SSE_part)

    def fit_sf_tuning(self, pref_ori, pref_sf, pref_phase, nc):
        '''performs gaussian or exponential fit on the spatial frequency tuning curve at preferred orientation/phase.

Parameters
----------
preferred orientation
preferred spatial frequency
preferred phase
cell index

Returns
-------
index for the preferred sf from the curve fit
prefered sf from the curve fit
low cutoff sf from the curve fit
high cutoff sf from the curve fit
        '''
        sf_tuning = self.response_events[pref_ori,1:,pref_phase,nc,0]
        fit_sf_ind = np.NaN
        fit_sf = np.NaN
        sf_low_cutoff = np.NaN
        sf_high_cutoff = np.NaN
        if pref_sf in range(1,4):
            try:
                popt, pcov = curve_fit(core.gauss_function, range(5), sf_tuning, p0=[np.amax(sf_tuning), pref_sf, 1.], maxfev=2000)
                sf_prediction = core.gauss_function(np.arange(0., 4.1, 0.1), *popt)
                fit_sf_ind = popt[1]
                fit_sf = 0.02*np.power(2,popt[1])
                low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin()+sf_prediction.argmax()                         
                if low_cut_ind>0:
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)
                elif high_cut_ind<49:
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
            except:
                pass 
        else:
            fit_sf_ind = pref_sf
            fit_sf = self.sfvals[pref_sf+1]
            try:
                popt, pcov = curve_fit(core.exp_function, range(5), sf_tuning, p0=[np.amax(sf_tuning), 2., np.amin(sf_tuning)], maxfev=2000)
                sf_prediction = core.exp_function(np.arange(0., 4.1, 0.1), *popt)
                if pref_sf==0:
                    high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin()+sf_prediction.argmax()
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
                else:
                    low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)                
            except:
                pass
        return fit_sf_ind, fit_sf, sf_low_cutoff, sf_high_cutoff

    def get_peak(self):
        '''creates a table of metrics for each cell

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','pref_ori_sg','pref_sf_sg','pref_phase_sg','num_pref_trials_sg',
                                     'responsive_sg','g_osi_sg','sfdi_sg','reliability_sg','lifetime_sparseness_sg', 'fit_sf_sg','fit_sf_ind_sg',
                                     'sf_low_cutoff_sg','sf_high_cutoff_sg'), index=range(self.numbercells))
        
        peak['lifetime_sparseness_sg'] = self.get_lifetime_sparseness()
        for nc in range(self.numbercells):
            pref_ori = np.where(self.response_events[:,1:,:,nc,0]==self.response_events[:,1:,:,nc,0].max())[0][0]
            pref_sf = np.where(self.response_events[:,1:,:,nc,0]==self.response_events[:,1:,:,nc,0].max())[1][0]
            pref_phase = np.where(self.response_events[:,1:,:,nc,0]==self.response_events[:,1:,:,nc,0].max())[2][0]
            peak.cell_specimen_id.iloc[nc] = self.specimen_ids[nc]
            peak.pref_ori_sg.iloc[nc] = self.orivals[pref_ori]
            peak.pref_sf_sg.iloc[nc] = self.sfvals[pref_sf+1]
            peak.pref_phase_sg.iloc[nc] = self.phasevals[pref_phase]
            peak.num_pref_trials_sg.iloc[nc] = self.response_events[pref_ori, pref_sf+1,pref_phase, nc,2]
            if self.response_events[pref_ori, pref_sf+1,pref_phase,nc,2]>11:
                peak.responsive_sg.iloc[nc] = True
            else:
                peak.responsive_sg.iloc[nc] = False
            
            peak.g_osi_sg.iloc[nc] = self.get_osi(pref_sf, pref_phase, nc)            
            peak.reliability_sg.iloc[nc] = self.get_reliability(pref_ori, pref_sf, pref_phase, nc)
            peak.sfdi_sg.iloc[nc] = self.get_sfdi(pref_ori, pref_phase, nc)
                          
            #SF fit only for responsive cells
            if self.response_events[pref_ori, pref_sf+1,pref_phase,nc,2]>11:
                peak.fit_sf_ind_sg.iloc[nc], peak.fit_sf_sg.iloc[nc], peak.sf_low_cutoff_sg.iloc[nc], peak.sf_high_cutoff_sg.iloc[nc] = self.fit_sf_tuning(pref_ori, pref_sf, pref_phase, nc)
        return peak
    
    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_sg_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['sweep_events'] = self.sweep_events
        store['mean_sweep_events'] = self.mean_sweep_events
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events', data=self.response_events)
        dset1 = f.create_dataset('response_trials', data=self.response_trials)
        f.close()
        
if __name__=='__main__':
#    session_id = 511458874
#    sg = StaticGratings(session_id=session_id)

    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    fail=[]
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file = manifest_path)
    exp = pd.DataFrame(boc.get_ophys_experiments())
    exp_dg = exp[exp.session_type=='three_session_B'].id.values
    for a in exp_dg:
        try:
            session_id = a
            sg = StaticGratings(session_id=session_id)
        except:
            fail.append(a)