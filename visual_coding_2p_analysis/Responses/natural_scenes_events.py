# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:08:15 2018

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

# class event_analysis(object):
#     def __init__(self, *args, **kwargs):
#         for k,v in kwargs.iteritems():
#             setattr(self, k, v)
#         self.session_id = session_id
#         save_path_head = core.get_save_path()
#         self.save_path = os.path.join(save_path_head, 'Natural Scenes')
#         self.l0_events = core.get_L0_events(self.session_id)
#         self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_scenes')
#         self.stim_table_sp,_,_ = core.get_stim_table(self.session_id, 'spontaneous')
#         self.dxcm = core.get_running_speed(self.session_id)
#
# class NaturalScenes(event_analysis):
#     def __init__(self, *args, **kwargs):
#       super(NaturalScenes, self).__init__(*args, **kwargs)


class NaturalScenes:
    def __init__(self, session_id):
        self.session_id = session_id
        save_path_head = core.get_save_path()
        self.save_path = os.path.join(save_path_head, 'NaturalScenes')
        self.l0_events = core.get_L0_events(self.session_id)
        self.stim_table, self.numbercells, self.specimen_ids = core.get_stim_table(self.session_id, 'natural_scenes')
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

    def get_image_selectivity(self, nc):
        '''calculates the image selectivity for cell

Parameters
----------
cell index

Returns
-------
image seletivity
        '''
        fmin = self.response_events[1:,nc,0].min()
        fmax = self.response_events[1:,nc,0].max()
        rtj = np.empty((1000,1))
        for j in range(1000):
            thresh = fmin + j*((fmax-fmin)/1000.)
            theta = np.empty((118,1))
            for im in range(118):
                if self.response_events[im+1,nc,0] > thresh:  #im+1 to only look at images, not blanksweep
                    theta[im] = 1
                else:
                    theta[im] = 0
            rtj[j] = theta.mean()
        biga = rtj.mean()
        return 1 - (2*biga)

    def get_reliability(self, pref_image, nc):
        '''computes trial-to-trial reliability of cell at its preferred condition

Parameters
----------
preferred image
cell index

Returns
-------
reliability metric
        '''
        subset = self.sweep_events[(self.stim_table.frame==pref_image)]
        corr_matrix = np.empty((len(subset),len(subset)))
        for i in range(len(subset)):
            for j in range(len(subset)):
                r,p = st.pearsonr(subset[str(nc)].iloc[i][28:35], subset[str(nc)].iloc[j][28:35])
                corr_matrix[i,j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0],inds[1]]
        return np.nanmean(upper)

    def get_running_modulation(self, pref_image, nc):
        '''computes running modulation of cell at its preferred condition provided there are at
        least 2 trials for both stationary and running conditions

Parameters
----------
preferred image

Returns
-------
p_value of running modulation
running modulation metric
mean response to preferred condition when running
mean response to preferred condition when stationary
        '''
        subset = self.mean_sweep_events[(self.stim_table.frame==pref_image)]
        speed_subset = self.running_speed[(self.stim_table.frame==pref_image)]

        subset_run = subset[speed_subset.running_speed>=1]
        subset_stat = subset[speed_subset.running_speed<1]
        if np.logical_and(len(subset_run)>1, len(subset_stat)>1):
            run = subset_run[str(nc)].mean()
            stat = subset_stat[str(nc)].mean()
            if run > stat:
                run_mod = (run - stat)/run
            elif stat > run:
                run_mod = -1 * (stat - run)/stat
            else:
                run_mod = 0
            (_,p) = st.ttest_ind(subset_run[str(nc)], subset_stat[str(nc)], equal_var=False)
            return p, run_mod, run, stat
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN

    def get_peak(self):
        '''creates a table of metrics for each cell

Parameters
----------

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('cell_specimen_id','pref_image_ns','num_pref_trials_ns','responsive_ns',
                                     'image_selectivity_ns','reliability_ns','lifetime_sparseness_ns',
                                     'run_pval_ns','run_mod_ns','run_resp_ns','stat_resp_ns'), index=range(self.numbercells))
        for nc in range(self.numbercells):
            pref_image = np.where(self.response_events[1:,nc,0]==self.response_events[1:,nc,0].max())[0][0]
            peak.cell_specimen_id.iloc[nc] = self.specimen_ids[nc]
            peak.pref_image_ns.iloc[nc] = pref_image
            peak.num_pref_trials_ns.iloc[nc] = self.response_events[pref_image+1,nc,2]
            if self.response_events[pref_image+1,nc,2]>11:
                peak.responsive_ns.iloc[nc] = True
            else:
                peak.responsive_ns.iloc[nc] = False
            peak.image_selectivity_ns.iloc[nc] = self.get_image_selectivity(nc)
            peak.reliability_ns.iloc[nc] = self.get_reliability(pref_image, nc)
            peak.run_pval_ns.iloc[nc], peak.run_mod_ns.iloc[nc], peak.run_resp_ns.iloc[nc], peak.stat_resp_ns.iloc[nc] = self.get_running_modulation(pref_image, nc)

        peak['lifetime_sparseness_ns'] = ((1-(1/118.)*((np.power(self.response_events[:,:,0].sum(axis=0),2))/
                                  (np.power(self.response_events[:,:,0],2).sum(axis=0))))/(1-(1/118.)))
        return peak

    def save_data(self):
        save_file = os.path.join(self.save_path, str(self.session_id)+"_ns_events_analysis.h5")
        print "Saving data to: ", save_file
        store = pd.HDFStore(save_file)
        store['sweep_events'] = self.sweep_events
        store['mean_sweep_events'] = self.mean_sweep_events
        store['sweep_p_values'] = self.sweep_p_values
        store['running_speed'] = self.running_speed
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_events', data=self.response_events)
        dset1 = f.create_dataset('response_trials', data=self.response_trials)
        f.close()

if __name__=='__main__':
    session_id = 511458874
    ns = NaturalScenes(session_id=session_id)
#
#    session_id = sys.argv[1]
#    ns = NaturalScenes(session_id=session_id)

#    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
#    fail=[]
#    manifest_path = core.get_manifest_path()
#    boc = BrainObservatoryCache(manifest_file = manifest_path)
#    exp = pd.DataFrame(boc.get_ophys_experiments())
#    exp_dg = exp[exp.session_type=='three_session_B'].id.values
#    for a in exp_dg:
#        try:
#            session_id = a
#            ns = NaturalScenes(session_id=session_id)
#        except:
#            fail.append(a)
