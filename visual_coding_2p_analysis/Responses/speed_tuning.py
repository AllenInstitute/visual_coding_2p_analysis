# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 17:01:43 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import core

def Speed_Tuning(session_id, binsize=900):
    save_path_head = core.get_save_path()
    save_path = os.path.join(save_path_head, 'SpeedTuning')
    l0_events = core.get_L0_events(session_id)
    dxcm = core.get_running_speed(session_id)
    _, numbercells, specimen_ids = core.get_stim_table(session_id, 'spontaneous')
#    numbercells = l0_events.shape
    numbins = 1+len(dxcm[np.where(dxcm>1)]/binsize)
    #remove any NaNs from running and activity traces
    dx_trim = dxcm[~np.isnan(dxcm)]
    l0_trim = l0_events[:,~np.isnan(dxcm)]
    #rank sort traces by running speed
    dx_sorted = dx_trim[np.argsort(dx_trim)]
    events_sorted = l0_trim[:,np.argsort(dx_trim)]
    #bin running and activity
    binned_cells = np.zeros((numbercells, numbins, 2))
    binned_dx = np.zeros((numbins, 2))
    for i in range(numbins):
        offset = np.were(dx_sorted>1)[0][0]
        if i==0:
            binned_dx[i,0] = np.mean(dx_sorted[:offset])
            binned_dx[i,1] = np.std(dx_sorted[:offset])/np.sqrt(offset)
            binned_cells[:,i,0] = np.mean(events_sorted[:,:offset], axis=1)
            binned_cells[:,i,1] = np.std(events_sorted[:,:offset], axis=1)/np.sqrt(offset)
        else:
            start = offset + (i-1)*binsize
            binned_dx[i,0] = np.mean(dx_sorted[start:start+binsize])
            binned_dx[i,1] = np.std(dx_sorted[start:start+binsize])/np.sqrt(binsize)
            binned_cells[:,i,0] = np.mean(events_sorted[:,start:start+binsize], axis=1)
            binned_cells[:,i,1] = np.std(events_sorted[:,start:start+binsize], axis=1)/np.sqrt(binsize)
    #shuffled activity to get significance
    binned_cells_shuffled = np.empty((numbercells, numbins, 2, 200))
    for shuf in range(200):
        events_shuffled = l0_trim[:, np.random.permutation(np.size(l0_trim,1))]
        events_shuffled_sorted = events_shuffled[:, np.argsort(dx_trim)]
        for i in range(numbins):
            offset = np.were(dx_sorted>1)[0][0]
            if i==0:
                binned_cells_shuffled[:,i,0,shuf] = np.mean(events_shuffled_sorted[:,:offset], axis=1)
                binned_cells_shuffled[:,i,1,shuf] = np.std(events_shuffled_sorted[:,:offset], axis=1)/np.sqrt(offset)
            else:
                start = offset + (i-1)*binsize
                binned_cells_shuffled[:,i,0,shuf] = np.mean(events_shuffled_sorted[:,start:start+binsize], axis=1)
                binned_cells_shuffled[:,i,1,shuf] = np.std(events_shuffled_sorted[:,start:start+binsize], axis=1)/np.sqrt(binsize)
    shuffled_variance = binned_cells_shuffled[:,:,0,:].std(axis=1)**2
    variance_threshold = np.percentile(shuffled_variance, 99.9, axis=1)
    response_variance = binned_cells[:,:,0].std(axis=1)**2
    
    peak = pd.DataFrame(columns=('cell_specimen_id','run_mod'),index=range(numbercells))
    peak.cell_specimen_id = specimen_ids
    peak.run_mod = response_variance>variance_threshold
    
    #save data
    save_file = os.path.join(save_path, str(session_id)+'_speed_tuning_events.h5')
    store = pd.HDFStore(save_file)
    store['peak'] = peak
    store.close()
    f = h5py.File(save_file, 'r+')
    dset = f.create_dataset('binned_dx', data=binned_dx)
    dset1 = f.create_dataset('binned_cells', data=binned_cells)
    dset2 = f.create_dataset('binned_cells_shuffled', data=binned_cells_shuffled)
    f.close()

if __name__=='__main__':
    session_id = 566752133
    Speed_Tuning(session_id, binsize=900)
    

