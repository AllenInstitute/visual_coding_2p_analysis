#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:11:33 2018

@author: saskiad
"""

import pandas as pd
import numpy as np
import h5py
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import core


def get_evoked_responses():
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))

    for i,a in enumerate(exps.experiment_container_id.unique()):
        if np.mod(i,50)==0:
            print i
        subset = exps[exps.experiment_container_id==a]
        session_A = subset[subset.session_type=='three_session_A'].id.values[0]
        dataset = boc.get_ophys_experiment_data(session_A)
        cell_ids_A = dataset.get_cell_specimen_ids()
        data_file_dg = os.path.join(save_path, 'DriftingGratings', str(session_A)+"_dg_events_analysis.h5")
        f = h5py.File(data_file_dg)
        response_dg = f['response_events'].value
        f.close()
        numbercells = response_dg.shape[2]
        evoked_response = pd.DataFrame(columns=('cell_specimen_id','max_evoked_response_dg'), index=range(numbercells))
        evoked_response['cell_specimen_id'] = cell_ids_A
        response2 = response_dg.reshape(48,numbercells,3)
        evoked_response.max_evoked_response_dg = np.nanmax(response2[:,:,0], axis=0)
        if i==0:
            evoked_response_all = evoked_response.copy()
        else:
            evoked_response_all = evoked_response_all.append(evoked_response)
    return evoked_response_all

if __name__=='__main__':
    evoked_repsonses = get_evoked_responses()
                
        