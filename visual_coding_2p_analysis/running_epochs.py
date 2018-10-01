#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:48:57 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import core
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import scipy.ndimage as ndi

save_path = core.get_save_path()
manifest_path = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file=manifest_path)
exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))

A = exps[exps.session_type=='three_session_A']['id'].values
running_array = np.empty((424, 7, 60))
mean_running_speed = np.empty((424, 7))
running_array[:] = np.NaN
mean_running_speed[:] = np.NaN

failed = []
for i, session in enumerate(A):
    try:
        dataset = boc.get_ophys_experiment_data(session)
        dxcm, ts = dataset.get_running_speed()
        temp = ndi.filters.gaussian_filter1d(dxcm, 10)
        epoch_table = dataset.get_stimulus_epoch_table()
        for index,row in epoch_table.iterrows():
            temp_epoch = temp[row.start:row.end]
            v,h = np.histogram(temp_epoch[np.isfinite(temp_epoch)], range=(0,120), bins=60)
            running_array[i, index,:] = v
            mean_running_speed[i, index] = np.nanmean(temp_epoch)    
    except:
        failed.append(session)