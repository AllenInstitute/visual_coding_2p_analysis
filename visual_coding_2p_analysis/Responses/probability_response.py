#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:42:34 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os
import core
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

save_path_head = core.get_save_path()
save_path = os.path.join(save_path_head, 'NaturalScenes')
manifest_path = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file = manifest_path)
exp = pd.DataFrame(boc.get_ophys_experiments())
exp_dg = exp[exp.session_type=='three_session_B'].id.values
for i,a in enumerate(exp_dg):
    session_id = a
    stim_table, numbercells, specimen_ids = core.get_stim_table(session_id, 'natural_scenes')
    file_name = os.path.join(save_path, str(session_id)+'_ns_events_analysis.h5')
    sweep_p_values = pd.read_hdf(file_name, 'sweep_p_values')
    resp = pd.DataFrame(index=range(118),columns=np.array(range(numbercells)).astype(str))
    for im in range(118):
        subset = sweep_p_values[stim_table.frame==im]
        resp.loc[im] = subset[subset<(0.05/len(subset))].count()
    resp[resp>0].count()/1.18
    
    output = pd.DataFrame(columns=('cell_specimen_id','probability_response_ns'), index=range(numbercells))
    output['cell_specimen_id'] = specimen_ids
    output['probability_response_ns'] = resp[resp>0].count().values/1.18
    
    if i==0:
        prob_response = output.copy()
    else:
        prob_response = prob_response.append(output)
