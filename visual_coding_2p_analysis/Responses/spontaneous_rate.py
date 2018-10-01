#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:36:18 2018

@author: saskiad
"""

# dataset
# get spontanoues stim table
# get L0
# spontaneous rate
# table cell id, cre, area, depth, cre_depth, spontaneous rate

import numpy as np
import pandas as pd
#import os, h5py
#import scipy.stats as st
import core
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

save_path = core.get_save_path()
manifest_path = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file=manifest_path)
exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))
for i,a in enumerate(exps.experiment_container_id.unique()):
    if np.mod(i,50)==0:
        print i
    subset = exps[exps.experiment_container_id==a]
    session_A = subset[subset.session_type=='three_session_A'].id.values[0]
    session_B = subset[subset.session_type=='three_session_B'].id.values[0]
    try:
        session_C = subset[subset.session_type=='three_session_C'].id.values[0]
    except:
        session_C = subset[subset.session_type=='three_session_C2'].id.values[0]
    dataset_A = boc.get_ophys_experiment_data(session_A)
    specimen_ids_A = dataset_A.get_cell_specimen_ids()
    dataset_B = boc.get_ophys_experiment_data(session_B)
    specimen_ids_B = dataset_B.get_cell_specimen_ids()
    dataset_C = boc.get_ophys_experiment_data(session_C)
    specimen_ids_C = dataset_C.get_cell_specimen_ids()
    
    l0_A = core.get_L0_events(session_A)
    l0_B = core.get_L0_events(session_B)
    l0_C = core.get_L0_events(session_C)
    
    stim = dataset_A.get_stimulus_table('spontaneous')
    spontaneous_rate_A = l0_A[:,int(stim.start):int(stim.end)].mean(axis=1)
    stim = dataset_B.get_stimulus_table('spontaneous')
    spontaneous_rate_B = l0_B[:,int(stim.start):int(stim.end)].mean(axis=1)
    stim = dataset_C.get_stimulus_table('spontaneous')
    spontaneous_1 = l0_C[:,int(stim.start[0]):int(stim.end[0])].mean(axis=1)
    spontaneous_2 = l0_C[:,int(stim.start[1]):int(stim.end[1])].mean(axis=1)
    spontaneous_rate_C = (spontaneous_1+spontaneous_2)/2.
    
    table_A = pd.DataFrame(columns=('cell_specimen_id','spontaneous_A'), index=range(len(specimen_ids_A)))
    table_B = pd.DataFrame(columns=('cell_specimen_id','spontaneous_B'), index=range(len(specimen_ids_B)))
    table_C = pd.DataFrame(columns=('cell_specimen_id','spontaneous_C'), index=range(len(specimen_ids_C)))
    
    table_A['cell_specimen_id'] = specimen_ids_A
    table_A['spontaneous_A'] = spontaneous_rate_A
    table_B['cell_specimen_id'] = specimen_ids_B
    table_B['spontaneous_A'] = spontaneous_rate_B
    table_C['cell_specimen_id'] = specimen_ids_C
    table_C['spontaneous_A'] = spontaneous_rate_C
    
    table = pd.merge(table_A, table_B, on='cell_specimen_id',how='outer')
    table = pd.merge(table, table_C, on='cell_specimen_id',how='outer')    
    
    table['experiment_container_id'] = a
    table['tld1_name'] = subset.cre_line.iloc[0]
    table['area'] = subset.targeted_structure.iloc[0]
    table['imaging_depth'] = subset.imaging_depth.iloc[0]
    if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
        depth_range = 200
    elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
        depth_range = 300
    else:
        depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
    table['depth_range'] = depth_range
    table['cre_depth'] = table[['tld1_name','depth_range']].apply(tuple, axis=1)
    
    if i==0:
        table_all = table.copy()
    else:
        table_all = table_all.append(table, ignore_index=True)
table_all.reset_index(inplace=True)
table_all['type'] = 'E'
table_all.ix[table_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
table_all.ix[table_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'