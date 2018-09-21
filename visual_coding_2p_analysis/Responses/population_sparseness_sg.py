#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:57:59 2018

@author: saskiad
"""


import numpy as np
import pandas as pd
import os, h5py
import core

manifest_path = core.get_manifest_path()

save_path_head = core.get_save_path()
save_path = os.path.join(save_path_head, 'StaticGratings')

manifest_path = core.get_manifest_path()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file = manifest_path)
exp = pd.DataFrame(boc.get_ophys_experiments(session_types=['three_session_B']))

pop_sparse = np.empty((120,424))
pop_sparse[:] = np.NaN
pop_sparseness = pd.DataFrame(columns=('experiment_container_id','id','cre','area','depth_range','cre_depth','population_sparseness_sg','number_cells_sg'), index=range(424))


for index,row in exp.iterrows():
    if np.mod(index,20)==0:
        print index
    pop_sparseness['experiment_container_id'].loc[index] = row.experiment_container_id
    pop_sparseness['cre'].loc[index] = row.cre_line
    pop_sparseness['area'].loc[index] = row.targeted_structure
    if row.cre_line in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
        depth_range = 200
    elif row.cre_line in ['Fezf2-CreER']:
        depth_range = 300
    else:
        depth_range = 100*((np.floor(row.imaging_depth/100)).astype(int))
    pop_sparseness['depth_range'].loc[index] = depth_range
    pop_sparseness['cre_depth'] = pop_sparseness[['cre','depth_range']].apply(tuple, axis=1)
    
    a = row.id
    file_path = os.path.join(save_path, str(a)+'_sg_events_analysis.h5')
    f = h5py.File(file_path)
    response = f['response_events'].value
    f.close()
    response2 = response[:,1:,:,:,0].reshape(120,-1)

    num_cells = response2.shape[1]
    pop_sparseness['number_cells_sg'].loc[index] = num_cells
    
    ps = ((1-(1/float(num_cells))*((np.power(response2[:,:].sum(axis=1),2))/
                                      (np.power(response2[:,:],2).sum(axis=1))))/(1-(1/float(num_cells))))
    pop_sparse[:,index] = ps
pop_sparseness['population_sparseness_sg'] = np.nanmean(pop_sparse, axis=0)

#areas=['VISp','VISpm','VISam','VISrl','VISal','VISl']
#depths = [100,200,300,500]
#results = np.empty((6,4,2))
#for ai,a in enumerate(areas):
#    for di, d in enumerate(depths):
#        subset = pop_sparseness[(pop_sparseness.area==a)&(pop_sparseness.depth_range==d)]
#        results[ai,di,0] = subset.population_sparseness_ns.median()
#        results[ai,di,1] = len(subset)
        
areas=['VISp','VISpm','VISam','VISrl','VISal','VISl']        
cre_depth = [('Cux2-CreERT2',100),('Rorb-IRES2-Cre',200),('Rbp4-Cre_KL100',300), ('Ntsr1-Cre_GN220',500)]
results_cre = np.empty((6,4,2))
for ai,a in enumerate(areas):
    for di,d in enumerate(cre_depth):
        subset = pop_sparseness[(pop_sparseness.area==a)&(pop_sparseness.depth_range==d[1])&(pop_sparseness.cre==d[0])]
        results_cre[ai,di,0] = subset.population_sparseness_sg.median()
        results_cre[ai,di,1] = len(subset)