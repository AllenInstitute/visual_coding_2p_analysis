# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:32:16 2017

@author: danielm
"""
import pandas as pd
import numpy as np

from population_overlap import PopulationOverlap

def run_analysis(metrics_path,save_path):
    
    areas = ['all','VISp','VISl','VISpm','VISam','VISrl','VISal']
    
    stim_names = ['LSN',
                  'SG',
                  'DG',
                  'NS',
                  'NM1a',
                  'NM1b',
                  'NM1c',
                  'NM2',
                  'NM3']
    
    stim = ['responsive_lsn',
            'responsive_sg',
            'responsive_dg',
            'responsive_ns',
            'responsive_nm1a',
            'responsive_nm1b',
            'responsive_nm1c',
            'responsive_nm2',
            'responsive_nm3']  

    cells = get_cell_metrics(metrics_path)
    
    for area in areas:
        if area == 'all':
            area_cells = cells.copy()
        else:
            area_cells = cells.iloc[(cells['area']==area).values]
                                    
        p = PopulationOverlap(area_cells,stim,stim_names,save_path+area+'\\')   
        
        p.plot_stim_pair_effects()
        p.plot_normalized_barplot_and_scatterplot()
        p.plot_unnormalized_barplot()
     
def get_cell_metrics(metrics_path):
    
    columns_to_keep = ['csid',
                       'ecid',
                       'area',
                       'responsive_dg',
                       'responsive_sg',
                       'responsive_ns',
                       'responsive_lsn',
                       'responsive_nm1a',
                       'responsive_nm1b',
                       'responsive_nm1c',
                       'responsive_nm2',
                       'responsive_nm3']
    
    dg = pd.read_hdf(metrics_path+'metrics_dg.h5')
    sg = pd.read_hdf(metrics_path+'metrics_sg.h5')
    ns = pd.read_hdf(metrics_path+'metrics_ns.h5')
    nm = pd.read_hdf(metrics_path+'metrics_nm.h5')
    lsn = pd.read_csv(metrics_path+'lsn_180413.csv')
    
    metrics = pd.merge(nm,dg,on='cell_specimen_id',how='outer',suffixes=['','_r'])
    metrics = pd.merge(metrics,sg,on='cell_specimen_id',how='outer',suffixes=['','_r'])
    metrics = pd.merge(metrics,ns,on='cell_specimen_id',how='outer',suffixes=['','_r'])
    metrics = pd.merge(metrics,lsn,on='cell_specimen_id',how='outer',suffixes=['','_r'])
    
    metrics.rename(columns = {'cell_specimen_id':'csid'}, inplace = True)
    metrics.rename(columns = {'experiment_container_id':'ecid'}, inplace = True)
    
    ok_after_datalock = exclude_failed_after_datalock(metrics)
    print 'number of cells: ' + str(sum(ok_after_datalock))
    
    return metrics[columns_to_keep][ok_after_datalock]
 
def exclude_failed_after_datalock(cells):
    
    failed_ecs = [
                  511510998,
                  511510681,
                  517328083,
                  527676429,
                  527550471,
                  530243910,
                  570278595,
                  571039045,
                  585905043,
                  587695553,
                  596780703,
                  598134911,
                  599587151,
                  605113106
                  ]
                  
    not_failed = np.ones((len(cells),)).astype(np.bool)
    ecids = cells['ecid'].values
    for ec in failed_ecs:
        ec_passed = ec != ecids
        not_failed = not_failed & ec_passed
        
    print 'cells removed: ' + str(np.sum(~not_failed))
        
    return not_failed