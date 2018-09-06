#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:23:18 2018

@author: saskiad
"""
import numpy as np
import pandas as pd
import os, h5py
import scipy.stats as st
import core
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

def get_movie_correlations():
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
        a_path = os.path.join(save_path, 'NaturalMoviesA', str(session_A)+'_nm_events_analysis.h5')
        b_path = os.path.join(save_path, 'NaturalMoviesB', str(session_B)+'_nm_events_analysis.h5')
        c_path = os.path.join(save_path, 'NaturalMoviesC', str(session_C)+'_nm_events_analysis.h5')
        dataset_A = boc.get_ophys_experiment_data(session_A)
        specimen_ids_A = dataset_A.get_cell_specimen_ids()
        dataset_B = boc.get_ophys_experiment_data(session_B)
        specimen_ids_B = dataset_B.get_cell_specimen_ids()
        dataset_C = boc.get_ophys_experiment_data(session_C)
        specimen_ids_C = dataset_C.get_cell_specimen_ids()

        
        f = h5py.File(a_path)
        response_A = f['response_events_1a'].value
        f.close()
        f = h5py.File(b_path)
        response_B = f['response_events_1b'].value
        f.close()
        f = h5py.File(c_path)
        response_C = f['response_events_1c'].value
        f.close()
        
        peak_A = pd.read_hdf(a_path, 'peak')
        peak_A['peak_frame_nm1a'] = peak_A.peak_frame_nm1a.astype(int)
        peak_B = pd.read_hdf(b_path, 'peak')
        peak_B['peak_frame_nm1b'] = peak_B.peak_frame_nm1b.astype(int)
        peak_C = pd.read_hdf(c_path, 'peak')
        peak_C['peak_frame_nm1c'] = peak_C.peak_frame_nm1c.astype(int)
        peak = pd.merge(peak_A, peak_B, on='cell_specimen_id', how='outer')
        peak = pd.merge(peak, peak_C, on='cell_specimen_id', how='outer')  
        
        peak['correlation_ab'] = np.NaN
        peak['correlation_bc'] = np.NaN
        peak['correlation_ac'] = np.NaN
        
        peak_subset = peak[np.isfinite(peak.peak_frame_nm1a)&np.isfinite(peak.peak_frame_nm1b)]
        for index,row in peak_subset.iterrows():
            nc = row.cell_specimen_id
            resp_A = response_A[:,np.where(specimen_ids_A==nc)[0][0],0]
            resp_B = response_B[:,np.where(specimen_ids_B==nc)[0][0],0]
            r,p = st.pearsonr(resp_A, resp_B)
            peak.correlation_ab.loc[index] = r
            
        peak_subset = peak[np.isfinite(peak.peak_frame_nm1a)&np.isfinite(peak.peak_frame_nm1c)]
        for index,row in peak_subset.iterrows():
            nc = row.cell_specimen_id
            resp_A = response_A[:,np.where(specimen_ids_A==nc)[0][0],0]
            resp_C = response_C[:,np.where(specimen_ids_C==nc)[0][0],0]
            r,p = st.pearsonr(resp_A, resp_C)
            peak.correlation_ac.loc[index] = r
        
        peak_subset = peak[np.isfinite(peak.peak_frame_nm1c)&np.isfinite(peak.peak_frame_nm1b)]
        for index,row in peak_subset.iterrows():
            nc = row.cell_specimen_id
            resp_C = response_C[:,np.where(specimen_ids_C==nc)[0][0],0]
            resp_B = response_B[:,np.where(specimen_ids_B==nc)[0][0],0]
            r,p = st.pearsonr(resp_C, resp_B)
            peak.correlation_bc.loc[index] = r
        
        peak['experiment_container_id'] = a
        peak['tld1_name'] = subset.cre_line.iloc[0]
        peak['area'] = subset.targeted_structure.iloc[0]
        peak['imaging_depth'] = subset.imaging_depth.iloc[0]
        if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
            depth_range = 200
        elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
            depth_range = 300
        else:
            depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
        peak['depth_range'] = depth_range
        peak['cre_depth'] = peak[['tld1_name','depth_range']].apply(tuple, axis=1)

        if i==0:
            peak_all = peak.copy()
        else:
            peak_all = peak_all.append(peak, ignore_index=True)
    peak_all.reset_index(inplace=True)
    peak_all['type'] = 'E'
    peak_all.ix[peak_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
    peak_all.ix[peak_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'

    peak_all.to_hdf(os.path.join(save_path, 'Metrics', 'metrics_nm.h5'), 'peak_all')
    return peak_all    

if __name__=='__main__':
    peak_all = get_movie_correlations()        
                
                
        
        
        
