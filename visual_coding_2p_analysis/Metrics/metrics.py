#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:18:10 2018

@author: saskiad
Merge all the peak metrics dataframes for all stimuli
"""

import pandas as pd
import numpy as np
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import core


def merge_all_metrics():
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))

    for i,a in enumerate(exps.experiment_container_id.unique()):
        subset = exps[exps.experiment_container_id==a]
        session_A = subset[subset.session_type=='three_session_A'].id.values[0]
        session_B = subset[subset.session_type=='three_session_B'].id.values[0]
        data_file_dg = os.path.join(save_path, 'Drifting Gratings', str(session_A)+"_dg_events_analysis.h5")
        peak_dg = pd.read_hdf(data_file_dg, 'peak')
        data_file_nm = os.path.join(save_path, 'Natural Movies', str(session_A)+"_nm_events_analysis.h5")
        peak_nm = pd.read_hdf(data_file_nm, 'peak')
        data_file_sg = os.path.join(save_path, 'Static Gratings', str(session_B)+"_sg_events_analysis.h5")
        peak_sg = pd.read_hdf(data_file_sg, 'peak')
        data_file_ns = os.path.join(save_path, 'Natural Scenes', str(session_B)+"_ns_events_analysis.h5")
        peak_ns = pd.read_hdf(data_file_ns, 'peak')
        
        peak_all = pd.merge(peak_dg, peak_nm, on='cell_specimen_id', how='outer')
        peak_all = pd.merge(peak_all, peak_sg, on='cell_specimen_id', how='outer')
        peak_all = pd.merge(peak_all, peak_ns, on='cell_specimen_id', how='outer')
    #    peak_all = pd.merge(peak_all, peak_lsn, on='cell_specimen_id', how='outer')
        
        peak_all['experiment_container_id'] = a
        peak_all['tld1_name'] = subset.cre_line.iloc[0]
        peak_all['area'] = subset.targeted_structure.iloc[0]
        peak_all['imaging_depth'] = subset.imaging_depth.iloc[0]    
        peak_all.to_csv(os.path.join(save_path, 'Metrics', str(a)+'_all_metrics.csv'))
        
        if i==0:
            metrics = peak_all.copy()
        else:
            metrics = metrics.append(peak_all)
    
    metrics.to_csv(os.path.join(save_path, 'Metrics', 'metrics.csv'))
    return metrics


def get_all_dg():
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))
    missing = []
    
    for i,a in enumerate(exps.experiment_container_id.unique()):
        subset = exps[exps.experiment_container_id==a]
        try:
            session_A = subset[subset.session_type=='three_session_A'].id.values[0]
            data_file_dg = os.path.join(save_path, 'DriftingGratings', str(session_A)+"_dg_events_analysis.h5")
            peak_dg = pd.read_hdf(data_file_dg, 'peak')
            peak_dg['experiment_container_id'] = a
            peak_dg['tld1_name'] = subset.cre_line.iloc[0]

            peak_dg['area'] = subset.targeted_structure.iloc[0]
            peak_dg['imaging_depth'] = subset.imaging_depth.iloc[0]
            
            if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                depth_range = 200
            elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
                depth_range = 300
            else:
                depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
            peak_dg['depth_range'] = depth_range
            peak_dg['cre_depth'] = peak_dg[['tld1_name','depth_range']].apply(tuple, axis=1)
            if i==0:
                peak_all = peak_dg.copy()
            else:
                peak_all = peak_all.append(peak_dg)
        except:
            missing.append(a)
    peak_all.reset_index(inplace=True)
    peak_all['type'] = 'E'
    peak_all.ix[peak_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
    peak_all.ix[peak_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'
#    peak_all['depth_range'] =np.floor(peak_all.imaging_depth/100)
#    peak_all[['depth_range']] = peak_all[['depth_range']].astype(int)
#    peak_all.depth_range*=100
#    peak_all.loc[peak_all.tld1_name=='Scnn1a-Tg3-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Nr5a1-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Fezf2-CreER', 'depth_range'] = 300

    peak_all.to_hdf(os.path.join(save_path, 'Metrics', 'metrics_dg.h5'), 'peak_all')
    
    return peak_all, missing

def get_all_ns():
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))
    missing = []
    
    for i,a in enumerate(exps.experiment_container_id.unique()):
        subset = exps[exps.experiment_container_id==a]
        try:
            session_B = subset[subset.session_type=='three_session_B'].id.values[0]
            data_file_ns = os.path.join(save_path, 'NaturalScenes', str(session_B)+"_ns_events_analysis.h5")
            peak_ns = pd.read_hdf(data_file_ns, 'peak')
            peak_ns['experiment_container_id'] = a
            peak_ns['tld1_name'] = subset.cre_line.iloc[0]
            peak_ns['area'] = subset.targeted_structure.iloc[0]
            peak_ns['imaging_depth'] = subset.imaging_depth.iloc[0]
            if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                depth_range = 200
            elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
                depth_range = 300
            else:
                depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
            peak_ns['depth_range'] = depth_range
            peak_ns['cre_depth'] = peak_ns[['tld1_name','depth_range']].apply(tuple, axis=1)

            if i==0:
                peak_all = peak_ns.copy()
            else:
                peak_all = peak_all.append(peak_ns, ignore_index=True)
        except:
            missing.append(a)
    peak_all.reset_index(inplace=True)
    peak_all['type'] = 'E'
    peak_all.ix[peak_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
    peak_all.ix[peak_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'
#    peak_all['depth_range'] =np.floor(peak_all.imaging_depth/100)
#    peak_all[['depth_range']] = peak_all[['depth_range']].astype(int)
#    peak_all.depth_range*=100
#    peak_all.loc[peak_all.tld1_name=='Scnn1a-Tg3-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Nr5a1-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Fezf2-CreER', 'depth_range'] = 300
            
    peak_all.to_hdf(os.path.join(save_path, 'Metrics', 'metrics_ns.h5'), 'peak_all')
    
    return peak_all, missing


def get_all_sg():
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))
    missing = []
    
    for i,a in enumerate(exps.experiment_container_id.unique()):
        subset = exps[exps.experiment_container_id==a]
        try:
            session_B = subset[subset.session_type=='three_session_B'].id.values[0]
            data_file_sg = os.path.join(save_path, 'StaticGratings', str(session_B)+"_sg_events_analysis.h5")
            peak_sg = pd.read_hdf(data_file_sg, 'peak')
            peak_sg['experiment_container_id'] = a
            peak_sg['tld1_name'] = subset.cre_line.iloc[0]
            peak_sg['area'] = subset.targeted_structure.iloc[0]
            peak_sg['imaging_depth'] = subset.imaging_depth.iloc[0]
            if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                depth_range = 200
            elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
                depth_range = 300
            else:
                depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
            peak_sg['depth_range'] = depth_range
            peak_sg['cre_depth'] = peak_sg[['tld1_name','depth_range']].apply(tuple, axis=1)

            if i==0:
                peak_all = peak_sg.copy()
            else:
                peak_all = peak_all.append(peak_sg)
        except:
            missing.append(a)
    peak_all.reset_index(inplace=True)
    peak_all['type'] = 'E'
    peak_all.ix[peak_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
    peak_all.ix[peak_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'
#    peak_all['depth_range'] =np.floor(peak_all.imaging_depth/100)
#    peak_all[['depth_range']] = peak_all[['depth_range']].astype(int)
#    peak_all.depth_range*=100
#    peak_all.loc[peak_all.tld1_name=='Scnn1a-Tg3-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Nr5a1-Cre','depth_range'] = 200
#    peak_all.loc[peak_all.tld1_name=='Fezf2-CreER', 'depth_range'] = 300
#    peak_all.reset_index(inplace=True)

    peak_all.to_hdf(os.path.join(save_path, 'Metrics', 'metrics_sg.h5'), 'peak_all')
    
    return peak_all, missing

def get_all_nm(session):
    save_path = core.get_save_path()
    manifest_path = core.get_manifest_path()
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    exps = pd.DataFrame(boc.get_ophys_experiments(include_failed=False))
    missing = []
    
    for i,a in enumerate(exps.experiment_container_id.unique()):
        subset = exps[exps.experiment_container_id==a]
        try:
            session_A = subset[subset.session_type=='three_session_A'].id.values[0]
            data_file_nm = os.path.join(save_path, 'NaturalMovies', str(session_A)+"_nm_events_analysis.h5")
            peak_nm = pd.read_hdf(data_file_nm, 'peak')
            peak_nm['experiment_container_id'] = a
            peak_nm['tld1_name'] = subset.cre_line.iloc[0]
            peak_nm['area'] = subset.targeted_structure.iloc[0]
            peak_nm['imaging_depth'] = subset.imaging_depth.iloc[0]
            if subset.cre_line.iloc[0] in ['Scnn1a-Tg3-Cre', 'Nr5a1-Cre']:
                depth_range = 200
            elif subset.cre_line.iloc[0] in ['Fezf2-CreER']:
                depth_range = 300
            else:
                depth_range = 100*((np.floor(subset.imaging_depth.iloc[0]/100)).astype(int))
            peak_nm['depth_range'] = depth_range
            peak_nm['cre_depth'] = peak_nm[['tld1_name','depth_range']].apply(tuple, axis=1)

            if i==0:
                peak_all = peak_nm.copy()
            else:
                peak_all = peak_all.append(peak_nm, ignore_index=True)
        except:
            missing.append(a)
    peak_all.reset_index(inplace=True)
    peak_all['type'] = 'E'
    peak_all.ix[peak_all.tld1_name=='Sst-IRES-Cre', 'type'] = 'I'
    peak_all.ix[peak_all.tld1_name=='Vip-IRES-Cre', 'type'] = 'I'

    peak_all.to_hdf(os.path.join(save_path, 'Metrics', 'metrics_nm_A.h5'), 'peak_all')
    
    return peak_all, missing
    
    
if __name__=='__main__':
    peak_dg, missing_dg = get_all_dg()
    peak_sg, missing_sg = get_all_sg()
    peak_ns, missing_ns = get_all_ns()
    peak_nm, missing_nm = get_all_nm()
    
# peak_nm[['tld1_name','depth_range']].apply(tuple, axis=1)   
    
#peak_dg[['g_dsi_dg','g_osi_dg','tfdi_dg','reliability_dg','fit_tf_dg','tf_low_cutoff_dg','tf_high_cutoff_dg']] = peak_dg[['g_dsi_dg','g_osi_dg','tfdi_dg','reliability_dg','fit_tf_dg','tf_low_cutoff_dg','tf_high_cutoff_dg']].astype(float)
#peak_dg[['cell_specimen_id','pref_ori_dg','pref_tf_dg','fit_tf_ind_dg','num_pref_trials_dg']] = peak_dg[['cell_specimen_id','pref_ori_dg','pref_tf_dg','fit_tf_ind_dg','num_pref_trials_dg']].astype(int)
#
#peak_sg[['num_pref_trials_sg','imaging_depth']] = peak_sg[['num_pref_trials_sg','imaging_depth']].astype(int)
#peak_ns[['image_selectivity_ns','reliability_ns','lifetime_sparseness_ns']] = peak_ns[['image_selectivity_ns','reliability_ns','lifetime_sparseness_ns']].astype(float)