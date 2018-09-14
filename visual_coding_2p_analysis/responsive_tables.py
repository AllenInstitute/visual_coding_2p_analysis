#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:16:50 2018

@author: saskiad
"""
import pandas as pd

cres
areas = ['VISp','VISl','VISal','VISpm','VISam','VISrl']
depths = [100,200,300,500]

table = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_dg','responsive_cells_dg', 'percent_dg'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth):
        i = (ci)+(19*ai)
        table.area.iloc[i] = a
        table.cre.iloc[i] = c[0]
        table.depth.iloc[i] = c[1]
        table.number_experiments.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])].experiment_container_id.unique())
        table.number_cells_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])])
        table.responsive_cells_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])&(peak_dg.responsive_dg)])            
        if len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])])>0:
            table.percent_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])&(peak_dg.responsive_dg)])/float(len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])]))
        
        
        table.number_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])
        table.responsive_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])            
        table.number_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])
        table.responsive_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])            
            
table_sg = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_sg','responsive_cells_sg', 'percent_sg'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth):
        i = (ci)+(19*ai)
        table_sg.area.iloc[i] = a
        table_sg.cre.iloc[i] = c[0]
        table_sg.depth.iloc[i] = c[1]
        table_sg.number_experiments.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])].experiment_container_id.unique())
        table_sg.number_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])
        table_sg.responsive_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])            
        if len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])>0:
            table_sg.percent_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])/float(len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])]))


table_ns = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_ns','responsive_cells_ns', 'percent_ns'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth):
        i = (ci)+(19*ai)
        table_ns.area.iloc[i] = a
        table_ns.cre.iloc[i] = c[0]
        table_ns.depth.iloc[i] = c[1]
        table_ns.number_experiments.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])].experiment_container_id.unique())
        table_ns.number_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])
        table_ns.responsive_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])            
        if len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])>0:
            table_ns.percent_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])/float(len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])]))

table_nm = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_nm','responsive_cells_nm', 'percent_nm'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth):
        i = (ci)+(19*ai)
        table_nm.area.iloc[i] = a
        table_nm.cre.iloc[i] = c[0]
        table_nm.depth.iloc[i] = c[1]
        table_nm.number_experiments.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])].experiment_container_id.unique())
        table_nm.number_cells_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])])
        table_nm.responsive_cells_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])&(peak_nm.responsive_nm3)])            
        if len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])])>0:
            table_nm.percent_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])&(peak_nm.responsive_nm3)])/float(len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])]))

table_nm = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_nm','responsive_cells_nm', 'percent_nm'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_nm.area.iloc[i] = a
        table_nm.cre.iloc[i] = c[0]
        table_nm.depth.iloc[i] = c[1]
        subset = metrics[(metrics['area_y.1']==a)&(metrics['tld1_name_y.1']==c[0])&(metrics['depth_range_y.1']==c[1])]
        table_nm.number_experiments.iloc[i] = len(subset.experiment_container_fix.unique())
        table_nm.number_cells_nm.iloc[i] = len(subset)
        resp_cells = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|(subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])            
        table_nm.responsive_cells_nm.iloc[i] = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|(subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])            
        if len(subset)>0:
            table_nm.percent_nm.iloc[i] = resp_cells/float(len(subset))

    exp_nm.area.loc[i] = subset['area_y.1'].iloc[0]
    exp_nm.cre.loc[i] = subset['tld1_name_y.1'].iloc[0]
    exp_nm.depth.loc[i] = subset['depth_range_y.1'].iloc[0]


table = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_lsn','responsive_cells_lsn', 'percent_lsn'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table.area.iloc[i] = a
        table.cre.iloc[i] = c[0]
        table.depth.iloc[i] = c[1]
        table.number_experiments.iloc[i] = len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])].experiment_container_id.unique())
        table.number_cells_lsn.iloc[i] = len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])])
        table.responsive_cells_lsn.iloc[i] = len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])&(rf_meta.responsive_lsn)])            
        if len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])])>0:
            table.percent_lsn.iloc[i] = len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])&(rf_meta.responsive_lsn)])/float(len(rf_meta[(rf_meta.area==a)&(rf_meta.tld1_name==c[0])&(rf_meta.depth_range==c[1])]))
        
        
        table.number_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])
        table.responsive_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])            
        table.number_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])
        table.responsive_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])            
 
table_dg = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_dg','responsive_cells_dg', 'percent_dg'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_dg.area.iloc[i] = a
        table_dg.cre.iloc[i] = c[0]
        table_dg.depth.iloc[i] = c[1]
        table_dg.number_experiments.iloc[i] = len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])].experiment_container_id.unique())
        table_dg.number_cells_dg.iloc[i] = len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])])
        table_dg.responsive_cells_dg.iloc[i] = len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])&(metrics_dg_new.responsive_dg)])            
        if len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])])>0:
            table_dg.percent_dg.iloc[i] = 100*len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])&(metrics_dg_new.responsive_dg)])/float(len(metrics_dg_new[(metrics_dg_new.area==a)&(metrics_dg_new.tld1_name==c[0])&(metrics_dg_new.depth_range==c[1])]))

table_sg = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_sg','responsive_cells_sg', 'percent_sg'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_sg.area.iloc[i] = a
        table_sg.cre.iloc[i] = c[0]
        table_sg.depth.iloc[i] = c[1]
        table_sg.number_experiments.iloc[i] = len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])].experiment_container_id.unique())
        table_sg.number_cells_sg.iloc[i] = len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])])
        table_sg.responsive_cells_sg.iloc[i] = len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])&(metrics_sg_new.responsive_sg)])            
        if len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])])>0:
            table_sg.percent_sg.iloc[i] = 100*len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])&(metrics_sg_new.responsive_sg)])/float(len(metrics_sg_new[(metrics_sg_new.area==a)&(metrics_sg_new.tld1_name==c[0])&(metrics_sg_new.depth_range==c[1])]))

table_ns = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_ns','responsive_cells_ns', 'percent_ns'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_ns.area.iloc[i] = a
        table_ns.cre.iloc[i] = c[0]
        table_ns.depth.iloc[i] = c[1]
        table_ns.number_experiments.iloc[i] = len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])].experiment_container_id.unique())
        table_ns.number_cells_ns.iloc[i] = len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])])
        table_ns.responsive_cells_ns.iloc[i] = len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])&(metrics_ns_new.responsive_ns)])            
        if len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])])>0:
            table_ns.percent_ns.iloc[i] = 100*len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])&(metrics_ns_new.responsive_ns)])/float(len(metrics_ns_new[(metrics_ns_new.area==a)&(metrics_ns_new.tld1_name==c[0])&(metrics_ns_new.depth_range==c[1])]))

table_lsn.to_html(r'/Users/saskiad/Documents/Data/CAM/lsn_table_new.html')

table_nm = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_nm','responsive_cells_nm', 'percent_nm'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_nm.area.iloc[i] = a
        table_nm.cre.iloc[i] = c[0]
        table_nm.depth.iloc[i] = c[1]
        subset = metrics_new[(metrics_new.area==a)&(metrics_new.tld1_name==c[0])&(metrics_new.depth_range==c[1])]
        table_nm.number_experiments.iloc[i] = len(subset.experiment_container_fix.unique())
        table_nm.number_cells_nm.iloc[i] = len(subset)
        resp_cells = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|(subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])            
        table_nm.responsive_cells_nm.iloc[i] = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|(subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])            
        if len(subset)>0:
            table_nm.percent_nm.iloc[i] = 100*resp_cells/float(len(subset))

table_lsn = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_lsn','responsive_cells_lsn', 'percent_lsn'), index=range(114))
for ai, a in enumerate(areas):
    for ci, c in enumerate(cre_depth_list):
        i = (ci)+(19*ai)
        table_lsn.area.iloc[i] = a
        table_lsn.cre.iloc[i] = c[0]
        table_lsn.depth.iloc[i] = c[1]
        subset = metrics_new[(metrics_new.area==a)&(metrics_new.tld1_name==c[0])&(metrics_new.depth_range==c[1])&np.isfinite(metrics_new.reliability_nm1c)]
        table_lsn.number_experiments.iloc[i] = len(subset.experiment_container_fix.unique())
        table_lsn.number_cells_lsn.iloc[i] = len(subset)
        table_lsn.responsive_cells_lsn.iloc[i] = len(subset[subset.responsive_lsn])            
        if len(subset)>0:
            table_lsn.percent_lsn.iloc[i] = 100*len(subset[subset.responsive_lsn])/float(len(subset))
 


exp = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','number_cells','responsive_cells','percent'), index=range(410))
exp_sg = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
exp_ns = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
exp_nm = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_dg_new.experiment_container_id.unique()):
    subset = metrics_dg_new[metrics_dg_new.experiment_container_id==e]
    exp.experiment_container_id.loc[i] = e
    exp.area.loc[i] = subset.area.iloc[0]
    exp.cre.loc[i] = subset.tld1_name.iloc[0]
    exp.depth.loc[i] = subset.depth_range.iloc[0]
    exp.number_cells.loc[i] = len(subset)
    exp.responsive_cells.loc[i] = len(subset[subset.responsive_dg==True])
    exp.percent.loc[i] = 100*len(subset[subset.responsive_dg==True])/float(len(subset[np.isfinite(subset.g_dsi_dg)]))
    subset = metrics_sg_new[metrics_sg_new.experiment_container_id==e]
    exp_sg.experiment_container_id.loc[i] = e
    exp_sg.area.loc[i] = subset.area.iloc[0]
    exp_sg.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_sg.depth.loc[i] = subset.depth_range.iloc[0]
    exp_sg.number_cells.loc[i] = len(subset)
    exp_sg.responsive_cells.loc[i] = len(subset[subset.responsive_sg])
    exp_sg.percent.loc[i] = 100*len(subset[subset.responsive_sg])/float(len(subset))
    subset = metrics_ns_new[metrics_ns_new.experiment_container_id==e]
    exp_ns.experiment_container_id.loc[i] = e
    exp_ns.area.loc[i] = subset.area.iloc[0]
    exp_ns.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_ns.depth.loc[i] = subset.depth_range.iloc[0]
    exp_ns.number_cells.loc[i] = len(subset)
    exp_ns.responsive_cells.loc[i] = len(subset[subset.responsive_ns])
    exp_ns.percent.loc[i] = 100*len(subset[subset.responsive_ns])/float(len(subset))
    subset = metrics_new[metrics_new.experiment_container_fix==e]
    exp_nm.experiment_container_id.loc[i] = e
    exp_nm.area.loc[i] = subset.area.iloc[0]
    exp_nm.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_nm.depth.loc[i] = subset.depth_range.iloc[0]
    exp_nm.number_cells.loc[i] = len(subset)
    responsive = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|
            (subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])
    exp_nm.responsive_cells.loc[i] = responsive
    exp_nm.percent.loc[i] = responsive/float(len(subset))


exp['cre_depth'] = exp[['cre','depth']].apply(tuple, axis=1)
exp_sg['cre_depth'] = exp_sg[['cre','depth']].apply(tuple, axis=1)
exp_ns['cre_depth'] = exp_ns[['cre','depth']].apply(tuple, axis=1)
exp_nm['cre_depth'] = exp_nm[['cre','depth']].apply(tuple, axis=1)


exp_sg = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_sg_new.experiment_container_id.unique()):
    subset = metrics_sg_new[metrics_sg_new.experiment_container_id==e]
    exp_sg.experiment_container_id.loc[i] = e
    exp_sg.area.loc[i] = subset.area.iloc[0]
    exp_sg.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_sg.depth.loc[i] = subset.depth_range.iloc[0]
    exp_sg.number_cells.loc[i] = len(subset)
    exp_sg.responsive_cells.loc[i] = len(subset[subset.responsive_sg])
    exp_sg.percent.loc[i] = 100*len(subset[subset.responsive_sg])/float(len(subset))

for index, row in exp_sg.iterrows():
    exp_sg.cre_depth.loc[index] = (row.cre, row.depth)


exp_ns = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_ns_new.experiment_container_id.unique()):
    subset = metrics_ns_new[metrics_ns_new.experiment_container_id==e]
    exp_ns.experiment_container_id.loc[i] = e
    exp_ns.area.loc[i] = subset.area.iloc[0]
    exp_ns.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_ns.depth.loc[i] = subset.depth_range.iloc[0]
    exp_ns.number_cells.loc[i] = len(subset)
    exp_ns.responsive_cells.loc[i] = len(subset[subset.responsive_ns])
    exp_ns.percent.loc[i] = 100*len(subset[subset.responsive_ns])/float(len(subset))

for index, row in exp_ns.iterrows():
    exp_ns.cre_depth.loc[index] = (row.cre, row.depth)


exp_nm = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_new.experiment_container_fix.unique()):
    subset = metrics_new[metrics_new.experiment_container_id==e]
    exp_nm.experiment_container_id.loc[i] = e
    exp_nm.area.loc[i] = subset.area.iloc[0]
    exp_nm.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_nm.depth.loc[i] = subset.depth_range.iloc[0]
    exp_nm.number_cells.loc[i] = len(subset)
    responsive = len(subset[(subset.responsive_nm3)|(subset.responsive_nm2)|
            (subset.responsive_nm1a)|(subset.responsive_nm1b)|(subset.responsive_nm1c)])
    exp_nm.responsive_cells.loc[i] = responsive
    exp_nm.percent.loc[i] = responsive/float(len(subset))

for index, row in exp_nm.iterrows():
    exp_nm.cre_depth.loc[index] = (row.cre, row.depth)
    
    
exp_nm = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics.experiment_container_fix.unique()):
    subset = metrics[metrics.experiment_container_fix==e]
    exp_nm.area.loc[i] = subset['area_y.1'].iloc[0]
    exp_nm.cre.loc[i] = subset['tld1_name_y.1'].iloc[0]
    exp_nm.depth.loc[i] = subset['depth_range_y.1'].iloc[0]
    exp_nm.number_cells.loc[i] = len(subset)
    exp_nm.responsive_cells.loc[i] = len(subset[subset.responsive_nm3|subset.responsive_nm2|subset.responsive_nm1c|subset.responsive_nm1b|subset.responsive_nm1a])
    exp_nm.percent.loc[i] = len(subset[subset.responsive_nm3|subset.responsive_nm2|subset.responsive_nm1c|subset.responsive_nm1b|subset.responsive_nm1a])/float(len(subset))

for index, row in exp_nm.iterrows():
    exp_nm.cre_depth.loc[index] = (row.cre, row.depth)
    
exp_lsn = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_lsn.experiment_container_id.unique()):
    subset = metrics_lsn[metrics_lsn.experiment_container_id==e]
    exp_lsn.experiment_container_id.loc[i] = e
    exp_lsn.area.loc[i] = subset.area.iloc[0]
    exp_lsn.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_lsn.depth.loc[i] = subset.depth_range.iloc[0]
    exp_lsn.number_cells.loc[i] = len(subset)
    exp_lsn.responsive_cells.loc[i] = len(subset[subset.responsive_lsn])
    exp_lsn.percent.loc[i] = len(subset[subset.responsive_lsn])/float(len(subset))

exp_lsn = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(410))
for i,e in enumerate(metrics_new.experiment_container_fix.unique()):
    subset = metrics_new[(metrics_new.experiment_container_fix==e)]
    exp_lsn.experiment_container_id.loc[i] = e
    exp_lsn.area.loc[i] = subset.area.iloc[0]
    exp_lsn.cre.loc[i] = subset.tld1_name.iloc[0]
    exp_lsn.depth.loc[i] = subset.depth_range.iloc[0]
    exp_lsn.number_cells.loc[i] = len(subset[np.isfinite(subset.reliability_nm1c)])
    exp_lsn.responsive_cells.loc[i] = len(subset[subset.responsive_lsn==True])
    exp_lsn.percent.loc[i] = 100*len(subset[subset.responsive_lsn==True])/float(len(subset[np.isfinite(subset.reliability_nm1c)]))

for index, row in exp_lsn.iterrows():
    exp_lsn.cre_depth.loc[index] = (row.cre, row.depth)

for index,row in metrics:
    if np.isfinite(row.experiment_container_x):
        metrics['experiment_container_fix'] = row.experiment_container_x
    elif np.isfinite(row.experiment_container_y)

    
sns.boxplot(x='cre_depth',y='percent',data=exp_visp,color='lightgray', order = cre_depth_list)    

area_dict = {}
area_dict['VISp']='V1'
area_dict['VISl']='LM'
area_dict['VISal']='AL'
area_dict['VISpm']='PM'
area_dict['VISam']='AM'
area_dict['VISrl']='RL'

for a in areas:
    plot_responsive_areas(exp_nm, a, 'nm_all', bar=True)
    plot_responsive_areas(exp, a, 'dg', bar=True)
    plot_responsive_areas(exp_sg, a, 'sg', bar=True)
    plot_responsive_areas(exp_ns, a, 'ns', bar=True)
    plot_responsive_areas(exp_lsn, a, 'lsn', bar=True)

def plot_responsive_areas(exp, area,stimulus_suffix, bar=False):
    exp_area = exp[exp.area==area]
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.stripplot(y='cre_depth',x='percent',data=exp_area, palette=cre_depth_palette, 
                      order=cre_depth_list, size=10)
        if bar:
            ax = sns.barplot(y='cre_depth', x='percent', data=exp_area, palette=cre_depth_palette, 
                             order=cre_depth_list, ci=None, alpha=0.5, estimator=np.mean)
        plt.yticks(range(19), cre_list)
        plt.yticks([])
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.ylabel("")
#        plt.xticks([0,0.2,0.4, 0.6,0.8,1],range(0,101,20))
        plt.xticks([0,20,40,60,80,100])
        plt.tick_params(labelsize=26)#20
        plt.xlabel("Percent responsive", fontsize=24)
        plt.xlim(-5, 105)
#        plt.title(area, fontsize=24)
        plt.title(area_dict[area], fontsize=30)#26#20
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+stimulus_suffix+'_responsive_'+area
        save_figure(fig, figname)