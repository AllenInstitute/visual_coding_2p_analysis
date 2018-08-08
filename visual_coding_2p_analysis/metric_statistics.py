#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:18:33 2018

@author: saskiad
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
area_labels = ['V1','LM','AL','PM','AM','RL']
cres = [u'Cux2-CreERT2', u'Emx1-IRES-Cre', u'Fezf2-CreER', u'Nr5a1-Cre',
       u'Ntsr1-Cre_GN220', u'Rbp4-Cre_KL100', u'Rorb-IRES2-Cre',
       u'Scnn1a-Tg3-Cre', u'Slc17a7-IRES2-Cre', u'Sst-IRES-Cre',
       u'Tlx3-Cre_PL56', u'Vip-IRES-Cre']
cre_depth_list = [('Emx1-IRES-Cre', 100),
 ('Slc17a7-IRES2-Cre', 100),
 ('Cux2-CreERT2', 100),
 ('Emx1-IRES-Cre', 200),
 ('Slc17a7-IRES2-Cre', 200),
 ('Cux2-CreERT2', 200),
 ('Rorb-IRES2-Cre', 200),
 ('Scnn1a-Tg3-Cre', 200),
 ('Nr5a1-Cre', 200),
 ('Emx1-IRES-Cre', 300),
 ('Slc17a7-IRES2-Cre', 300),
 ('Rbp4-Cre_KL100', 300),
 ('Fezf2-CreER', 300),
 ('Tlx3-Cre_PL56', 300),
 ('Ntsr1-Cre_GN220', 500), 
 ('Vip-IRES-Cre', 100),
 ('Vip-IRES-Cre', 200),
 ('Sst-IRES-Cre', 200),
 ('Sst-IRES-Cre', 300)]

cre_list = ['Emx1',
 'Slc17a7',
 'Cux2',
 'Emx1',
 'Slc17a7',
 'Cux2',
 'Rorb',
 'Scnn1a',
 'Nr5a1',
 'Emx1',
 'Slc17a7',
 'Rbp4',
 'Fezf2',
 'Tlx3',
 'Ntsr1',
 'Vip 2/3',
 'Vip 4',
 'Sst 4',
 'Sst 5',
 ]

def metric_stats(data_input, metric, ttest=False):
    '''performs statistical test (either ttest_ind or ks_2samp) for a specified 
    metric value comparing all Cre_depth within each area

Parameters
----------
data_input: dataframe of data
metric: string of the metric to use for statistical test
ttest: Boolean. True uses ttest, False uses KS test. Defaults to False.

Returns
--------
dictionary of p-value arrays for each area
        '''
    metric_dict = {}
    for a in areas:
        temp = data_input[data_input.area==a]
        temp_stats = np.empty((19,19))
        for i,c in enumerate(cre_depth_list):
            dist1 = temp[temp.cre_depth==c][metric].values
            for i2,c2 in enumerate(cre_depth_list):
                if i==i2:
                    temp_stats[i,i2] = np.NaN
                else:
                    dist2 = temp[temp.cre_depth==c2][metric].values
                    if np.logical_and(len(dist1)>0,len(dist2)>0):
                        if ttest:
                            r,p = st.ttest_ind(dist1,dist2,equal_var=False)
                        else:
                            r,p = st.ks_2samp(dist1,dist2)                        
                        temp_stats[i,i2] = np.log10(p)
                    else:
                        temp_stats[i,i2] = np.NaN
        metric_dict[a] = temp_stats
    return metric_dict
        

def metric_stats_cre(data_input, metric,ttest=False):
    '''performs statistical test (either ttest_ind or ks_2samp) for each
    Cre line comparing across areas

Parameters
----------
data_input: dataframe of data
metric: string of the metric to use for statistical test
ttest: Boolean. True uses ttest, False uses KS test. Defaults to False.

Returns
--------
dictionary of p-values for each Cre line
        '''
    metric_dict = {}
    for c in cres:
        temp = data_input[data_input.tld1_name==c]
        temp_stats = np.empty((6,6))
        for i,a in enumerate(areas):
            dist1 = temp[(temp.area==a)][metric].values
            for j,b in enumerate(areas):
                if i==j:
                    temp_stats[i,j] = np.NaN
                else:
                    dist2 = temp[(temp.area==b)][metric].values   
                    if np.logical_and(len(dist1)>0,len(dist2)>0):
                        if ttest:
                            r,p = st.ttest_ind(dist1,dist2,equal_var=False)
                        else:
                            r,p = st.ks_2samp(dist1,dist2)
                        if p==0:
                            print c, i, j
                        temp_stats[i,j] = np.log10(p)
                    else:
                        temp_stats[i,j] = np.NaN                    
        metric_dict[c] = temp_stats
    return metric_dict


def plot_p_values(metric_array, title, area_flag=True):
    '''plots and saves a heatmap of the p-values computed. Colorscale is centered 
    at the significance threshold after multiple comparison correction.

Parameters
----------
metric_array: 
title: string of figure title
area_flag: Boolean. True is for heatmap of a Cre line, comparing across areas. 
    False is for a heatmap of all Cre line/depths within an area.
        '''
    fig = plt.figure(figsize=(4.5,4))
    ax = fig.add_subplot(111)

    if area_flag:
        numcond = np.isfinite(metric_array[0]).sum()
        centervalue = np.log10(0.05/numcond)
        print numcond, 0.05/numcond
        ticks=[centervalue-2,centervalue-1,centervalue,centervalue+1,centervalue+2]
        ticklabels = ['%.0e' % x for x in np.power(10,ticks)]
        with sns.axes_style('white'):
            sns.heatmap(metric_array, cmap='seismic_r', linewidths=0.5, center=centervalue, 
                        vmin=-4, cbar_kws={'ticks':ticks})
#            sns.heatmap(metric_array, cmap='seismic_r', linewidths=0.5, center=-1.6, 
#                        vmin=-4, cbar_kws={'label': 'log p value',
#                                           'ticks':[-3.6,-2.6,-1.6,-0.6,0.4]}) 
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        plt.xticks(np.arange(0.5,6,1.), area_labels)
        plt.yticks(np.arange(0.5,6,1.), area_labels, rotation=0)
        plt.tick_params(labelsize=16)
    else:
        numcond = np.isfinite(metric_array[1]).sum()
        if numcond==0:
            numcond = np.isfinite(metric_array[0]).sum()
        centervalue = np.log10(0.05/numcond)
        print numcond, 0.05/numcond
        ticks=[centervalue-2,centervalue-1,centervalue,centervalue+1,centervalue+2]
        ticklabels = ['%.0e' % x for x in np.power(10,ticks)]
        with sns.axes_style('white'):
            ax=sns.heatmap(metric_array, cmap='seismic_r', linewidths=0.5, center=centervalue, 
                        vmin=-4.6, cbar_kws={'ticks':ticks})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)

        plt.axhline(y=3, color='k')
        plt.axvline(x=3, color='k')
        plt.axhline(y=9, color='k')
        plt.axvline(x=9, color='k')
        plt.axhline(y=14, color='k')
        plt.axvline(x=14, color='k')
        plt.axhline(y=15, color='k')
        plt.axvline(x=15, color='k')
        plt.axhline(y=17, color='k')
        plt.axvline(x=17, color='k')
        plt.xticks(np.arange(0.5, 19, 1.), cre_list, rotation=90)
        plt.yticks(np.arange(0.5, 19, 1), cre_list, rotation=0)
#    plt.title(title)
    fig.tight_layout()
    save_figure(fig, r'/Users/saskiad/Documents/CAM/paper figures/stats/'+title)


def save_figure(fig, fname, formats=['.png','.pdf'], transparent=False, dpi=300, facecolor=None, **kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    else:
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight())
        # fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )