#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:54:44 2018

@author: saskiad
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import core

area_dict = {}
area_dict['VISp']='V1'
area_dict['VISl']='LM'
area_dict['VISal']='AL'
area_dict['VISpm']='PM'
area_dict['VISam']='AM'
area_dict['VISrl']='RL'

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

def plot_responsive_areas(exp, area, stimulus_suffix, bar=False):
    cre_depth_list = [('Emx1-IRES-Cre',100), ('Slc17a7-IRES2-Cre', 100),('Cux2-CreERT2',100),('Vip-IRES-Cre',100),
                  ('Emx1-IRES-Cre',200),('Slc17a7-IRES2-Cre', 200),('Cux2-CreERT2',200),
                  ('Rorb-IRES2-Cre',200),('Scnn1a-Tg3-Cre',200),('Nr5a1-Cre',200),('Sst-IRES-Cre',200),('Vip-IRES-Cre',200),
                  ('Emx1-IRES-Cre',300),('Slc17a7-IRES2-Cre', 300),
                  ('Rbp4-Cre_KL100',300),('Fezf2-CreER',300),('Tlx3-Cre_PL56',300),('Sst-IRES-Cre',300),
                  ('Ntsr1-Cre_GN220',500)]

    cre_color_dict = core.get_cre_colors()
    cre_depth_colors = []
    cre_list = []
    for c in cre_depth_list:
        cre_list.append(c[0].split('-')[0])
        cre_depth_colors.append(cre_color_dict[c[0]])
    cre_depth_palette = sns.color_palette(cre_depth_colors)
    
    percent_stim = 'percent_'+stimulus_suffix
    
    exp_area = exp[exp.area==area]
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.stripplot(y='cre_depth',x=percent_stim,data=exp_area, palette=cre_depth_palette, 
                      order=cre_depth_list, size=10)
        if bar:
            ax = sns.barplot(y='cre_depth', x=percent_stim, data=exp_area, palette=cre_depth_palette, 
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
        plt.xticks([0,20,40,60,80,100])
        plt.tick_params(labelsize=26)#20
        plt.xlabel("Percent responsive", fontsize=24)
        plt.xlim(-5, 105)
        plt.title(area_dict[area], fontsize=30)#26#20
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/revision/'+stimulus_suffix+'_responsive_'+area
        save_figure(fig, figname)
        
stim_colors={}
stim_colors['dg'] = '#05728C'
stim_colors['sg'] = '#3DB1AA'
stim_colors['ns'] = '#FD9D4D'
stim_colors['nm'] = '#E35786'
stim_colors['lsn'] = '#1B378A'
stim_colors['all'] = 'gray'




def plot_responsive_summary(metrics, stimulus_suffix):
    areas = ['VISp','VISl','VISal','VISpm','VISam','VISrl']
    area_labels = ['V1','LM','AL','PM','AM','RL']
    resp = np.empty((6))
    if stimulus_suffix=='nm':
        for i,a in enumerate(areas):
            subset = metrics[metrics.area==a]
            resp[i] = len(subset[(subset.responsive_nm1a==True)|(subset.responsive_nm1b==True)|(subset.responsive_nm1c==True)|(subset.responsive_nm2==True)|(subset.responsive_nm3==True)])/float(len(subset))
        resp*=100
    else:
        responsive_stim = 'responsive_'+stimulus_suffix
        if stimulus_suffix=='dg':
            reliability_stim = 'reliability_nm1a'
        else:
            reliability_stim = 'reliability_nm1b'
        for i,a in enumerate(areas):
            resp[i] = len(metrics[(metrics[responsive_stim]==True)&(metrics.area==a)])/float(len(metrics[(metrics.area==a)&np.isfinite(metrics[reliability_stim])]))
        resp*=100
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(4,4))
        ax = plt.subplot(111)
        plt.bar(range(6), resp, color=stim_colors[stimulus_suffix])
        plt.xticks(range(6), area_labels)
        plt.ylim(0,70)
        ax.grid(b=False, axis='x')
        plt.tick_params(labelsize=18)
        plt.yticks(range(0,71,10))
        plt.ylabel("Percent responsive neurons", fontsize=18)
        plt.tight_layout()
        save_figure(fig, r'/Users/saskiad/Documents/CAM/paper figures/revision/responsive_'+stimulus_suffix)