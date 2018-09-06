#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:38:25 2018

@author: saskiad
with save_figure code from dougo
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import core

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



def plot_metric_cre_depth(data_input, metric, stimulus_suffix, label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots'):
    '''creates and saves a violinplot of metric for Cre/layers in VISp

Parameters
----------
data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
metric: string of the name of the metric
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')
label: string for the y-axis label

        '''
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

    responsive = 'responsive_'+stimulus_suffix
    visp = data_input[(data_input.area=='VISp')&(data_input[responsive]==True)]

    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)

    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.violinplot(y='cre_depth', x=metric, data=visp, inner='quartile', cut=0,scale='area',
                            linewidth=1, saturation=0.7, order=cre_depth_list, palette=cre_depth_palette)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric].median(), [i], 'bo')

        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.tick_params(labelsize=18)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
#        plt.text(0.96,1.5, "2/3", fontsize=14)
#        plt.text(0.99,7.5, "4", fontsize=14)
#        plt.text(0.99,14.5, "5", fontsize=14)
#        plt.text(0.99,18.1, "6", fontsize=14)
#        plt.title("VISp", fontsize=16)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()

        figname = os.path.join(fig_base_dir, metric + '_visp_cre')
        save_figure(fig, figname)

def plot_metric_fit_cre_depth(data_input, metric_fit, label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots'):
    '''creates and saves a violinplot of metric for Cre/layers in VISp

Parameters
----------
data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
metric: string of the name of the metric
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')
label: string for the y-axis label

        '''
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

    visp = data_input[(data_input.area=='VISp')&(data_input[metric_fit]>=0)&(data_input[metric_fit]<=4)]

    if not np.issubdtype(visp[metric_fit].dtype, np.number):
        visp[metric_fit] = visp[metric_fit].astype(float)

    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.violinplot(y='cre_depth', x=metric_fit, data=visp, inner='quartile', cut=0,scale='area',
                            linewidth=1, saturation=0.7, order=cre_depth_list, palette=cre_depth_palette)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric_fit].median(), [i], 'bo')

        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.tick_params(labelsize=18)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
#        plt.text(0.96,1.5, "2/3", fontsize=14)
#        plt.text(0.99,7.5, "4", fontsize=14)
#        plt.text(0.99,14.5, "5", fontsize=14)
#        plt.text(0.99,18.1, "6", fontsize=14)
#        plt.title("VISp", fontsize=16)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()
        figname = os.path.join(fig_base_dir, metric_fit + '_visp_cre')
        save_figure(fig, figname)

def plot_metric_cre_depth_run(data_input, metric, label):
    '''creates and saves a violinplot of running modulation metric for Cre/layers in VISp

Parameters
----------
data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
metric: string of the name of the metric
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')
label: string for the y-axis label

        '''
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

    visp = data_input[(data_input.area=='VISp')&(data_input.run_pval_dg<0.05)]

    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)

    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.violinplot(y='cre_depth', x=metric, data=visp, inner='quartile', cut=0,scale='area',
                            linewidth=1, saturation=0.7, order=cre_depth_list, palette=cre_depth_palette)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric].median(), [i], 'bo')

        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.tick_params(labelsize=18)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
#        plt.text(0.96,1.5, "2/3", fontsize=14)
#        plt.text(0.99,7.5, "4", fontsize=14)
#        plt.text(0.99,14.5, "5", fontsize=14)
#        plt.text(0.99,18.1, "6", fontsize=14)
#        plt.title("VISp", fontsize=16)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric+'_visp_cre'
        save_figure(fig, figname)

def plot_metric_population_cre_depth(data_input, metric, label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots'):
    '''creates and saves a violinplot of population metric for Cre/layers in VISp

Parameters
----------
data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
metric: string of the name of the metric
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')
label: string for the y-axis label

        '''
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

    visp = data_input[(data_input.area=='VISp')]

    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)

    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.violinplot(y='cre_depth', x=metric, data=visp, inner='quartile', cut=0,scale='area',
                            linewidth=1, saturation=0.7, order=cre_depth_list, palette=cre_depth_palette)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric].median(), [i], 'bo')

        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.tick_params(labelsize=18)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
#        plt.text(0.96,1.5, "2/3", fontsize=14)
#        plt.text(0.99,7.5, "4", fontsize=14)
#        plt.text(0.99,14.5, "5", fontsize=14)
#        plt.text(0.99,18.1, "6", fontsize=14)
#        plt.title("VISp", fontsize=16)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()

        figname = os.path.join(fig_base_dir, metric + '_visp_cre')
        save_figure(fig, figname)
