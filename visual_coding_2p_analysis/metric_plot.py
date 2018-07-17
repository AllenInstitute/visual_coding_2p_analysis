#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:38:25 2018

@author: saskiad
with save_figure code from dougo
"""
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_metric_box_cre_depth(data_input,area, metric, stimulus_suffix, label, xlim=None):
    '''creates and saves a boxplot of metric for Cre/layers in specified area

Parameters
----------
data_input: pandas dataframe. 
area: string of the visual area to plot
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
    visp = data_input[(data_input['area']==area)&(data_input[responsive]==True)]
    
    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)
    
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        sns.boxplot(y='cre_depth',x=metric,data=visp, palette=cre_depth_palette, order=cre_depth_list, 
                    saturation=0.7, width=0.7)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric].median(), [i], 'bo')
            
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.tick_params(labelsize=22)#26#18
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.title(area_dict[area], fontsize=26)#30#20
#        plt.xlim(0,1250)#rf area
#        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        plt.xticks([0,25,50,75,100])
#        plt.xticks([0,250,500,750,1000,1250]) #rf area
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric+'_'+area+'_cre_box'
        save_figure(fig, figname)

def plot_metric_box_cre_depth_sup(data_input,area, metric, stimulus_suffix, label, xlim=None):
    '''creates and saves a boxplot of metric for Cre/layers in specified area, formatted for supplemental figures

Parameters
----------
data_input: pandas dataframe. 
area: string of the cortical area
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
    visp = data_input[(data_input['area']==area)&(data_input[responsive]==True)]
    
    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)
    
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        sns.boxplot(y='cre_depth',x=metric,data=visp, palette=cre_depth_palette, order=cre_depth_list, 
                    saturation=0.7, width=0.7)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric].median(), [i], 'bo')
            
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.yticks([]) #sup
        plt.tick_params(labelsize=26)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.title(area_dict[area], fontsize=30)
#        plt.xlim(0,1250)#rf area
#        plt.xticks([0,0.2,0.4,0.6,0.8,1])
#        plt.xticks([0,250,500,750,1000,1250]) #rf area
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric+'_'+area+'_cre_box_sup'
        save_figure(fig, figname)

 
def plot_metric_fit_cre_depth_box(data_input,area, metric_fit, label):
    '''creates and saves a boxplot of a fit metric for Cre/layers in specified area

Parameters
----------
data_input: pandas dataframe. 
area: string of the cortical area
metric: string of the name of the metric 
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

    visp = data_input[(data_input.area==area)&(data_input[metric_fit]>=0)&(data_input[metric_fit]<=4)]
    
    if not np.issubdtype(visp[metric_fit].dtype, np.number):
        visp[metric_fit] = visp[metric_fit].astype(float)
    
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        ax = sns.boxplot(y='cre_depth',x=metric_fit, data=visp, palette=cre_depth_palette, order=cre_depth_list, 
                    saturation=0.7, width=0.7)
        for i,c in enumerate(cre_depth_list):
            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric_fit].median(), [i], 'bo')
            
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.yticks(range(19), cre_list)
        plt.yticks([]) #sup
        if metric_fit=='fit_tf_ind_dg':
            plt.xticks(range(5), [1,2,4,8,15])
        if metric_fit=='fit_sf_ind_sg':
            plt.xticks(range(5), [0.02, 0.04, 0.08, 0.16, 0.32])
        plt.tick_params(labelsize=22)#26
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
        plt.title(area_dict[area], fontsize=26)#30

        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric_fit+'_'+area+'_cre_box_sup'
        save_figure(fig, figname)
       
#def plot_metric_fit_box_cre_depth(data_input,area, metric_fit, label):
#    '''creates and saves a violinplot of metric for Cre/layers in VISp
#
#Parameters
#----------
#data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
#metric: string of the name of the metric 
#stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')
#label: string for the y-axis label
#
#        '''
#    cre_depth_list = [('Emx1-IRES-Cre',100), ('Slc17a7-IRES2-Cre', 100),('Cux2-CreERT2',100),('Vip-IRES-Cre',100),
#                  ('Emx1-IRES-Cre',200),('Slc17a7-IRES2-Cre', 200),('Cux2-CreERT2',200),
#                  ('Rorb-IRES2-Cre',200),('Scnn1a-Tg3-Cre',200),('Nr5a1-Cre',200),('Sst-IRES-Cre',200),('Vip-IRES-Cre',200),
#                  ('Emx1-IRES-Cre',300),('Slc17a7-IRES2-Cre', 300),
#                  ('Rbp4-Cre_KL100',300),('Fezf2-CreER',300),('Tlx3-Cre_PL56',300),('Sst-IRES-Cre',300),
#                  ('Ntsr1-Cre_GN220',500)]
#    cre_color_dict = core.get_cre_colors()
#    cre_depth_colors = []
#    cre_list = []
#    for c in cre_depth_list:
#        cre_list.append(c[0].split('-')[0])
#        cre_depth_colors.append(cre_color_dict[c[0]])
#    cre_depth_palette = sns.color_palette(cre_depth_colors)
#
#    visp = data_input[(data_input.area==area)&(data_input[metric_fit]>=0)&(data_input[metric_fit]<=4)]
#    
#    if not np.issubdtype(visp[metric_fit].dtype, np.number):
#        visp[metric_fit] = visp[metric_fit].astype(float)
#    
#    with sns.axes_style('whitegrid'):
#        fig = plt.figure(figsize=(6,16))
#        ax = fig.add_subplot(111)
#        ax = sns.boxplot(y='cre_depth',x=metric_fit, data=visp, palette=cre_depth_palette, order=cre_depth_list, 
#                    saturation=0.7, width=0.7)
##        ax = sns.violinplot(y='cre_depth', x=metric_fit, data=visp, inner='quartile', cut=0,scale='area', 
##                            linewidth=1, saturation=0.7, order=cre_depth_list, palette=cre_depth_palette)
#        for i,c in enumerate(cre_depth_list):
#            ax.plot(visp[(visp.tld1_name==c[0])&(visp.depth_range==c[1])][metric_fit].median(), [i], 'bo')
#            
#        plt.axhline(y=3.5, lw=7, color='w')
#        plt.axhline(y=11.5, lw=7, color='w')
#        plt.axhline(y=17.5, lw=7, color='w')
#        plt.yticks(range(19), cre_list)
#        if metric_fit=='fit_tf_ind_dg':
#            plt.xticks(range(5), [1,2,4,8,15])
#        if metric_fit=='fit_sf_ind_sg':
#            plt.xticks(range(5), [0.02, 0.04, 0.08, 0.15, 0.32])
#        plt.tick_params(labelsize=20)
#        ax.yaxis.tick_right()
#        plt.ylabel("")
#        plt.xlabel(label, fontsize=24)
#        plt.title(area_dict[area], fontsize=24)
##        plt.text(0.96,1.5, "2/3", fontsize=14)
##        plt.text(0.99,7.5, "4", fontsize=14)
##        plt.text(0.99,14.5, "5", fontsize=14)
##        plt.text(0.99,18.1, "6", fontsize=14)
#        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
#        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
#        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
#        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
#        fig.tight_layout()
#        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric_fit+'_'+area+'_cre_box'
#        save_figure(fig, figname)
        
def plot_metric_cre_depth_run(data_input, metric, area, label):
    '''creates and saves a violinplot of running modulation metric for Cre/layers in specified area

Parameters
----------
data_input: pandas dataframe. If dataframe does not have a column "cre_depth" it will create it, which is slow
metric: string of the name of the metric 
area: string of the cortical area
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

    visp = data_input[(data_input.area==area)&(data_input.run_pval_dg<0.05)]
    
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
        plt.tick_params(labelsize=22)
        ax.yaxis.tick_right()
        plt.ylabel("")
        plt.xlabel(label, fontsize=20)
        plt.title(area_dict[area], fontsize=26)
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric+area+'_cre'
        save_figure(fig, figname)
        

def plot_metric_box_cre_depth_noresp(data_input, area, metric, label, xlim=None):
    '''creates and saves a boxplot of metric for Cre/layers in specified area with not responsiveness criteria

Parameters
----------
data_input: pandas dataframe. 
area: string of the cortical area
metric: string of the name of the metric 
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

    visp = data_input[(data_input.area==area)]
    
    if not np.issubdtype(visp[metric].dtype, np.number):
        visp[metric] = visp[metric].astype(float)
    
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)
        sns.boxplot(y='cre_depth',x=metric,data=visp, palette=cre_depth_palette, order=cre_depth_list, 
                    saturation=0.7, width=0.7)
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
        plt.title(area_dict[area], fontsize=20)
#        plt.xlim(0,1250)#rf area
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        fig.tight_layout()
        figname = r'/Users/saskiad/Documents/CAM/paper figures/'+metric+'_'+area+'_cre_box'
        save_figure(fig, figname)
