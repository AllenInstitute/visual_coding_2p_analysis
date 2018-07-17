#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:16:50 2018

@author: saskiad
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import core, os

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


def plot_strip_plot(data_input, area, plot_key, x_label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots', figname='dg_decoding_performance', Nticks=10, xlim=None, box=False, bar=False, point=False):


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

    exp_area = data_input[data_input.area==area]
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)

        ax = sns.stripplot(y='cre_depth',x=plot_key,data=exp_area,palette=cre_depth_palette,
                      order=cre_depth_list, size=10)

        if box:
            ax = sns.boxplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, whis=np.inf)
        if bar:
            ax = sns.barplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, ci=None, alpha=0.5, estimator=np.median)
        if point:
            ax = sns.pointplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list,
                             ci=None, markers='|', markersize=30, join=False)


        # if np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])]) <= 1:
        #     plt.xticks([0,0.2,0.4, 0.6,0.8,1],range(0,101,20))
        #     plt.xlim(-0.05, 1.05)
        # else:

        if xlim is None:
            xmax = int(np.ceil(np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            xmin = int(np.floor(np.amin(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)

            xmin -= np.remainder(xmin, x_spacing)

        else:
            xmin, xmax = xlim
#            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)
            x_spacing = (xmax-xmin)/float(Nticks)

#        plt.xticks(range(xmin, xmax+x_spacing, x_spacing ) )
        plt.xticks(np.arange(xmin, xmax+x_spacing, x_spacing))
#        plt.xlim(xmin-0.1, xmax+.1)
        plt.xlim(xmin-0.02, xmax+0.02)


        if xmax > 0 and xmin < 0:
            plt.axvline(x=0, lw=1, color='k')

        plt.yticks(range(19), cre_list)
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.ylabel("")
        ax.yaxis.tick_right()

        plt.tick_params(labelsize=22)#22
        plt.xlabel(x_label, fontsize=24)
        plt.title(area_dict[area], fontsize=26)#26
        fig.tight_layout()

        figname = os.path.join(fig_base_dir, figname+'_'+area)
        save_figure(fig, figname)

def plot_strip_plot_sup(data_input, area, plot_key, x_label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots', figname='dg_decoding_performance', Nticks=10, xlim=None, box=False, bar=False, point=False):


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

    exp_area = data_input[data_input.area==area]
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)

        ax = sns.stripplot(y='cre_depth',x=plot_key,data=exp_area,palette=cre_depth_palette,
                      order=cre_depth_list, size=10)

        if box:
            ax = sns.boxplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, whis=np.inf)
        if bar:
            ax = sns.barplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, ci=None, alpha=0.5, estimator=np.median)
        if point:
            ax = sns.pointplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list,
                             ci=None, markers='|', markersize=30, join=False)


        # if np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])]) <= 1:
        #     plt.xticks([0,0.2,0.4, 0.6,0.8,1],range(0,101,20))
        #     plt.xlim(-0.05, 1.05)
        # else:

        if xlim is None:
            xmax = int(np.ceil(np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            xmin = int(np.floor(np.amin(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)

            xmin -= np.remainder(xmin, x_spacing)

        else:
            xmin, xmax = xlim
#            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)
            x_spacing = (xmax-xmin)/float(Nticks)

#        plt.xticks(range(xmin, xmax+x_spacing, x_spacing ) )
        plt.xticks(np.arange(xmin, xmax+x_spacing, x_spacing))
#        plt.xlim(xmin-0.1, xmax+.1)
        plt.xlim(xmin-0.02, xmax+0.02)


        if xmax > 0 and xmin < 0:
            plt.axvline(x=0, lw=1, color='k')

        plt.yticks(range(19), cre_list)
        plt.yticks([])#sup
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.ylabel("")
        ax.yaxis.tick_right()

        plt.tick_params(labelsize=26)#22
        plt.xlabel(x_label, fontsize=24)
        plt.title(area_dict[area], fontsize=30)#26
        fig.tight_layout()

        figname = os.path.join(fig_base_dir, figname+'_'+area+'_sup')
        save_figure(fig, figname)