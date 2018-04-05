#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:58:18 2018

@author: saskiad
with paw_plot code from davidf and savefig from dougo

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.colorbar
import matplotlib.gridspec as gridspec
import os

def mega_paw_plot(sdata_list=None, sdata_bg_list=None, cdata_list=None,
                  cmap=None, clim=None,
                  edgecolor='#555555', bgcolor='#cccccc',
                  figsize=None, cbar_orientation='horizontal'):
    if cmap is None:
        cmap = 'magma'

    lens = []
    for dl in [ sdata_list, sdata_bg_list, cdata_list ]:
        if dl is not None:
            lens.append(len(dl))

    if len(np.unique(lens)) != 1:
        raise Exception("sdata_list, sdata_bg_list, and cdata_list must have the same length")

    nrows = len(sdata_list)
    fig = plt.figure(figsize=figsize)
    if cbar_orientation == 'vertical':
        gs = gridspec.GridSpec(nrows, 2, width_ratios=[10, 1])
    elif cbar_orientation == 'horizontal':
        gs = gridspec.GridSpec(nrows+1, 1, height_ratios=[10]*nrows + [1])

    for i, (sdata, sdata_bg, cdata) in enumerate(zip(sdata_list, sdata_bg_list, cdata_list)):
        ax = plt.subplot(gs[i,0])
        paw_plot(sdata, sdata_bg, cdata, cmap, clim, edgecolor, bgcolor, legend=False, ax=ax)

    if cdata_list:
        if cbar_orientation == 'vertical':
            cbar_ax = plt.subplot(gs[:,1])
        elif cbar_orientation == 'horizontal':
            cbar_ax = plt.subplot(gs[-1])

        if clim:
            norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        else:
            norm = mcolors.Normalize(vmin=min([cdata.min() for cdata in cdata_list]),
                                     vmax=max([cdata.max() for cdata in cdata_list]))
        cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                                norm=norm,
                                                orientation=cbar_orientation)
        cbar.ax.tick_params(labelsize=16)
    return fig


def paw_plot(sdata=None, sdata_bg=None, cdata=None, cmap=None, clim=None, edgecolor='#555555', bgcolor='#cccccc', legend=True, ax=None):
    ax = ax if ax else plt.gca()

    r = np.array([ 0, 1, 1, 1, 1, 1 ])
    theta = np.array([ 0, 30, 75, 120, 165, 210 ]) * np.pi / 180.0

    if sdata is None:
        sdata = np.ones(len(r)) * 1000

    if cdata is None:
        cdata = np.ones(len(r))

    if cmap is None:
        cmap = 'magma'

    if clim is None:
        clim = [0,1]

    xx = r * np.cos(theta)
    yy = r * np.sin(theta)

    patches = [ mpatch.Circle((x,y),rad) for (x,y,rad) in zip(xx,yy,np.sqrt(sdata)) ]
    col = PatchCollection(patches, zorder=2)
    col.set_array(cdata)
    col.set_cmap(cmap)
    col.set_edgecolor(edgecolor)
    col.set_clim(vmin=clim[0], vmax=clim[1])

    ax.add_patch(mpatch.Circle((0,0),1,edgecolor=edgecolor,facecolor='none', zorder=0, linestyle='--'))
    ax.add_collection(col)


    if legend:
        plt.colorbar(col)

    if sdata_bg is not None:
        patches = [ mpatch.Circle((x,y),rad) for (x,y,rad) in zip(xx,yy,np.sqrt(sdata_bg)) ]
        col = PatchCollection(patches, zorder=1)
        col.set_edgecolor(edgecolor)
        col.set_facecolor(bgcolor)
        ax.add_collection(col)

    ax.axis('equal')
    ax.axis('off')

    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.05,1.35])

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

def make_pawplot_metric(data_input, metric, stimulus_suffix, clim=None, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots'):
    '''creates and saves a pawplot for a specified single cell metric

Parameters
----------
data_input: pandas dataframe
metric: string of the name of the metric represented in the paw plot
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')

        '''
    areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = [100,200,300,500]
    responsive = 'responsive_'+stimulus_suffix
    resp = data_input[data_input[responsive]==True]
    results = np.empty((6,4,3))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(depths):
            results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.depth_range==d)])
            results[i,j,1] = len(resp[(resp.area==a)&(resp.depth_range==d)])
            results[i,j,0] = resp[(resp.area==a)&(resp.depth_range==d)][metric].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)

    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.000035, results[:,1,1]*.000035, results[:,2,1]*.000035, results[:,3,1]*.000035],
                 sdata_bg_list=[results[:,0,2]*.000035, results[:,1,2]*.000035, results[:,2,2]*.000035, results[:,3,2]*.000035],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim, edgecolor='#cccccc', figsize=(5,15))

    figname = os.path.join(fig_base_dir, metric+'_pawplot')
    save_figure(fig, figname)

def make_pawplot_fit(data_input, metric_fit, clim=None):
    '''creates and saves a pawplot for a fit parameter (eg. TF or SF)

Parameters
----------
data_input: pandas dataframe
metric: string of the name of the metric represented in the paw plot

        '''
    areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = [100,200,300,500]
    resp = data_input[(data_input[metric_fit]>=0)&(data_input[metric_fit]<=4)]
    results = np.empty((6,4,3))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(depths):
            results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.depth_range==d)])
            results[i,j,1] = len(resp[(resp.area==a)&(resp.depth_range==d)])
            results[i,j,0] = resp[(resp.area==a)&(resp.depth_range==d)][metric_fit].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)
    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.000035, results[:,1,1]*.000035, results[:,2,1]*.000035, results[:,3,1]*.000035],
                 sdata_bg_list=[results[:,0,2]*.000035, results[:,1,2]*.000035, results[:,2,2]*.000035, results[:,3,2]*.000035],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim, edgecolor='#cccccc', figsize=(5,15))
    figname = os.path.join(fig_base_dir, metric_fit+'_pawplot')
    save_figure(fig, figname)

def make_pawplot_run(data_input, clim=None):
    '''creates and saves a pawplot for a running modulation in drifting gratings

Parameters
----------
data_input: pandas dataframe
metric: string of the name of the metric represented in the paw plot
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')

        '''
    areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    depths = [100,200,300,500]
    data_input = data_input.dropna(subset=['run_pval_dg'])
    resp = data_input[data_input.run_pval_dg<0.05]
    results = np.empty((6,4,3))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(depths):
            results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.depth_range==d)])
            results[i,j,1] = len(resp[(resp.area==a)&(resp.depth_range==d)])
            results[i,j,0] = resp[(resp.area==a)&(resp.depth_range==d)]['run_mod_dg'].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)

    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.00006, results[:,1,1]*.00006, results[:,2,1]*.00006, results[:,3,1]*.00006],
                 sdata_bg_list=[results[:,0,2]*.00006, results[:,1,2]*.00006, results[:,2,2]*.00006, results[:,3,2]*.00006],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim,
                 cmap='PuOr_r',edgecolor='#cccccc', figsize=(5,15))
    figname = os.path.join(fig_base_dir, 'run_mod_dg_pawplot')
    save_figure(fig, figname)


#population pawplot hasn't been tested yet
def make_pawplot_population(results, filename, clim=None, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots'):
    '''creates and saves a pawplot for a population level metric (eg. with no annulus)

Parameters
----------
results: numpy array of shape (6,4,2) with [:,:,0] representing the value and [:,:,1] representing the number of datasets
filename: string to be used in creating file name for saving figure

        '''
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)
    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.005, results[:,1,1]*.005, results[:,2,1]*.005, results[:,3,1]*.005],
                 sdata_bg_list=[results[:,0,1]*.005, results[:,1,1]*.005, results[:,2,1]*.005, results[:,3,1]*.005],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]],
                 clim=clim, edgecolor='#cccccc', figsize=(5,15))

    figname = os.path.join(fig_base_dir, filename+'_pawplot')
    save_figure(fig, figname)
