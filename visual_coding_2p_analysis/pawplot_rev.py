#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:32:32 2018

@author: saskiad
with paw_plot code from davidf and savefid code from dougo
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.colorbar 
import matplotlib.gridspec as gridspec
import os

def mega_paw_plot(sdata_list=None, sdata_bg_list=None, cdata_list=None, 
                  cmap=None, clim=None, 
                  edgecolor='#555555', bgcolor='#cccccc', 
                  figsize=None, cbar_orientation='horizontal', cticks=None):
    if cmap is None:
        cmap = 'plasma'
        
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
        if cticks==None:
            cbar.set_ticks(np.arange(clim[0], clim[1]+0.05, 0.1))
            cbar.set_ticklabels(np.arange(clim[0], clim[1]+0.05, 0.1))
        else:
            cbar.set_ticks(cticks)
        cbar.set_ticks([1,2,3])
#        cbar.set_ticklabels([1,2,4])
        cbar.set_ticklabels([0.02,0.04,0.08])
    return fig

    
def paw_plot(sdata=None, sdata_bg=None, cdata=None, cmap=None, clim=None, edgecolor='#555555', bgcolor='#cccccc', legend=True, ax=None):
    ax = ax if ax else plt.gca()
        
    r = np.array([ 0, 1, 1, 1, 1, 1 ])
    theta = np.array([ 0, 30, 75, 120, 165, 210 ]) * np.pi / 180.0
    theta = np.array([ -5, 25, 70, 115, 160, 205 ]) * np.pi / 180.0
#    theta = np.array([ 0, 30, 72, 114, 156, 198 ]) * np.pi / 180.0
    
    if sdata is None:
        sdata = np.array([4,1,1,1,1,1])*.08
                
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
    
    ax.add_patch(mpatch.Arc((0.0, 0.0), 2.0, 2.0, angle=0.0, 
                            theta1=theta[1]*180.0/np.pi, theta2=theta[-1]*180.0/np.pi, 
                            edgecolor=edgecolor,facecolor='none', zorder=0, linestyle='--'))
   
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
    ax.set_ylim([-.8,1.35])
    
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
    
def make_pawplot_metric_crespecific(data_input, metric, stimulus_suffix, clim=None,cticks=None, fig_base_dir=r'/Users/saskiad/Documents/CAM/paper figures/revision'):
    '''creates and saves a pawplot for a specified single cell metric

Parameters
----------
data_input: pandas dataframe
metric: string of the name of the metric represented in the paw plot
stimulus_suffix: string of the stimulus abbreviation (eg. 'dg','sg','ns')

        '''
    areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    cre_depth = [('Cux2-CreERT2',100),('Rorb-IRES2-Cre',200),('Rbp4-Cre_KL100',300), ('Ntsr1-Cre_GN220',500)]
    responsive = 'responsive_'+stimulus_suffix
#    resp = data_input[data_input[responsive]==True]
    results = np.empty((6,4,3))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(cre_depth):
            subset = data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])]
            if len(subset)>0:
                results[i,j,1] = len(subset[subset[responsive]==True])/float(len(subset))
                results[i,j,0] = subset[subset[responsive]==True][metric].median()
            else:
                results[i,j,1] = np.NaN
                results[i,j,0] = np.NaN
#            results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])])
#            results[i,j,1] = len(resp[(resp.area==a)&(resp.tld1_name==d[0])&(resp.depth_range==d[1])])
#            results[i,j,0] = resp[(resp.area==a)&(resp.tld1_name==d[0])&(resp.depth_range==d[1])][metric].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)

    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.1, results[:,1,1]*.1, results[:,2,1]*.1, results[:,3,1]*.1],
                 sdata_bg_list=[np.array([1,1,1,1,1,1])*0.1, np.array([1,1,1,1,1,1])*0.1, np.array([1,1,1,1,1,1])*0.1, np.array([1,1,0,0,0,1])*0.1],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim, cticks=cticks, edgecolor='#cccccc', figsize=(5,15))

    figname = os.path.join(fig_base_dir, metric+'_cre_pawplot')
    print figname
    save_figure(fig, figname)

def make_pawplot_fit_crespecific(data_input, metric_fit, clim=None, cticks=None, fig_base_dir=r'/Users/saskiad/Documents/CAM/paper figures/revision'):
    '''creates and saves a pawplot for a fit parameter (eg. TF or SF)

Parameters
----------
data_input: pandas dataframe
metric_fit: string of the name of the fit metric represented in the paw plot

        '''
    areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    cre_depth = [('Cux2-CreERT2',100),('Rorb-IRES2-Cre',200),('Rbp4-Cre_KL100',300), ('Ntsr1-Cre_GN220',500)]
#    resp = data_input[(data_input[metric_fit]>=0)&(data_input[metric_fit]<=4)]
    results = np.empty((6,4,3))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(cre_depth):
            subset = data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])]
            if len(subset)>0:
                resp = subset[(subset[metric_fit]>=0)&(subset[metric_fit]<=4)]
                results[i,j,1] = len(resp)/float(len(subset))
                results[i,j,0] = resp[metric_fit].median()
            else:
                results[i,j,1] = np.NaN
                results[i,j,0] = np.NaN
#            results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])])
#            results[i,j,1] = len(resp[(resp.area==a)&(resp.tld1_name==d[0])&(resp.depth_range==d[1])])
#            results[i,j,0] = resp[(resp.area==a)&(resp.tld1_name==d[0])&(resp.depth_range==d[1])][metric_fit].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)
    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.1, results[:,1,1]*.1, results[:,2,1]*.1, results[:,3,1]*.1],
                 sdata_bg_list=[np.array([1,1,1,1,1,1])*0.1, np.array([1,1,1,1,1,1])*0.1, np.array([1,1,1,1,1,1])*0.1, np.array([1,1,0,0,0,1])*0.1],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim, cticks=cticks, edgecolor='#cccccc', figsize=(5,15))
    figname = os.path.join(fig_base_dir, metric_fit+'_cre_pawplot')
    save_figure(fig, figname)
    

if __name__ == "__main__":
    mega_paw_plot(sdata_list=[ np.array([5,1,1,1,1,1])*.05, np.array([5,2,1,1,1,1])*.05, np.array([2,1,.5,.5,.5])*.0005 ],
                  sdata_bg_list=[ np.array([7,2.5,2,2,2,2])*.05, np.array([6,2.5,2,2,2,2])*0.05, np.array([2.1,2.5,2,2,2,2])*.0005 ],
                  cdata_list=[ np.random.random(6), np.random.random(6)*3, np.random.random(6)*2 ],
                  cmap='hot',
                  clim=[0.0,5.0],
                  figsize=(5,15),
                  cbar_orientation='horizontal')
    plt.show()