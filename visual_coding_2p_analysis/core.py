#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:14:06 2018

@author: saskiad

Core functions used across stimulus specific analysis scripts
"""

import os
import numpy as np
import sys


def get_save_path():
    '''provides the appropriate paths for a given platform

Returns
-------
save_path
        '''

    # if sys.platform=='win32':
    #     save_path = r'\\allen\programs\braintv\workgroups\nc-ophys\Saskia\Visual Coding Event Analysis'
    # elif sys.platform=='darwin':
    #     save_path = r'/Volumes/programs/braintv/workgroups/nc-ophys/Saskia/Visual Coding Event Analysis'
    # elif sys.platform=='linux2':
    #     save_path = r'/allen/programs/braintv/workgroups/nc-ophys/Saskia/Visual Coding Event Analysis'

    if sys.platform == 'win32':
        save_path = r'\\allen\programs\braintv\workgroups\nc-ophys\ObservatoryPlatformPaperAnalysis\analysis_files_pre_2018_3_29'
    elif sys.platform == 'darwin':
        save_path = r'/Volumes/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/analysis_files_pre_2018_3_29'
    elif sys.platform == 'linux2':
        save_path = '/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/analysis_files_pre_2018_3_29'
    return save_path


def get_stim_table(session_id, stimulus):
    '''uses allenSDK to get stimulus table, specimen IDs, and the number of cells in specific session.

Parameters
----------
session_id (int)
stimulus name (string)

Returns
-------
stim_table (pandas DataFrame)
numbercells
specimen IDs
        '''
    manifest_path = get_manifest_path()
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    data_set = boc.get_ophys_experiment_data(session_id)
    specimen_ids = data_set.get_cell_specimen_ids()
    stim_table = data_set.get_stimulus_table(stimulus)
    numbercells = len(specimen_ids)
    return stim_table, numbercells, specimen_ids

def get_L0_events(session_id):
    '''gets the L0 event time series for a given session

    Parameters
    ----------
    session_id (int)

    Returns
    -------
    l0 event traces (numpy array)
        '''

    # event_path = get_event_path()
    # event_file = os.path.join(event_path, str(session_id)+'.npz')
    # print "Loading L0 events from: ", event_file
    # events = np.load(event_file)['ev']
    # return events

    print("Loading L0 events for: " + str(session_id))
    manifest_path = get_manifest_path()
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    data_set = boc.get_ophys_experiment_data(session_id)
    from l0_analysis import L0_analysis
    l0 = L0_analysis(data_set)
    events = l0.get_events()
    return events


def get_manifest_path():
    '''provides the path to the manifest for the boc

Returns
-------
manifest path
        '''
    if sys.platform=='win32':
        manifest_path = r"\\allen\programs\braintv\workgroups\nc-ophys\ObservatoryPlatformPaperAnalysis\platform_boc_pre_2018_3_16\manifest.json"
    elif sys.platform=='darwin':
        manifest_path = r"/Volumes/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json"
    elif sys.platform=='linux2':
        manifest_path = r"/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json"


    return manifest_path

def get_cache_path():
    '''returns the cache for  the L0 event files

Returns
-------
cache path
        '''
    if sys.platform=='win32':
        cache_path = r"\\allen\programs\braintv\workgroups\nc-ophys\ObservatoryPlatformPaperAnalysis\events_pre_2018_3_29"
    elif sys.platform=='darwin':
        cache_path = r'/Volumes/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/events_pre_2018_3_29/'
    elif sys.platform=='linux2':
        cache_path = r'/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/events_pre_2018_3_29/'
    return cache_path

def get_event_path():
    '''returns the path for the L0 event files
This should be deprecated now. The l0_analysis module doesn't use hash anymore, so no need to use this path.
Returns
-------
event path
        '''
    if sys.platform=='win32':
        event_path = r'\\allen\aibs\technology\allensdk_data\platform_events_pre_2018_3_19\events_2'
    elif sys.platform=='darwin':
        event_path = r'/Volumes/aibs/technology/allensdk_data/platform_events_pre_2018_3_19/events_2'
    elif sys.platform=='linux2':
        event_path = r'/allen/aibs/technology/allensdk_data/platform_events_pre_2018_3_19/events_2'
    return event_path

def get_running_speed(session_id):
    '''uses allenSDK to get the running speed for a specified session

Parameters
----------
session_id

Returns
-------
running speed
        '''
    manifest_path = get_manifest_path()
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    data_set = boc.get_ophys_experiment_data(session_id)
    running_speed,_ = data_set.get_running_speed()
    return running_speed


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c

def get_cre_colors():
    '''returns dictionary of colors for specific Cre lines

Returns
-------
cre color dictionary
        '''
    cre_colors = {}
    cre_colors['Emx1-IRES-Cre'] = '#9f9f9f'
    cre_colors['Slc17a7-IRES2-Cre'] = '#5c5c5c'
    cre_colors['Cux2-CreERT2'] = '#a92e66'
    cre_colors['Rorb-IRES2-Cre'] = '#7841be'
    cre_colors['Scnn1a-Tg3-Cre'] = '#4f63c2'
    cre_colors['Nr5a1-Cre'] = '#5bb0b0'
    cre_colors['Fezf2-CreER'] = '#3A6604'
    cre_colors['Tlx3-Cre_PL56'] = '#99B20D'
    cre_colors['Rbp4-Cre_KL100'] = '#5cad53'
    cre_colors['Ntsr1-Cre_GN220'] = '#875e24'
    cre_colors['Sst-IRES-Cre'] = '#ff944d'
    cre_colors['Vip-IRES-Cre'] = '#ffe066'
    return cre_colors


