#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:14:06 2018

@author: saskiad

Core functions used across stimulus specific analysis scripts
"""

import os
import pwd
import numpy as np


event_path_dict = {}
event_path_dict['saskiad'] = r'/Volumes/External Data/L0_Files'
save_path_dict = {}
save_path_dict['saskiad'] = r'/Volumes/programs/braintv/workgroups/nc-ophys/Saskia/Visual Coding Event Analysis'
manifest_path_dict = {}
#manifest_path_dict['saskiad'] = r'/Volumes/aibs/technology/allensdk_data/2018-01-30T10_59_26.662324/boc/manifest.json'
manifest_path_dict['saskiad'] = r'/Volumes/External Data/BrainObservatory/manifest.json'

def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ]

def get_save_path():
    '''provides the appropriate paths for a given user
        
Returns
-------
save_path
        '''
    user_name = get_username()
    save_path = save_path_dict[user_name]
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
path to L0 event files
session_id (int)

Returns
-------
l0 event traces (numpy array)
        '''
    user_name = get_username()
    event_path = event_path_dict[user_name]
    file_path = os.path.join(event_path, str(session_id)+'_L0_events.npy')
    print "Loading L0 events from: ", file_path
    l0_events = np.load(file_path)
    return l0_events
    
def get_manifest_path():
    '''provides the path to the manifest for the boc
        
Returns
-------
manifest path
        '''
    user_name= get_username()
    return manifest_path_dict[user_name]

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c

