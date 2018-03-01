# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:32:36 2018

@author: danielm
"""

import numpy as np

def trial_response_significance(sweep_events,num_shuffles=10000):
    
    # inputs:
    # sweep_events: 2D numpy array (num_sweeps X frames_per_sweep)
    # outputs: p-values for each sweep (1D numpy array of length num_sweeps)
    
    num_sweeps = np.shape(sweep_events)[0]
    frames_per_sweep = np.shape(sweep_events)[1]
    num_shuffles = int(num_shuffles / num_sweeps) + 1#each sweep contributes one shuffle to the null distribution

    null_distribution = np.zeros((num_sweeps,num_shuffles))
    shuffled_events = np.zeros(np.shape(sweep_events))
    for sh in range(num_shuffles):
        for f in range(frames_per_sweep):
            idx = np.random.permutation(num_sweeps)
            shuffled_events[:,f] = sweep_events[idx,f]
        null_distribution[:,sh] = np.sum(shuffled_events,axis=1)
    null_distribution = null_distribution.flatten()
    
    actual_event_counts = np.sum(sweep_events,axis=1)
    null_dist_mat = np.tile(null_distribution,reps=(num_sweeps,1))
    actual_is_less = actual_event_counts.reshape(num_sweeps,1) <= null_dist_mat
    p_values = np.mean(actual_is_less,axis=1)
    
    return p_values


#def trial_significance(sweep_events, num_shuffles=10000):
#    num_sweeps = np.shape(sweep_events)[0]
#    num_shuffles = int(num_shuffles / num_sweeps) #each sweep contributes one shuffle to the null distribution