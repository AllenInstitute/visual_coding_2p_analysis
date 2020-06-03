# -*- coding: utf-8 -*-
"""
Created on Wed Jan 09 19:49:40 2019

@author: danielm
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def run_analysis():

#    savepath = r'C:\\Users\\danielm\\Desktop\\population_overlap\\'
    savepath = r'/Users/saskiad/Documents/Data/CAM/revision/population_overlap'
#    fraction_responsive_trials_path = r'C:\\Users\\danielm\\Desktop\\population_overlap\\stimulus_percents.csv'
    fraction_responsive_trials_path = r'/Users/saskiad/Documents/Data/CAM/revision/percent_trials.csv'
    fraction_responsive_trials = pd.read_csv(fraction_responsive_trials_path)
    

    
    num_shuffles = 100
    
    stim_names = ['LSN',
                  'SG',
                  'DG',
                  'NS',
                  'NM1a',
                  'NM1b',
                  'NM1c',
                  'NM2',
                  'NM3']
    
    resampled_fraction_responsive_trials = resample_all_stim(fraction_responsive_trials,stim_names,num_shuffles,savepath)

    ss_corr = stim_stim_correlation(resampled_fraction_responsive_trials,savepath)
    
    plt.figure(figsize=(8,8))
    
    ax = plt.subplot(111)
    ax.imshow(ss_corr,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='none')
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(stim_names,fontsize=14)
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(stim_names,fontsize=14)
    plt.show()
    
def stim_stim_correlation(resampled_fraction_responsive_trials,savepath):
    
    if os.path.isfile(savepath+'fraction_responsive_trials_correlation.npy'):
    
        ss_corr = np.load(savepath+'fraction_responsive_trials_correlation.npy')
        
    else:
        
        num_stim = np.shape(resampled_fraction_responsive_trials)[0]
        num_shuffles = np.shape(resampled_fraction_responsive_trials)[2]

        ss_corr = np.zeros((num_stim,num_stim,num_shuffles))
        
        for stim_1 in range(num_stim):
            for stim_2 in range(num_stim):
                print str(stim_1) + '  ' + str(stim_2)
                for ns in range(num_shuffles):

                    has_stim_1 = np.isfinite(resampled_fraction_responsive_trials[stim_1,:,ns])
                    has_stim_2 = np.isfinite(resampled_fraction_responsive_trials[stim_2,:,ns])
                    
                    has_both_stim = has_stim_1 & has_stim_2
                    
                    sample_1 = resampled_fraction_responsive_trials[stim_1,has_both_stim,ns]
                    sample_2 = resampled_fraction_responsive_trials[stim_2,has_both_stim,ns]
                    
                    (r,p) = stats.pearsonr(sample_1,sample_2)
                    
                    ss_corr[stim_1,stim_2,ns] = r

        np.save(savepath+'fraction_responsive_trials_correlation.npy',ss_corr)

    ss_corr = np.median(ss_corr,axis=2)
    
    return ss_corr
    
def resample_all_stim(fraction_responsive_trials,stim_names,num_shuffles,savepath):
    
    if os.path.isfile(savepath+'resampled_fraction_responsive_trials.npy'):
        resampled_fraction_responsive_trials = np.load(savepath+'resampled_fraction_responsive_trials.npy')
    else:
        num_cells = len(fraction_responsive_trials)
        num_stim = len(stim_names)
        
        trials_per_stim = [100,50,15,50,10,10,10,10,10]
        
        resampled_fraction_responsive_trials = np.zeros((num_stim,num_cells,num_shuffles))
        for si, stim in enumerate(stim_names):
            
            stim_str = 'percent_trials_' + stim.lower()
            has_stim = fraction_responsive_trials[stim_str].notnull()
            has_stim_idx = np.argwhere(has_stim)[:,0]
            
            stim_fraction = fraction_responsive_trials[stim_str][has_stim].values / 100.0
            num_responsive_trials = np.round(stim_fraction * trials_per_stim[si])
            
            for trials in range(trials_per_stim[si]):
                
                print stim + ' ' + str(trials)
                
                frac = trials / float(trials_per_stim[si])
                
                cells_with_trials = num_responsive_trials == trials
                
                num_cells_with_trials = np.sum(cells_with_trials)
                
                resampled_trials = np.random.choice(2,p=[1.0-frac,frac],size=(num_cells_with_trials,num_shuffles,trials_per_stim[si]))
    
                resampled_fraction_responsive_trials[si,has_stim_idx[cells_with_trials],:] = np.mean(resampled_trials,axis=2)

            resampled_fraction_responsive_trials[si,has_stim==False,:] = np.NaN
            
        np.save(savepath+'resampled_fraction_responsive_trials.npy',resampled_fraction_responsive_trials)

    return resampled_fraction_responsive_trials
    
if __name__=='__main__':
    run_analysis()