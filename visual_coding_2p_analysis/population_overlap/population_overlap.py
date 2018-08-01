# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:32:16 2017

@author: danielm
"""

#import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

class PopulationOverlap:
    
    def __init__(self,cells,stim,stim_names,savepath,num_shuffles=100,min_cells=20):
        
        self.cells = cells
        self.stim = stim
        self.stim_names = stim_names
        self.savepath = savepath
        self.num_shuffles = num_shuffles
        self.min_cells = min_cells
        
        self.calculate_overlap()
        
    def calculate_overlap(self):
        
        num_stim = len(self.stim)
        
        self.intersections = []
        self.prod_marg = []
        self.zscores = []
        self.maxes = []
        self.mins = []
        self.frac_1 = []
        self.frac_2 = []
        self.num_cells = []
        
        for i in range(num_stim):
            for j in range(num_stim):
                if i<j:
                    
                    frac_stim_1, frac_stim_2, frac_stim_12, prod_marginals, cells_per_exp = self.__selectivity_by_container(self.stim[i],self.stim[j])                               
                    num_ec = len(frac_stim_1)
                    
                    z_scores = np.zeros((num_ec,))
                    for ec in range(num_ec):
                        z_scores[ec] = self.__shuffle_marginals(frac_stim_1[ec],frac_stim_2[ec],frac_stim_12[ec],cells_per_exp[ec])
                    print 'processing ' + self.stim[i] + ' versus ' + self.stim[j]                
                    
                    max_overlap = np.zeros((num_ec,))
                    min_overlap = np.zeros((num_ec,))
                    for ec in range(num_ec):
                        # overlap can be no larger than the lesser of the two marginals
                        max_overlap[ec] = np.min([frac_stim_1[ec],frac_stim_2[ec]])
                        # overlap has a minimum greater than zero if the sum of the marginals is greater than 1.0
                        if ([frac_stim_1[ec] + frac_stim_2[ec]]) > 1.0:                 
                            min_overlap[ec] = (frac_stim_1[ec] + frac_stim_2[ec]) - 1.0
                                        
                    self.intersections.append(frac_stim_12)
                    self.prod_marg.append(prod_marginals)
                    self.zscores.append(z_scores)
                    self.maxes.append(max_overlap)
                    self.mins.append(min_overlap)
                    self.frac_1.append(frac_stim_1)
                    self.frac_2.append(frac_stim_2)
                    self.num_cells.append(cells_per_exp)
        
    def plot_normalized_barplot_and_scatterplot(self):
        
        print 'Plotting scatterplots and normalized barplots...'
        
        num_stim = len(self.stim_names)
        
        plt.figure(figsize=(16,16))
        curr = 0
        for i in range(num_stim):
            for j in range(num_stim):
                if i<j:
                    
                    ax = plt.subplot(num_stim,num_stim,self.__get_subplot_position(j,i))                                 
                    self.__make_single_normalized_barplot(ax,i,j,curr)
                    
                    ax = plt.subplot(num_stim,num_stim,self.__get_subplot_position(i,j))                 
                    self.__make_single_pairwise_scatterplot(ax,i,j,curr)
        
                    curr += 1                             
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0) 
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        plt.savefig(self.savepath+'population_overlap_scatter_and_bar_plot.png')
        plt.show()
    
    def plot_unnormalized_barplot(self):
        
        print 'Plotting unnormalized barplots...'
        
        num_stim = len(self.stim_names)
        
        #bar plot that shows entire range of possible overlap unnormalized    
        plt.figure(figsize=(8,8))
        curr = 0
        for i in range(num_stim):
            for j in range(num_stim):
                if i<j:
                    ax = plt.subplot(num_stim,num_stim,self.__get_subplot_position(i,j)) 
                    self.__make_single_unnormalized_barplot(ax,i,j,curr)
                    curr += 1     
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        plt.savefig(self.savepath+'population_overlap_full_bar.png',dpi=600)            
        plt.show()

    def plot_stim_pair_effects(self):
        
        print 'Plotting stimulus pair effects matrix...'
        
        stim_pair_effects = self.__get_stim_pair_effects()
        
        num_stim = len(self.stim_names)   
        
        plt.figure(figsize=(8,8))
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(stim_pair_effects,origin='lower',interpolation='none',cmap='RdBu_r',vmin=-1.0,vmax=1.0)
        ax.set_yticks(range(num_stim))    
        ax.set_yticklabels(self.stim_names,fontsize=12)
        ax.yaxis.tick_right()
        ax.set_xticks(range(num_stim))
        ax.set_xticklabels(self.stim_names,fontsize=12)
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        plt.savefig(self.savepath+'population_overlap_medians.pdf',format='pdf')     
        plt.show()

    def __get_subplot_position(self,i,j):
        
        num_stim = len(self.stim_names)
        
        subplot_pos = np.arange(num_stim*num_stim).reshape(num_stim,num_stim)+1  
        return subplot_pos[i,j]
    
    def __get_stim_pair_effects(self):
        
        num_stim = len(self.stim_names)
        
        curr = 0
        stim_pair_effects = np.ones((num_stim,num_stim))
        for i in range(num_stim):
            for j in range(num_stim):
                if i<j:
                    stim_pair_effects[i,j] = self.__get_pairwise_effect(curr) 
                    stim_pair_effects[j,i] = self.__get_pairwise_effect(curr)
                    curr += 1
                    
        return stim_pair_effects
    
    def __get_pairwise_effect(self,curr):
        
        actual_overlap = self.intersections[curr]
        chance_overlap = self.prod_marg[curr]
        max_overlap = self.maxes[curr]
        min_overlap = self.mins[curr]        

        num_ec = len(actual_overlap)    
        
        effect_magnitude = np.zeros((num_ec,))
        effect_direction = np.zeros((num_ec,))
        for ec in range(num_ec):
            effect_direction[ec] = np.sign(actual_overlap[ec]-chance_overlap[ec])
            if effect_direction[ec] > 0:
                effect_magnitude[ec] = (actual_overlap[ec]-chance_overlap[ec])/(max_overlap[ec]-chance_overlap[ec])
            else: 
                effect_magnitude[ec] = (chance_overlap[ec]-actual_overlap[ec])/(chance_overlap[ec]-min_overlap[ec])
        
        if num_ec > 0:   
            sort_idx = np.argsort(effect_magnitude*effect_direction)             
            return effect_magnitude[sort_idx[num_ec/2]] * effect_direction[sort_idx[num_ec/2]]
    
        return 0.0        
    
    def __zscore_to_color(self,z_scores):
        
        max_zscore = -np.log10(1.0/float(self.num_shuffles))
        
        num_ec = len(z_scores)    
        
        sig_color = np.ones((num_ec,3))
        for ec in range(num_ec):        
            rgb = cm.get_cmap('bwr')(z_scores[ec]/max_zscore+0.5)[:3] 
            sig_color[ec,:] = rgb   
        
        return sig_color
    
    def __make_single_normalized_barplot(self,ax,i,j,curr,axis_ub=1.0):
        
        frac_stim_12 = self.intersections[curr]
        prod_marginals = self.prod_marg[curr]
        max_overlap = self.maxes[curr]
        min_overlap = self.mins[curr]
        cells_per_exp = self.num_cells[curr]
        z_scores = self.zscores[curr]        

        num_ec = len(frac_stim_12)
        num_stim = len(self.stim_names)  
         
        #calculate bar dimensions
        bar_width = axis_ub/float(num_ec)
        bar_height = np.zeros((num_ec,))
        for ec in range(num_ec):
            if prod_marginals[ec] < frac_stim_12[ec]: #white, red, gray, white
                bar_height[ec] = axis_ub/2.0*(frac_stim_12[ec]-prod_marginals[ec])/(max_overlap[ec]-prod_marginals[ec])
            else: # white, blue, gray, white
                bar_height[ec] = axis_ub/2.0*(prod_marginals[ec]-frac_stim_12[ec])/(prod_marginals[ec]-min_overlap[ec])
        
        #sort in ascending order of effect size     
        sort_idx = np.argsort(bar_height*np.sign(frac_stim_12-prod_marginals))  
        frac_stim_12 = frac_stim_12[sort_idx]
        prod_marginals = prod_marginals[sort_idx]
        bar_height = bar_height[sort_idx]
        cells_per_exp = cells_per_exp[sort_idx]
        max_overlap = max_overlap[sort_idx]
        min_overlap = min_overlap[sort_idx]
        z_scores = z_scores[sort_idx]
        
        sig_color = self.__zscore_to_color(z_scores)
        
        #plot bars
        for ec in range(num_ec):
            if prod_marginals[ec] < frac_stim_12[ec]: #white, red, gray, white
                p0 = patches.Rectangle(xy=(ec*bar_width,axis_ub/2.0),height=bar_height[ec],width=bar_width,facecolor=sig_color[ec,:],edgecolor='none',fill=True)
            else: # white, blue, gray, white
                p0 = patches.Rectangle(xy=(ec*bar_width,axis_ub/2.0-bar_height[ec]),height=bar_height[ec],width=bar_width,facecolor=sig_color[ec,:],edgecolor='none',fill=True)         
            ax.add_patch(p0)                
        
        #stimulus names on sides of grid
        if i==0:
            ax.set_ylabel(self.stim_names[j],fontsize=14)
        if j==num_stim-1:
            ax.set_xlabel(self.stim_names[i],fontsize=14)
    
        #show zero line
        ax.plot([0,axis_ub],[axis_ub/2.0,axis_ub/2.0],'k',linewidth=1.5)
        
        #place a triangle to indicate median effect size
        triangle_width = axis_ub/10.0
        triangle_height = axis_ub/20.0    
        if num_ec>0:
            med_height = axis_ub/2.0 + bar_height[num_ec/2] * np.sign(frac_stim_12[num_ec/2]-prod_marginals[num_ec/2])            
            triangle_pos = [[0,med_height],[triangle_width,med_height-triangle_height],[triangle_width,med_height+triangle_height]]                
            triangle = plt.Polygon(triangle_pos,color='k')  
            ax.add_patch(triangle)
        
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('square')
        ax.set_xlim([0,axis_ub])
        ax.set_ylim([0,axis_ub])
    
    def __make_single_unnormalized_barplot(self,ax,i,j,curr):
    
        chance_overlap = self.prod_marg[curr]
        max_overlap = self.maxes[curr]
        min_overlap = self.mins[curr]
        actual_overlap = self.intersections[curr]
        
        num_stim = len(self.stim_names)
    
        sort_idx = np.argsort(chance_overlap)
        max_overlap = max_overlap[sort_idx]
        actual_overlap =  actual_overlap[sort_idx]
        chance_overlap = chance_overlap[sort_idx]
        min_overlap = min_overlap[sort_idx]
        
        white = [1.0,1.0,1.0]
        gray = [0.8,0.8,0.8]
        red = [1.0,0.0,0.0]
        blue = [0.0,0.0,1.0]
    
        num_ec = len(max_overlap)
        bar_height = 1.0 / float(num_ec)
        for ec in range(num_ec):
            
            # gray bar drawn up to the maximum possible overlap
            p0 = patches.Rectangle(xy=(0.0,ec*bar_height),height=bar_height,width=max_overlap[ec],facecolor=gray,edgecolor='none',fill=True)
    
            # red or blue bar to show deviation of actual overlap from chance overlap
            if actual_overlap[ec] > chance_overlap[ec]:#red bar
                p1 = patches.Rectangle(xy=(chance_overlap[ec],ec*bar_height),height=bar_height,width=actual_overlap[ec]-chance_overlap[ec],facecolor=red,edgecolor='none',fill=True)
            else:#blue bar
                p1 = patches.Rectangle(xy=(actual_overlap[ec],ec*bar_height),height=bar_height,width=chance_overlap[ec]-actual_overlap[ec],facecolor=blue,edgecolor='none',fill=True)
             
            # white bar drawn up to the minimum possible overlap (covering up gray where overlap value isn't possible)
            p2 = patches.Rectangle(xy=(0.0,ec*bar_height),height=bar_height,width=min_overlap[ec],facecolor=white,edgecolor='none',fill=True)
                                 
            ax.add_patch(p0)
            ax.add_patch(p1)
            ax.add_patch(p2)
        
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])  
        ax.set_xticks([])
        if i==(num_stim-2):
            ax.set_xticks([0.0,1.0])
        ax.set_yticks([])
        
        
        if i==0:
            ax.set_title(self.stim_names[j],fontsize=14)
        if j==(num_stim-1):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(self.stim_names[i],fontsize=14)  
        
    def __make_single_pairwise_scatterplot(self,ax,i,j,curr,axis_ub=0.7):                             
                   
        num_stim = len(self.stim_names)        
          
        sig_color = self.__zscore_to_color(self.zscores[curr])      
          
        gray = [0.5,0.5,0.5]
        ax.plot([0,axis_ub],[0,axis_ub],'k--')
        ax.scatter(self.prod_marg[curr],self.intersections[curr],s=22,c=sig_color,edgecolors=gray)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
        plt.axis('square')
        ax.set_xlim([0,axis_ub])
        ax.set_ylim([0,axis_ub])   
        if i==0:
            ax.set_title(self.stim_names[j],fontsize=14)
        if j==(num_stim-1):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(self.stim_names[i],fontsize=14)
     
    def __shuffle_marginals(self,frac_1,frac_2,true_intersect,num_cells):    
        # returns a z-score for the true intersection among a distribution of 
        # intersections resulting from random shuffling responsive cells with the
        # observed marginal distributions
        
        num_1 = int(frac_1*num_cells)
        num_2 = int(frac_2*num_cells)    
        
        num_stim_selective = np.zeros((self.num_shuffles,num_cells))
        for s in range(self.num_shuffles):
            selective_1 = np.random.permutation(num_cells)[:num_1]
            selective_2 = np.random.permutation(num_cells)[:num_2]
            
            num_stim_selective[s,selective_1] += 1
            num_stim_selective[s,selective_2] += 1  
            
        selective_for_both = num_stim_selective > 1
        shuffled_intersections = np.mean(selective_for_both,axis=1)
        
        percentile = np.mean(true_intersect > shuffled_intersections)
        
        # convert the percentile to a logarithmic value with the median of the 
        # distribution centered at zero.
        if percentile==1.0:
            z_score = -np.log10(1.0/self.num_shuffles) + np.log10(0.5)
        elif percentile==0.0:
            z_score = -1.0*(-np.log10(1.0/self.num_shuffles) + np.log10(0.5))
        elif percentile > 0.5:
            z_score = -np.log10((1.0-percentile)) + np.log10(0.5)
        else:
            z_score = -1.0*(-np.log10(percentile) + np.log10(0.5))
                  
        return z_score
     
    def __selectivity_by_container(self,stim_1,stim_2):
            
        has_stim_1 = ~self.cells[stim_1].isnull()
        has_stim_2 = ~self.cells[stim_2].isnull()
        
        ec_ids = np.sort(np.unique(self.cells['ecid'].values))
    
        frac_stim_1 = []
        frac_stim_2 = []
        frac_stim_12 = []
        cells_per_ec = []
        for ec in ec_ids:
            
            is_ec = (self.cells['ecid'] == ec).values    
        
            cell_in_ec = is_ec & has_stim_1 & has_stim_2
            num_cells_in_ec = np.sum(cell_in_ec).astype(float)
            
            stim_1_selective = cell_in_ec & (self.cells[stim_1].values > 0)
            stim_2_selective = cell_in_ec & (self.cells[stim_2].values > 0)
            
            #don't consider experiment containers with fewer than min_cells
            if num_cells_in_ec > self.min_cells:
                frac_stim_1.append(np.sum(stim_1_selective)/num_cells_in_ec)
                frac_stim_2.append(np.sum(stim_2_selective)/num_cells_in_ec)
                frac_stim_12.append(np.sum(stim_1_selective & stim_2_selective)/num_cells_in_ec)
                cells_per_ec.append(num_cells_in_ec)
    
        frac_stim_1 = np.array(frac_stim_1)
        frac_stim_2 = np.array(frac_stim_2)
        frac_stim_12 = np.array(frac_stim_12)
        prod_marginals = frac_stim_1 * frac_stim_2
        cells_per_ec = np.array(cells_per_ec).astype(int)
        
        return frac_stim_1, frac_stim_2, frac_stim_12, prod_marginals, cells_per_ec   