#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:27:04 2018

@author: saskiad
"""

pop_results['experiment_container_id'] = np.NaN
pop_results['area'] = np.NaN
pop_results['tld1_name'] = np.NaN
pop_results['depth_range'] = np.NaN
pop_results['cre_depth'] = np.NaN

for index,row in pop_results.iterrows():
    session_id = row.session_id
    try: 
        pop_results.experiment_container_id.loc[index] = datacube[datacube.session_A==session_id].experiment_id.values[0]
        pop_results.area.loc[index] = datacube[datacube.session_A==session_id].area.values[0]
        pop_results.tld1_name.loc[index] = datacube[datacube.session_A==session_id].cre.values[0]
        depth = datacube[datacube.session_A==session_id].depth.values[0]
        pop_results.depth_range.loc[index] = 100*((np.floor(depth/100)).astype(int))
    except:
        try:
            pop_results.experiment_container_id.loc[index] = datacube[datacube.session_B==session_id].experiment_id.values[0]
            pop_results.area.loc[index] = datacube[datacube.session_B==session_id].area.values[0]
            pop_results.tld1_name.loc[index] = datacube[datacube.session_B==session_id].cre.values[0]
            depth = datacube[datacube.session_B==session_id].depth.values[0]
            pop_results.depth_range.loc[index] = 100*((np.floor(depth/100)).astype(int))
        except:
            try:
                pop_results.experiment_container_id.loc[index] = datacube[datacube.session_C==session_id].experiment_id.values[0]
                pop_results.area.loc[index] = datacube[datacube.session_C==session_id].area.values[0]
                pop_results.tld1_name.loc[index] = datacube[datacube.session_C==session_id].cre.values[0]
                depth = datacube[datacube.session_C==session_id].depth.values[0]
                pop_results.depth_range.loc[index] = 100*((np.floor(depth/100)).astype(int))
            except:
                print session_id

pop_results.ix[pop_results.tld1_name=='Scnn1a-Tg3-Cre', 'depth_range'] = 200
pop_results.ix[pop_results.tld1_name=='Nr5a1-Cre', 'depth_range'] = 200
pop_results.ix[pop_results.tld1_name=='Fezf2-CreERT', 'depth_range'] = 300

pop_results['cre_depth'] = pop_results[['tld1_name','depth_range']].apply(tuple, axis=1)

import os
figure_directory = r'/Users/saskiad/Documents/CAM/paper figures/Gabe stuff'

def plot_gabe_results(results, plot_key, x_label, area, figure_name, xlim, nticks):
    sp.plot_strip_plot(results, area=area,plot_key=plot_key, x_label=x_label, 
               fig_base_dir=figure_directory, 
               figname=figure_name, bar=True, xlim=xlim, Nticks=nticks)
    print figname


def make_pp_results(results, plot_key, stimulus):
    data_input = results[np.isfinite(results[plot_key])]
    print len(data_input)
    results = np.empty((6,4,2))
    for i,a in enumerate(areas_pp):
        for j,d in enumerate(cre_depth):
            results[i,j,1] = len(data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])])
            results[i,j,0] = data_input[(data_input.area==a)&(data_input.tld1_name==d[0])&(data_input.depth_range==d[1])][plot_key].median()
    return results

plot_gabe_results(pop_results_new[pop_results_new.stim=='natural_scenes'], 'noise_corr','','VISp',figure_name='noise_corr',xlim=(0,0.2),nticks=2)
plot_gabe_results(pop_results_new[pop_results_new.stim=='natural_scenes'], 'rho_signal_noise_corrs','','VISp',figure_name='rho_signal_noise_corrs',xlim=(-1,1),nticks=5)

test = make_pp_results(pop_results_new[pop_results_new.stim=='natural_scenes'], 'noise_corr','ns')
pp.make_pawplot_population(test, 'noise_corr_ns',clim=(0.007, 0.016), fig_base_dir=figure_directory)

test = make_pp_results(pop_results_new[pop_results_new.stim=='natural_scenes'], 'rho_signal_noise_corrs','ns')
pp.make_pawplot_population(test, 'rho_corrs_ns',clim=(0.1,0.32), cticks=[0.1, 0.2,0.3], fig_base_dir=figure_directory)

test = make_pp_results(pop_results_new[pop_results_new.stim=='natural_scenes'], 'ns_frame_KNeighbors','ns')
pp.make_pawplot_population(test, 'rho_corrs_ns',clim=(0.1,0.32), cticks=[0.1, 0.2,0.3], fig_base_dir=figure_directory)


plot_gabe_results(pop_results_new,'dg_orientation_KNeighbors','','VISp',figure_name='dg_KNN_cre',xlim=(0,12), nticks=6)
test = make_pp_results(pop_results_new, 'dg_orientation_KNeighbors','dg')
pp.make_pawplot_population(test, 'dg_KNN',clim=(2,7.05), fig_base_dir=figure_directory)

test = make_pp_results(pop_results_new, 'dg_orientation_KNeighbors')
pp.make_pawplot_population(test, 'dg_KNN', fig_base_dir=figure_directory)

ns_frame_KNeighbors_confusion_sparsity
metrics_ns = pd.merge(metrics_ns, prob_repsonse, on='cell_specimen_id')

pp.make_pawplot_population(decoding_sparsity, 'decoding_sparsity',fig_base_dir=figure_directory,clim=(0.5, 0.9),cticks=[0.5, 0.7, 0.9])

results = np.empty((6,4,3))
for i,a in enumerate(areas_pp):
    for j,d in enumerate(cre_depth):
        results[i,j,2] = len(metrics_ns[(metrics_ns.area==a)&(metrics_ns.tld1_name==d[0])&(metrics_ns.depth_range==d[1])])
        results[i,j,1] = len(metrics_ns[(metrics_ns.area==a)&(metrics_ns.tld1_name==d[0])&(metrics_ns.depth_range==d[1])])
        results[i,j,0] = metrics_ns[(metrics_ns.area==a)&(metrics_ns.tld1_name==d[0])&(metrics_ns.depth_range==d[1])]['probability_response_ns'].median()

with sns.axes_style('whitegrid'):
    fig = plt.figure(figsize=(8,4))
    plt.plot(100*vgg_ns.responsivity[1:], 'o-')
    plt.axhline(y=21, color='k', ls='--')
    plt.axhline(y=5, color='k', ls='--')
    plt.ylim(0,100)
    plt.xticks([])
    plt.tick_params(labelsize=16)
    plt.grid(axis='x')
    plt.ylabel("Percent effective stimuli", fontsize=20)
    pp.save_figure(fig, r'/Users/saskiad/Documents/CAM/paper figures/percent effective stimuli_fig6')