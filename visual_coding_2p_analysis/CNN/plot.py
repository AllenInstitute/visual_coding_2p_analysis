import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py, os

from scipy.stats import t, norm

from visual_coding_2p_analysis.pawplot import make_pawplot_population, mega_paw_plot, save_figure

JUST_POOLING = True
NO_LEGEND = True

BONFERONI = True
ALPHA = 0.01/2

NUM_CLUSTERS = 5

def only_pool_and_input(model_layers):

    filter_func = lambda x:  'input' in x.split('_') or 'pool' in x.split('_')
    return filter(filter_func, model_layers)

def map_r_to_probs_by_cluster(r_ssm, sig, clusters):

    # print "r_ssm.shape:  ", r_ssm.shape
    # print "sig.shape:  ", sig.shape
    # print "clusters:  ", clusters
    cluster_probs = np.zeros(NUM_CLUSTERS)

    labels, idx = np.unique(clusters, return_index=True)

    cluster_order = clusters[sorted(idx)]
    for i, c in enumerate(cluster_order):
        in_cluster = clusters==c
        r_vals = r_ssm[np.logical_and(in_cluster, sig)]
        if len(r_vals)>0:
            total = np.mean(r_vals)
        else:
            total = 0

        cluster_probs[i] = total

    # cluster_probs /= np.sum(cluster_probs)

    return cluster_probs


def selectivity(probs):

    if np.all(probs==0):
        return 0

    m = np.mean(probs)
    s = np.mean(probs**2)
    n = probs.shape[0]
    d = 1.0 - 1.0/n

    sel = (1.0 - m**2/s)/d

    return sel


def make_pawplot_results(results, name, clim=None, fig_base_dir='plots'):

    '''results is an array of shape (6,4,3), indexed by area, layer
    each 3 tuple is main value (color), interior radius, exterior radius)'''
    # areas_pp = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISal', 'VISl']
    # depths = [100,200,300,500]
    # responsive = 'responsive_'+stimulus_suffix
    # resp = data_input[data_input[responsive]==True]
    # results = np.empty((6,4,3))
    # for i,a in enumerate(areas_pp):
    #     for j,d in enumerate(depths):
    #         results[i,j,2] = len(data_input[(data_input.area==a)&(data_input.depth_range==d)])
    #         results[i,j,1] = len(resp[(resp.area==a)&(resp.depth_range==d)])
    #         results[i,j,0] = resp[(resp.area==a)&(resp.depth_range==d)][metric].median()
    if clim is None:
        cmin = np.round(np.nanmin(results[:,:,0]), 1)
        cmax = np.round(np.nanmax(results[:,:,0]), 1)
        clim = (cmin, cmax)

    fig = mega_paw_plot(sdata_list=[results[:,0,1]*.000035, results[:,1,1]*.000035, results[:,2,1]*.000035, results[:,3,1]*.000035],
                 sdata_bg_list=[results[:,0,2]*.000035, results[:,1,2]*.000035, results[:,2,2]*.000035, results[:,3,2]*.000035],
                 cdata_list=[results[:,0,0], results[:,1,0], results[:,2,0], results[:,3,0]], clim=clim, edgecolor='#cccccc', figsize=(5,15))

    figname = os.path.join(fig_base_dir, name+'_pawplot')
    save_figure(fig, figname)


def wavelet_values(ax):
    
    model_name = 'vgg16'
    new_size = 50

    wavelet_data_file = 'Results/'+model_name+'_wavelet_ssm_'+str(new_size)+'.h5'

    results = h5py.File(wavelet_data_file, 'r')

    model_layers = results.keys()
    model_layers = [l for l in model_layers if l.split('_')[0].isdigit()]
    model_layers.sort(key=lambda x: int(x.split('_')[0]))

    # r_vals = []
    # for l in layers:
    #     r_vals.append(data[l+'/ssm/r'].value)
    BONF_CRE = len(model_layers)
    POOL_INDICES = np.array([0,3,6,10,14,18])

    ssm = np.array([results[l]['ssm/r'].value for l in model_layers])
    shuffles = [results[l]['ssm/rshuffles'] for l in model_layers]
    t_shuffles = [t.fit(s) for s in shuffles]
    # norm_shuffles = [norm.fit(s) for s in shuffles]

    p_vals = np.array([1.0 - t.cdf(s, *tparams) for s,tparams in zip(ssm, t_shuffles)])
    # norm_p_vals = [1.0 - norm.cdf(s, *tparams) for s,tparams in zip(ssm, norm_shuffles)]

    shuffle_mean_ssm = [np.mean(results[l]['ssm/rshuffles'].value) for l in model_layers]
    shuffle_std_ssm = [np.std(results[l]['ssm/rshuffles'].value) for l in model_layers]

    shuffle_min_ssm = np.array(shuffle_mean_ssm) - np.array(shuffle_std_ssm)
    shuffle_max_ssm = np.array(shuffle_mean_ssm) + np.array(shuffle_std_ssm)


    ssm = ssm[POOL_INDICES]
    p_vals = p_vals[POOL_INDICES]
    shuffle_min_ssm = shuffle_min_ssm[POOL_INDICES]
    shuffle_max_ssm = shuffle_max_ssm[POOL_INDICES]


    sig = np.array([p < ALPHA/BONF_CRE for p in p_vals])
    sig_index = np.where(sig)[0]

    ax.plot(sig_index, ssm[sig_index], 'o',color='k', markersize=5, alpha=0.5)

    ax.plot(ssm, '--', color='k', linewidth=3, alpha=0.5)

def main(model_name, n):

    

    results = h5py.File('Results/'+model_name+'_ssm_cca_'+str(n)+'.h5','r')
    clusters = h5py.File('Results/'+model_name+'_clusters_'+str(n)+'.h5', 'r')
    clusters = clusters['clusters'].value

    area_list = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']
    alt_area_list = ['V1', 'LM', 'AL', 'PM', 'AM', 'RL']
    layer_list = ['layer23', 'layer4', 'layer5', 'layer6']

    area_index = {'VISp': 0,
                  'VISl': 1,
                  'VISal':2,
                  'VISpm':3,
                  'VISam':4,
                  'VISrl':5}

    area_pawplot_index = {'VISp': 0,
                  'VISl': 5,
                  'VISal':4,
                  'VISpm':1,
                  'VISam':2,
                  'VISrl':3}

    layer_index = {'layer23': 0,
                   'layer4':  1,
                   'layer5':  2,
                   'layer6':  3} 

    # fig_cca, ax_cca = plt.subplots(4,6, figsize=(24,16))
    fig_ssm, ax_ssm = plt.subplots(4,6, figsize=(24,16), sharex=True, sharey=True)
    fig_p, ax_p = plt.subplots(4,6, figsize=(24,16))
    fig_cluster, ax_cluster = plt.subplots(4,6, figsize=(24,16))

    # fig_cca_ei, ax_cca_ei = plt.subplots(4,6, figsize=(24,16))
    fig_ssm_ei, ax_ssm_ei = plt.subplots(4,6, figsize=(24,16))
    fig_p_ei, ax_p_ei = plt.subplots(4,6, figsize=(24,16))
    fig_cluster_ei, ax_cluster_ei = plt.subplots(4,6, figsize=(24,16))
    
    color_dict = {'excitatory': 'red',
                  'inhibitory': 'blue',
                    'Emx1' : '#9f9f9f',
                    'Slc17a7' : '#5c5c5c',
                    'Cux2': '#a92e66',
                    'Rorb' : '#7841be',
                    'Scnn1a' : '#4f63c2',
                    'Nr5a1' : '#5bb0b0',
                    'Fezf2' : '#3A6604',
                    'Tlx3' : '#99B20D',
                    'Rbp4' : '#5cad53',
                    'Ntsr1' : '#ff3b39',
                    'Sst' : '#7B5217',
                    'Vip' : '#b49139'}

    BONF_EI = 0
    BONF_CRE = 0
    for area in area_list:
        for cre in results[area]:
            for layer in results[area][cre]:
                if cre=='excitatory' or cre=='inhibitory':
                    BONF_EI += 1
                else:
                    BONF_CRE += 1

    # pawplot_array_dict = {}
    # pawplot_mean_array_dict = {}

    pawplot_results_dict = {}

    # for area in results:
    for area in area_list:
        cre_lines = results[area].keys()
        if 'inhibitory' in cre_lines:
            cre_lines.remove('inhibitory')
        cre_lines.remove('excitatory')
        for cre in results[area]:
            # pawplot_array_dict[cre] = pawplot_array_dict.get(cre, np.zeros([6,4,2]))
            # pawplot_mean_array_dict[cre] = pawplot_mean_array_dict.get(cre, np.zeros([6,4,2]))

            pawplot_results_dict[cre] = pawplot_results_dict.get(cre, np.zeros([6,4,3]))

            for layer in results[area][cre]:
                model_layers = results[area][cre][layer].keys()
                model_layers = [m for m in model_layers if m[:3]!='sim']
                model_layers.sort(key= lambda x: int(x.split('_')[0]))

                if JUST_POOLING:
                    model_layers = only_pool_and_input(model_layers)

                model_layer_names = ['_'.join(l.split('_')[1:]) for l in model_layers]

                # model_layer_names = only_pool_and_input(model_layers)
                # import sys
                # sys.exit()

                ssm = np.array([results[area][cre][layer][l]['ssm/r'].value for l in model_layers])
                shuffles = [results[area][cre][layer][l]['ssm/rshuffles'] for l in model_layers]
                t_shuffles = [t.fit(s) for s in shuffles]
                norm_shuffles = [norm.fit(s) for s in shuffles]

                p_vals = [1.0 - t.cdf(s, *tparams) for s,tparams in zip(ssm, t_shuffles)]
                norm_p_vals = [1.0 - norm.cdf(s, *tparams) for s,tparams in zip(ssm, norm_shuffles)]

                shuffle_mean_ssm = [np.mean(results[area][cre][layer][l]['ssm/rshuffles'].value) for l in model_layers]
                shuffle_std_ssm = [np.std(results[area][cre][layer][l]['ssm/rshuffles'].value) for l in model_layers]

                shuffle_min_ssm = np.array(shuffle_mean_ssm) - np.array(shuffle_std_ssm)
                shuffle_max_ssm = np.array(shuffle_mean_ssm) + np.array(shuffle_std_ssm)

                

                # fold_min_ssm = [np.min(results[area][cre][layer][l]['ssm/rfolds'].value) for l in model_layers]
                # fold_max_ssm = [np.max(results[area][cre][layer][l]['ssm/rfolds'].value) for l in model_layers]

                if cre=='excitatory' or cre=='inhibitory':
                    # cca = [np.mean(results[area][cre][layer][l]['rho']) for l in model_layers]
                    sig = np.array([p < ALPHA/BONF_EI for p in p_vals])
                    sig_index = np.where(sig)[0]

                    ax_ssm_ei[layer_index[layer], area_index[area]].plot(sig_index, ssm[sig_index], 'o', color=color_dict[cre])
                    # ax_cca_ei[layer_index[layer], area_index[area]].plot(cca, color=color_dict[cre], label=cre)
                    ax_ssm_ei[layer_index[layer], area_index[area]].plot(ssm, color=color_dict[cre], label=cre)
                    ax_ssm_ei[layer_index[layer], area_index[area]].fill_between(np.arange(len(ssm)),
                                                                            shuffle_min_ssm, 
                                                                            shuffle_max_ssm, 
                                                                            facecolor=color_dict[cre],
                                                                            alpha=0.5)

                    # ax_ssm_ei[layer_index[layer], area_index[area]].fill_between(np.arange(len(ssm)),
                    #                                                         fold_min_ssm, 
                    #                                                         fold_max_ssm, 
                    #                                                         facecolor=color_dict[cre],
                    #                                                         alpha=0.5)

                    if not NO_LEGEND:
                        ax_ssm_ei[layer_index[layer], area_index[area]].legend()

                    ax_p_ei[layer_index[layer], area_index[area]].plot(p_vals, color=color_dict[cre], label=cre)
                    # ax_p_ei[layer_index[layer], area_index[area]].plot(norm_p_vals, color=color_dict[cre], label=cre)
                    ax_p_ei[layer_index[layer], area_index[area]].legend()

                    
                    if not JUST_POOLING:
                        cluster_probs = map_r_to_probs_by_cluster(ssm, sig, clusters)
                        cluster_norm = np.sum(cluster_probs)
                        if cluster_norm > 0:
                            cluster_mean = np.sum(np.arange(5)*cluster_probs)/cluster_norm
                        else:
                            cluster_mean = 0.0
                        

                        siz = np.max(cluster_probs)*3000
                        sel = selectivity(cluster_probs)
                        cluster_results = np.array([cluster_mean, sel*siz , siz])
                        print cluster_results
                        pawplot_results_dict[cre][area_pawplot_index[area], layer_index[layer]] = cluster_results

                    

                    # if np.all(np.isfinite(cluster_probs)):
                    #     cluster_assignment = np.argmax(cluster_probs)
                    #     pawplot_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([cluster_assignment, 10])

                    #     norm = np.sum(cluster_probs)
                    #     if norm > 0:
                    #         cluster_mean = np.sum(np.arange(5)*cluster_probs)/norm
                    #     else:
                    #         cluster_mean = 0.0

                    #     pawplot_mean_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([cluster_mean, 10])
                    # else:
                    #     pawplot_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([-1, 10])
                    #     pawplot_mean_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([-1, 10])
                    # print area, layer, cre
                    # print cluster_probs
                    # print
                    # print cluster_probs
                        ax_cluster_ei[layer_index[layer], area_index[area]].plot(cluster_probs, color=color_dict[cre], label=cre, linewidth=5.0)
                        ax_cluster_ei[layer_index[layer], area_index[area]].set_ylim(0,0.6)
                    # ax_cluster_ei[layer_index[layer], area_index[area]].legend()
                else:
                    # cca = [np.mean(results[area][cre][layer][l]['rho']) for l in model_layers]
                    # ssm = [results[area][cre][layer][l]['ssm/r'].value for l in model_layers]
                    # shuffle_mean_ssm = [np.mean(results[area][cre][layer][l]['ssm/rshuffles'].value) for l in model_layers]
                    # shuffle_std_ssm = [np.std(results[area][cre][layer][l]['ssm/rshuffles'].value) for l in model_layers]

                    # shuffle_min_ssm = np.array(shuffle_mean_ssm) - np.array(shuffle_std_ssm)
                    # shuffle_max_ssm = np.array(shuffle_mean_ssm) + np.array(shuffle_std_ssm)

                    # ax_cca[layer_index[layer], area_index[area]].plot(cca, label=cre)


                    
                    sig = np.array([p < ALPHA/BONF_CRE for p in p_vals])
                    sig_index = np.where(sig)[0]

                    
                    

                    ax_ssm[layer_index[layer], area_index[area]].plot(sig_index, ssm[sig_index], 'o',color=color_dict[cre], markersize=10)

                    ax_ssm[layer_index[layer], area_index[area]].plot(ssm, color=color_dict[cre],label=cre, linewidth=5.0)
                    ax_ssm[layer_index[layer], area_index[area]].fill_between(np.arange(len(ssm)),
                                                        shuffle_min_ssm, 
                                                        shuffle_max_ssm,
                                                        color=color_dict[cre], alpha=0.3)

                    if not NO_LEGEND:
                        ax_ssm[layer_index[layer], area_index[area]].legend()

                    ax_p[layer_index[layer], area_index[area]].plot(p_vals, color=color_dict[cre],label=cre)
                    # ax_p[layer_index[layer], area_index[area]].plot(norm_p_vals, color=color_dict[cre],label=cre)
                    ax_p[layer_index[layer], area_index[area]].legend()

                    if not JUST_POOLING:
                        cluster_probs = map_r_to_probs_by_cluster(ssm, sig, clusters) 
                        cluster_norm = np.sum(cluster_probs)
                        if cluster_norm > 0:
                            cluster_mean = np.sum(np.arange(5)*cluster_probs)/cluster_norm
                        else:
                            cluster_mean = 0.0
                        

                        siz = np.max(cluster_probs)*3000
                        # siz = np.max(cluster_probs)
                        sel = selectivity(cluster_probs)
                        cluster_results = np.array([cluster_mean, sel*siz , siz])
                        try:
                            pawplot_results_dict[cre][area_pawplot_index[area], layer_index[layer]] = cluster_results
                        except IndexError as ie:
                            print pawplot_results_dict[cre].shape # [area_pawplot_index[area], layer_index[layer]].shape
                            print results.shape
                            print
                            raise ie


                    # if np.all(np.isfinite(cluster_probs)):
                    #     cluster_assignment = np.argmax(cluster_probs)
                    #     pawplot_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([cluster_assignment, 10])

                    #     cluster_mean = np.sum(np.arange(5)*cluster_probs)
                    #     pawplot_mean_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([cluster_mean, 10])
                    # else:
                    #     pawplot_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([-1, 10])
                    #     pawplot_mean_array_dict[cre][area_pawplot_index[area], layer_index[layer]] = np.array([-1, 10])
                    # # print area, layer, cre
                    # # print cluster_probs
                    # # print
                        ax_cluster[layer_index[layer], area_index[area]].plot(cluster_probs, color=color_dict[cre], label=cre, linewidth=5.0)
                        ax_cluster[layer_index[layer], area_index[area]].set_ylim(0,0.6)
                    # ax_cluster[layer_index[layer], area_index[area]].legend()
                    

    for axlist in ax_ssm:
        for ax in axlist:
            wavelet_values(ax)     
                    
    ax_ssm[0,0].set_ylim([0.0, 0.6])

    fig_ssm.delaxes(ax_ssm[3, 2])
    fig_ssm.delaxes(ax_ssm[3, 4])
    fig_ssm.delaxes(ax_ssm[3, 5])

    for i in range(6):
        # ax_cca[0,i].set_title(area_list[i])
        ax_ssm[0,i].set_title(alt_area_list[i],fontsize=40)
        ax_ssm[-1,i].tick_params(labelsize=20)
        ax_ssm[-1,i].xaxis.set_ticks(np.arange(6))
#        ax_ssm[-1,i].set_xtickm
#        ax_ssm[0,i].
        # ax_cca_ei[0,i].set_title(area_list[i])
        ax_ssm_ei[0,i].set_title(area_list[i])
        ax_p[0,i].set_title(area_list[i])
        ax_p_ei[0,i].set_title(area_list[i])
        if not JUST_POOLING:
            ax_cluster[0,i].set_title(area_list[i])
            ax_cluster_ei[0,i].set_title(area_list[i])
            ax_cluster[0,i].title.set_fontsize(40)
            ax_cluster_ei[0,i].title.set_fontsize(40)

    for i in range(4):
        # ax_cca[i,0].set_ylabel(layer_list[i])
        ax_ssm[i,0].set_ylabel(layer_list[i])
        ax_ssm[i,0].yaxis.label.set_fontsize(40)
        ax_ssm[i,0].tick_params(labelsize=20)

        # ax_cca_ei[i,0].set_ylabel(layer_list[i])
        ax_ssm_ei[i,0].set_ylabel(layer_list[i])
        ax_p[i,0].set_ylabel(layer_list[i])
        ax_p_ei[i,0].set_ylabel(layer_list[i])

        if not JUST_POOLING:
            ax_cluster[i,0].set_ylabel(layer_list[i])
            ax_cluster_ei[i,0].set_ylabel(layer_list[i])

            ax_cluster[i,0].yaxis.label.set_fontsize(40)
            ax_cluster_ei[i,0].yaxis.label.set_fontsize(40)

    # fig_cca.suptitle(model_name)
    # fig_ssm.suptitle(model_name)

    # fig_cca_ei.suptitle(model_name)
    fig_ssm_ei.suptitle(model_name)

    fig_p.suptitle(model_name)
    fig_p_ei.suptitle(model_name)

    if not JUST_POOLING:
        fig_cluster.suptitle(model_name)
        fig_cluster_ei.suptitle(model_name)

    # fig_cca.text(0.1, 0.05, '->'.join(model_layer_names))
    # fig_ssm.text(0.1, 0.05, '->'.join(model_layer_names))
    fig_p.text(0.1, 0.05, '->'.join(model_layer_names))

    # fig_cca_ei.text(0.1, 0.05, '->'.join(model_layer_names))
    fig_ssm_ei.text(0.1, 0.05, '->'.join(model_layer_names))
    fig_p_ei.text(0.1, 0.05, '->'.join(model_layer_names))


    # fig_cca.savefig(model_name+'_cca_'+str(n)+'.pdf')
    fig_ssm.savefig('plots/'+model_name+'_ssm_'+str(n)+'.pdf')
    fig_p.savefig('plots/'+model_name+'_ssm-pval_'+str(n)+'.pdf')

    

    # fig_cca_ei.savefig(model_name+'_cca_ei_'+str(n)+'.pdf')
    fig_ssm_ei.savefig('plots/'+model_name+'_ssm_ei_'+str(n)+'.pdf')
    fig_p_ei.savefig('plots/'+model_name+'_ssm_ei-pval_'+str(n)+'.pdf')

    if not JUST_POOLING:
        fig_cluster.savefig('plots/'+model_name+'_assignment_'+str(n)+'.pdf')
        fig_cluster_ei.savefig('plots/'+model_name+'_assignment_ei_'+str(n)+'.pdf')

        for cre in pawplot_results_dict:
            name =  model_name+'_'+cre+'_'+str(n)  # +'.pdf'
            make_pawplot_results(pawplot_results_dict[cre], name=name)
        # make_pawplot_population(pawplot_array_dict[cre], name, fig_base_dir='plots')

        # name =  model_name+'_'+cre+'_'+str(n)+'_mean'  # .pdf'
        # make_pawplot_population(pawplot_array_dict[cre], name, fig_base_dir='plots')

if __name__ == '__main__':


    main('vgg16', 50)
    # main('vgg16', 100)
    # main('vgg16', 200)
    # main('vgg16', 400)

    # main('vgg19', 50)
    # main('vgg19', 100)
    # main('vgg19', 200)
    # main('vgg19', 400)
    # main('vgg19')

    # main('lemouse', 50)
    # main('lemouse', 100)
    # main('lemouse', 200)
    # main('lemouse', 400)

    # main('inceptionv3', 200)
    # main('inceptionv3', 400)

    # main('alexnet', 227)

    # main('resnet50', 200)
    # main('resnet50', 400)