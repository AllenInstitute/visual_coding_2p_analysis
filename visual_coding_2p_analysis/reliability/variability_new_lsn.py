import h5py
from unnatural_movie_analysis_lsn import unnatural_movie_analysis_lsn
from natural_movie_analysis import natural_movie_analysis
import os
import numpy as np
import sys
from gabor_wavelet_preprocessing_linquad import gabor_wavelet_preprocessing_linquad
from threshgrad import threshold_grad_desc_cv, lsq, dotdelay, soft_rect, soft_rect_poisson, weighted_correlation
import itertools
from scipy.stats import mode
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from scipy.ndimage.filters import gaussian_filter1d
# from scipy.signal import boxcar

# exp = np.int(sys.argv[1])
# std = np.float32(sys.argv[1])


boc = BrainObservatoryCache(manifest_file='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json')

# ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp', 'VISl', 'VISal'])
ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam'])
# ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp'])
ec_ids = [ec['id'] for ec in ecs]
CC_max_all = {'locally_sparse_noise':{}, 'locally_sparse_noise_4deg':{}, 'locally_sparse_noise_8deg':{}}
SP_all = {'locally_sparse_noise':{}, 'locally_sparse_noise_4deg':{}, 'locally_sparse_noise_8deg':{}}

stims = ['locally_sparse_noise', 'locally_sparse_noise_4deg', 'locally_sparse_noise_8deg']

for exp in ec_ids:
    print(exp)

    for kk in CC_max_all.keys():
        CC_max_all[kk][exp] = []
        SP_all[kk][exp] = []

    for ss in stims:
        try:

            unma = unnatural_movie_analysis_lsn(exp, .05,  movie_list=[ss])

            delays = np.arange(10)

            (stim, resp, frame_numbers, weights, dffs,running, idx, loc) = unma.vectorize_data_simple(delays)


            stim = stim.reshape(stim.shape[0], -1).T
            idx0 = []
            idx255 = []
            for p in stim:
                idx0.append(np.where(p == 0)[0])
                idx255.append(np.where(p == 255)[0])


            if std > 0:
                resp = gaussian_filter1d(resp, std)
            resp = np.float32(resp.T * 10)
            
            start=0
            end=8
            rse = range(start, end)
            lrse = len(rse)
            reps = []
            for idx00 in idx0:
                starts = []
                for i in idx00:
                    # fidx = np.where(frame_numbers == i)[0][0]
                    tmp = np.where(frame_numbers == i)[0]
                    if len(tmp) > 0:
                        starts.append(tmp[0])
                starts = np.array(starts)[:,None]
                fidx = np.concatenate([starts+ii for ii in range(start, end)], axis=1)
                reps.append(fidx.shape[0])

            for idx00 in idx255:
                starts = []
                for i in idx00:
                    # fidx = np.where(frame_numbers == i)[0][0]
                    tmp = np.where(frame_numbers == i)[0]
                    if len(tmp) > 0:
                        starts.append(tmp[0])
                starts = np.array(starts)[:,None]
                fidx = np.concatenate([starts+ii for ii in range(start, end)], axis=1)
                reps.append(fidx.shape[0])

            N=mode(np.array(reps)).mode[0]
            T=(len(idx0) + len(idx255))*lrse
            resp_array = np.zeros((np.array(reps).max(),T, resp.shape[1]))*np.NaN
            cnt = 0
            for idx00 in idx0:
                starts = []
                for i in idx00:
                    # fidx = np.where(frame_numbers == i)[0][0]
                    tmp = np.where(frame_numbers == i)[0]
                    if len(tmp) > 0:
                        starts.append(tmp[0])
                starts = np.array(starts)[:,None]
                fidx = np.concatenate([starts+ii for ii in range(start, end)], axis=1)
                resp_array[:len(fidx), (cnt)*lrse:((cnt+1)*lrse)] = resp[fidx]
                cnt += 1

            for idx00 in idx255:
                starts = []
                for i in idx00:
                    # fidx = np.where(frame_numbers == i)[0][0]
                    tmp = np.where(frame_numbers == i)[0]
                    if len(tmp) > 0:
                        starts.append(tmp[0])
                starts = np.array(starts)[:,None]
                fidx = np.concatenate([starts+ii for ii in range(start, end)], axis=1)
                resp_array[:len(fidx), (cnt)*lrse:((cnt+1)*lrse)] = resp[fidx]
                cnt += 1


            resp_array = resp_array.transpose(2,0,1)
            for R in resp_array:
                y = np.nanmean(R, axis=0)

                # Precalculate basic values for efficiency
                Ey = np.nanmean(y)                              # E[y]
                Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


                # Calculate signal power (see [1])
                SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
                SP_all[ss][exp].append(SP)

                CC_max = np.sqrt(SP/Vy)
                if not np.isnan(CC_max):
                    CC_max_all[ss][exp].append(CC_max)
                else:
                    CC_max_all[ss][exp].append(0)

        except:
            continue


np.savez('/home/michaelo/variability_new_' + str(std) + '_CCmax.npz' , **CC_max_all)
np.savez('/home/michaelo/variability_new_' + str(std) + '_SP.npz' , **SP_all)

with open('/home/michaelo/variability_new_' + str(std) + '_CCmax.pkl', 'wb') as handle:
    pickle.dump(CC_max_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('/home/michaelo/variability_new_' + str(std) + '_SP_all.pkl', 'wb') as handle:
    pickle.dump(SP_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

figure(figsize=(32,9))
subplot(1,4,1)
hist(np.concatenate(CC_max_all['drifting_gratings'].values()))
title('Drifting Gratings')
xlabel('CC_max')
ylabel('# neurons')
subplot(1,4,2)
hist(np.concatenate(CC_max_all['natural_movie_one'].values()))
title('Natural Movie One')
xlabel('CC_max')
subplot(1,4,3)
hist(np.concatenate(CC_max_all['natural_movie_two'].values()))
title('Natural Movie Two')
xlabel('CC_max')
subplot(1,4,4)
hist(np.concatenate(CC_max_all['natural_movie_three'].values()))
title('Natural Movie Three')
xlabel('CC_max')
savefig('variability_moving.png')


figure(figsize=(16,9))
subplot(1,2,1)
hist(np.minimum(1, np.concatenate(CC_max_all['static_gratings'].values())))
title('Static Gratings')
xlabel('CC_max')
ylabel('# neurons')
subplot(1,2,2)
hist(np.minimum(1, np.concatenate(CC_max_all['natural_scenes'].values())))
title('Natural Scenes')
xlabel('CC_max')
savefig('variability_static.png')

dfc = pd.concat([df_nma, df_ns, df_dg, df_sg, df_lsn], ignore_index=True)

dfall = {}
for s in CC_max_all.keys():
    df = []
    for exp in CC_max_all[s].keys():

        area = dfc[dfc.experiment_container_id==exp].area.unique()[0]
        depth_range = dfc[dfc.experiment_container_id==exp].depth_range.unique()[0]
        tld1_name = dfc[dfc.experiment_container_id==exp].tld1_name.unique()[0]
        cre_depth = dfc[dfc.experiment_container_id==exp].cre_depth.unique()[0]

        d1 =   {'area':area,
                'depth_range':depth_range,
                'tld1_name':tld1_name,
                'cre_depth':str(cre_depth),
                'CC_max':CC_max_all[s][exp]
                }
        d1 = pd.DataFrame(d1)
        if len(df) == 0:
            df = d1
        else:
            df = pd.concat([df,d1], ignore_index=True)
    dfall[s] = df


stims = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'static_gratings', 'drifting_gratings', 'natural_scenes']
areas = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']
for a in areas:
    for s in stims:
        plot_metric_box_cre_depth(dfall[s], a, 'CC_max', s, 'r')

for s in stims:
    make_pawplot_metric_crespecific(dfall[s], 'CC_max', s, clim=(.1, .8))


for s in stims:
    dfall[s].to_csv(s+'.csv')
