import h5py
from unnatural_movie_analysis import unnatural_movie_analysis
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
std = np.float32(4)


boc = BrainObservatoryCache(manifest_file='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json')

# ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp', 'VISl', 'VISal'])
ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam'])
# ecs = boc.get_experiment_containers(simple=False, include_failed=False, targeted_structures=['VISp'])
ec_ids = [ec['id'] for ec in ecs]
CC_max_all = {'drifting_gratings':{}, 'natural_movie_one':{}, 'natural_movie_two':{}, 'natural_movie_three':{}, 'natural_scenes':{}, 'static_gratings':{}}
SP_all = {'drifting_gratings':{}, 'natural_movie_one':{}, 'natural_movie_two':{}, 'natural_movie_three':{}, 'natural_scenes':{}, 'static_gratings':{}}
mean_resp_all = {'drifting_gratings':{}, 'natural_movie_one':{}, 'natural_movie_two':{}, 'natural_movie_three':{}, 'natural_scenes':{}, 'static_gratings':{}}

for exp in ec_ids:
    print(exp)

    for kk in CC_max_all.keys():
        CC_max_all[kk][exp] = []
        SP_all[kk][exp] = []
        mean_resp_all[kk][exp] = []

    try:

        unma = unnatural_movie_analysis(exp, .05,  movie_list=['static_gratings'])

        delays = np.arange(10)

        (stim, resp, frame_numbers, weights, dffs,running, idx, loc) = unma.vectorize_data_simple(delays)
        f2 = np.where(diff(frame_numbers)==2)[0]
        for ff in f2:
            frame_numbers[ff] += 1

        if std > 0:
            resp = gaussian_filter1d(resp, std)
        resp = np.float32(resp.T * 10)
        
        start=-2
        end=10
        rse = range(start, end)
        lrse = len(rse)
        reps = []
        for i in range(1,frame_numbers.max()+1):
            fidx = np.where(frame_numbers == i)[0]
            ends = np.concatenate([np.where(np.diff(fidx)>1)[0], np.array([len(fidx)-1])])[:,None]
            fidx = np.concatenate([np.minimum(fidx[ends]-ii, resp.shape[0]-1) for ii in range(start, end)][::-1], axis=1)
            reps.append(fidx.shape[0])

        N=mode(np.array(reps)).mode[0]
        T=frame_numbers.max()*lrse
        resp_array = np.zeros((np.array(reps).max(),frame_numbers.max()*lrse, resp.shape[1]))*np.NaN
        for i in range(1,frame_numbers.max()+1):
            fidx = np.where(frame_numbers == i)[0]
            ends = np.concatenate([np.where(np.diff(fidx)>1)[0], np.array([len(fidx)-1])])[:,None]
            fidx = np.concatenate([np.minimum(fidx[ends]-ii, resp.shape[0]-1) for ii in range(start, end)][::-1], axis=1)
            resp_array[:len(fidx), (i-1)*lrse:(i*lrse)] = resp[fidx]

        resp_array = resp_array.transpose(2,0,1)
        for R in resp_array:
            y = np.nanmean(R, axis=0)

            # Precalculate basic values for efficiency
            Ey = np.nanmean(y)                              # E[y]
            Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


            # Calculate signal power (see [1])
            SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
            SP_all['static_gratings'][exp].append(SP)

            CC_max = np.sqrt(SP/Vy)
            mean_resp_all['static_gratings'][exp].append(np.nanmean(R))
            if not np.isnan(CC_max):
                CC_max_all['static_gratings'][exp].append(CC_max)
            else:
                CC_max_all['static_gratings'][exp].append(0)



        nma = natural_movie_analysis(exp, .05,  movie_list=['natural_scenes'])

        delays = np.arange(10)

        (stim, resp, frame_numbers, weights, dffs,running, idx, loc) = nma.vectorize_data_simple(delays)
        f2 = np.where(diff(frame_numbers)==2)[0]
        for ff in f2:
            frame_numbers[ff] += 1

        if std > 0:
            resp = gaussian_filter1d(resp, std)
        resp = np.float32(resp.T * 10)

        start=-2
        end=10
        rse = range(start, end)
        lrse = len(rse)
        reps = []
        for i in range(1,frame_numbers.max()+1):
            fidx = np.where(frame_numbers == i)[0]
            ends = np.concatenate([np.where(np.diff(fidx)>1)[0], np.array([len(fidx)-1])])[:,None]
            fidx = np.concatenate([np.minimum(fidx[ends]-ii, resp.shape[0]-1) for ii in range(start, end)][::-1], axis=1)
            reps.append(fidx.shape[0])

        N=mode(np.array(reps)).mode[0]
        T=frame_numbers.max()*lrse
        resp_array = np.zeros((np.array(reps).max(),frame_numbers.max()*lrse, resp.shape[1]))*np.NaN
        for i in range(1,frame_numbers.max()+1):
            fidx = np.where(frame_numbers == i)[0]
            ends = np.concatenate([np.where(np.diff(fidx)>1)[0], np.array([len(fidx)-1])])[:,None]
            fidx = np.concatenate([np.minimum(fidx[ends]-ii, resp.shape[0]-1) for ii in range(start, end)][::-1], axis=1)
            resp_array[:len(fidx), (i-1)*lrse:(i*lrse)] = resp[fidx]

        resp_array = resp_array.transpose(2,0,1)
        for R in resp_array:
            y = np.nanmean(R, axis=0)

            # Precalculate basic values for efficiency
            Ey = np.nanmean(y)                              # E[y]
            Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


            # Calculate signal power (see [1])
            SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
            SP_all['natural_scenes'][exp].append(SP)

            CC_max = np.sqrt(SP/Vy)
            mean_resp_all['natural_scenes'][exp].append(np.nanmean(R))
            if not np.isnan(CC_max):
                CC_max_all['natural_scenes'][exp].append(CC_max)
            else:
                CC_max_all['natural_scenes'][exp].append(0)


        unma = unnatural_movie_analysis(exp, .05,  movie_list=['drifting_gratings'])

        delays = np.arange(10)

        (stim, resp, frame_numbers, weights, dffs,running, idx, loc) = unma.vectorize_data_simple(delays)
        f2 = np.where(diff(frame_numbers)==2)[0]
        for ff in f2:
            frame_numbers[ff] += 1

        if std > 0:
            resp = gaussian_filter1d(resp, std)
        resp = np.float32(resp.T * 10)

        reps = []
        for i in range(1,frame_numbers.max()+1):
            fidx = np.where(frame_numbers == i)[0]
            reps.append(len(fidx))

        N=mode(np.array(reps)).mode[0]
        T=(61+lag*2)*40
        # T=frame_numbers.max()

        lag = 11/2 + 10 + 1
        # ufn = np.unique(frame_numbers)
        # respout = np.zeros(((61+lag*2)*40, resp.shape[1]), dtype='float32')
        resp_array = np.zeros((np.array(reps).max(),(61+lag*2)*40, resp.shape[1]))*np.NaN

        start = -lag
        end = 61 + lag
        rse = range(start, end)
        lrse = len(rse)
        
        # resp_array = np.zeros((np.array(reps).max(),frame_numbers.max()*lrse, resp.shape[1]))*np.NaN
        for i in np.arange(40):
            fn = 61*i + 1
            fidx = np.where(frame_numbers == fn)[0][:,None]
            # ends = np.concatenate([np.where(np.diff(fidx)>1)[0], np.array([len(fidx)-1])])[:,None]
            fidx = np.concatenate([np.minimum(fidx+ii, resp.shape[0]-1) for ii in range(start, end)], axis=1)
            resp_array[:len(fidx), (i)*lrse:((i+1)*lrse)] = resp[fidx]

        resp_array = resp_array.transpose(2,0,1)


        for R in resp_array:
            y = np.nanmean(R, axis=0)

            # Precalculate basic values for efficiency
            Ey = np.nanmean(y)                              # E[y]
            Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


            # Calculate signal power (see [1])
            SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
            SP_all['drifting_gratings'][exp].append(SP)

            CC_max = np.sqrt(SP/Vy)
            mean_resp_all['drifting_gratings'][exp].append(np.nanmean(R))
            if not np.isnan(CC_max):
                CC_max_all['drifting_gratings'][exp].append(CC_max)
            else:
                CC_max_all['drifting_gratings'][exp].append(0)



        for mov in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']:
            nma = natural_movie_analysis(exp, .05,  movie_list=[mov])

            delays = np.arange(10)

            (stim, resp, frame_numbers, weights, dffs,running, idx, loc) = nma.vectorize_data_simple(delays)
            f2 = np.where(diff(frame_numbers)==2)[0]
            for ff in f2:
                frame_numbers[ff] += 1

            if std > 0:
                resp = gaussian_filter1d(resp, std)
            resp = np.float32(resp.T * 10)

            reps = []

            for i in range(frame_numbers.max()+1):
                fidx = np.where(frame_numbers == i)[0]
                reps.append(len(fidx))

            N=mode(np.array(reps)).mode[0]
            T=frame_numbers.max()+1
            resp_array = np.zeros((np.array(reps).max(),frame_numbers.max()+1, resp.shape[1]))*np.NaN
            for i in range(frame_numbers.max()+1):
                fidx = np.where(frame_numbers == i)[0]
                # resp_array[:len(fidx), i] = resp[fidx]
                last = -10
                cnt = 0
                for ff in fidx:
                    if ff == last +1:
                        resp_array[cnt, i] = resp_array[cnt, i] + resp[ff]
                        last = ff
                    else:
                        resp_array[cnt, i] = resp[ff]
                        cnt += 1
                        last = ff

            resp_array = resp_array.transpose(2,0,1)

            for R in resp_array:
                y = np.nanmean(R, axis=0)

                # Precalculate basic values for efficiency
                Ey = np.nanmean(y)                              # E[y]
                Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


                # Calculate signal power (see [1])
                SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
                SP_all[mov][exp].append(SP)
                CC_max = np.sqrt(SP/Vy)
                mean_resp_all[mov][exp].append(np.nanmean(R))
                if not np.isnan(CC_max):
                    CC_max_all[mov][exp].append(CC_max)
                else:
                    CC_max_all[mov][exp].append(0)

    except:
        continue


np.savez('/home/michaelo/variability_new_' + str(std) + '_CCmax.npz' , **CC_max_all)
np.savez('/home/michaelo/variability_new_' + str(std) + '_SP.npz' , **SP_all)

with open('/home/michaelo/variability_new_' + str(std) + '_CCmax.pkl', 'wb') as handle:
    pickle.dump(CC_max_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('/home/michaelo/variability_new_' + str(std) + '_SP_all.pkl', 'wb') as handle:
    pickle.dump(SP_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/michaelo/variability_new_' + str(std) + '_mean_resp_all.pkl', 'wb') as handle:
    pickle.dump(mean_resp_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

        d1 =   {'experiment_container_id':exp,
                'area':area,
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

sns.set(style="whitegrid", font_scale = 2.5)
figure(figsize=(36,12))
subplot(1,4,1)
tmp = np.concatenate(CC_max_all['drifting_gratings'].values())**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Drifting Gratings \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
ylabel('# neurons')
subplot(1,4,2)
tmp = np.concatenate(CC_max_all['natural_movie_one'].values())**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Natural Movie One \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
subplot(1,4,3)
tmp =  np.concatenate(CC_max_all['natural_movie_two'].values())**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Natural Movie Two \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
subplot(1,4,4)
tmp = np.concatenate(CC_max_all['natural_movie_three'].values())**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Natural Movie Three \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
savefig('exp_var_moving.png')


figure(figsize=(18,12))
subplot(1,2,1)
tmp = np.minimum(1, np.concatenate(CC_max_all['static_gratings'].values()))**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Static Gratings \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
ylabel('# neurons')
subplot(1,2,2)
tmp =  np.minimum(1, np.concatenate(CC_max_all['natural_scenes'].values()))**2
hist(tmp)
axvline(np.median(tmp), color='k', linestyle='dashed', linewidth=1)
title('Natural Scenes \n median: {0:.3f}'.format(np.median(tmp)))
xlabel('Explainable Var - $(CC_{max})^2$')
savefig('exp_var_static.png')


scatter(np.log(np.concatenate(mean_resp_all['natural_scenes'].values())), np.concatenate(CC_max_all['natural_scenes'].values()), alpha=.15, s=50, color='k')
scatter(np.log(np.concatenate(mean_resp_all['natural_scenes'].values())), np.concatenate(CC_max_all['natural_scenes'].values()), alpha=.15, s=50, color='b')

scatter((np.concatenate(mean_resp_all['natural_scenes'].values())), np.concatenate(CC_max_all['natural_scenes'].values()), alpha=.15, s=50, color='k')
scatter((np.concatenate(mean_resp_all['natural_scenes'].values())), np.concatenate(CC_max_all['natural_scenes'].values()), alpha=.15, s=50, color='b')

xlim(0, .08)




with open('/home/michaelo/variability_new_' + str(std) + '_CCmax.pkl', 'r') as handle:
    CC_max_all2 = pickle.load(handle)

dfc = pd.concat([df_nma, df_ns, df_dg, df_sg, df_lsn], ignore_index=True)

dfall = {}
for s in CC_max_all2.keys():

    df = []
    for exp in CC_max_all2[s].keys():
        print(exp)

        if len(CC_max_all2[s][exp])==0:
            continue

        if s in ['drifting_gratings', 'static_gratings']:
            unma = unnatural_movie_analysis(exp, .05,  movie_list=[s], load_data=False)
            cell_ids = unma.cell_ids
        elif s in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'natural_scenes']:
            nma = natural_movie_analysis(exp, .05,  movie_list=[s], load_data=False)
            cell_ids = nma.cell_ids

        area = dfc[dfc.experiment_container_id==exp].area.unique()[0]
        depth_range = dfc[dfc.experiment_container_id==exp].depth_range.unique()[0]
        tld1_name = dfc[dfc.experiment_container_id==exp].tld1_name.unique()[0]
        cre_depth = dfc[dfc.experiment_container_id==exp].cre_depth.unique()[0]

        d1 =   {'experiment_container_id':exp,
                'area':area,
                'depth_range':depth_range,
                'tld1_name':tld1_name,
                'cre_depth':str(cre_depth),
                'CC_max':CC_max_all2[s][exp],
                'cell_ids':cell_ids
                }
        d1 = pd.DataFrame(d1)
        if len(df) == 0:
            df = d1
        else:
            df = pd.concat([df,d1], ignore_index=True)
    dfall[s] = df

for s in stims:
    dfall[s].to_csv(s+'.csv')