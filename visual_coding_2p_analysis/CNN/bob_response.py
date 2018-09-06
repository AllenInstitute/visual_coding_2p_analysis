import numpy as np

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from visual_coding_2p_analysis.core import get_L0_events

# MANIFEST_FILE = '/allen/aibs/technologyEVENTS_DIR/allensdk_data/2018-01-30T10_59_26.662324/boc/manifest.json'
MANIFEST_FILE = '/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json'
# EVENTS_DIR = '/allen/aibs/mat/gkocker/l0events_threshold2'
MANIFEST_FILE = '/home/michaelbu/Code/new_boc/boc-pre-release/boc/manifest.json'

BAD_CONTAINER_IDS = [511510998,
511510681,
517328083,
527676429,
527550471,
530243910,
570278595,
571039045,
585905043,
587695553,
596780703,
598134911,
599587151,
605113106]


def compute_response_dict_from_bob(all_trials=False):

    boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)
    expt_list = boc.get_ophys_experiments(session_types=['three_session_B'])

    response_dict = {}
    for expt in expt_list:
        # Skip all experiment container ids that have been labeled 'BAD'
        if expt['experiment_container_id'] in BAD_CONTAINER_IDS:
            continue
        eid = expt['id']
        print "Processing experiment ", eid

        cre = expt['cre_line'].split('-')[0]
        depth = expt['imaging_depth']
        area = expt['targeted_structure']

        response_dict[area] = response_dict.get(area, {})
        response_dict[area][cre] = response_dict[area].get(cre, {})
        response_dict[area][cre][depth] = response_dict[area][cre].get(depth, [])

        try:
            # events = np.load(os.path.join(EVENTS_DIR, 'expt_'+str(eid)+'_events.npy'))
            events = get_L0_events(eid)
        except IOError:
            print 'Events file not found for ', str(eid), ".  Skipping."
            continue
        
        print events.shape

        data = boc.get_ophys_experiment_data(eid)

        stim_table = data.get_stimulus_table('natural_scenes')

        frames = np.unique(stim_table.frame)
        # print frames

        if not all_trials:
            response = np.zeros((len(frames), events.shape[0]))
        else:
            response_temp_list = [] 
        for f in frames:
            f_table = stim_table[stim_table.frame==f]

            trials = np.array([np.sum(events[:,row.start:row.end], axis=1) for i, row in f_table.iterrows()])
            # print trials.shape
            if not all_trials:
                response[f] = np.mean(trials, axis=0)
            else:
                response_temp_list.append(trials)  

        if all_trials:
            frames_sort_index = np.argsort(frames)

            response = [response_temp_list[i] for i in frames_sort_index]
            trials = [frames[i]*np.ones(response_temp_list[frames[i]].shape[0]) for i in frames_sort_index]

            # put grey screen (f== -1) at the end of the array, should be sorted to first
            grey = response.pop(0)
            response.append(grey)  

            grey_trial = trials.pop(0)
            trials.append(grey_trial)
            trials = np.hstack(trials)


            response = np.vstack(response)  # [stim_table.shape[0], events.shape[0]]

            # add list of trial ids here
            response_dict[area][cre][depth].append((response, trials))


        else:
            response_dict[area][cre][depth].append(response)

    return response_dict

def which_layer(depth):
    if depth < 200:
        return 'layer23'
    if depth >= 200 and depth < 300:
        return 'layer4'
    if depth >= 300 and depth < 500:
        return 'layer5'
    if depth >= 500:
        return 'layer6'

def response_dict_by_layer_from_depth(response_dict):

    all_trials = type(response_dict['VISal']['Cux2'][175][0])==tuple
    if all_trials:
        print "Detected that response_dict contains individual trials, not trial average"

    trials_list = []
    new_response_dict = {}
    print "Constructing response matrices per location"
    for area in response_dict.keys():
        new_response_dict[area] = {}
        for cre in response_dict[area].keys():

            depth_dict = {'layer23': [],
                          'layer4': [],
                          'layer5': [],
                          'layer6': []}
            for depth in response_dict[area][cre].keys():
            
                if cre=='Nr5a1' or cre=='Scnn1a':
                    depth_dict['layer4'] += response_dict[area][cre][depth]
                else:
                    depth_dict[which_layer(depth)] += response_dict[area][cre][depth]
                
                
            new_response_dict[area][cre] = {}
            # create single representation with all neurons in the dataset for each location
            for layer in depth_dict.keys():
                if len(depth_dict[layer])!=0: 

                    if all_trials:
                        trials = [r[1] for r in depth_dict[layer]]
                        # print trials
                        trials = np.vstack(trials)
                        std = np.std(trials, axis=0)
                        assert np.allclose(std, 0.0), "Trials don't match across experiments"
                        trials_list.append(trials[0])
                        #print area+" "+cre+" "+str(depth)+" num_trials:  "+str(len(trials))

                        depth_dict[layer] = np.hstack([r[0] for r in depth_dict[layer]])
                    else:
                        depth_dict[layer] = np.hstack(depth_dict[layer])

                    new_response_dict[area][cre][layer] = depth_dict[layer]

    if all_trials:
        trials = np.vstack(trials_list)
        std = np.std(trials, axis=0)
        assert np.allclose(std, 0.0), "Trials don't match across locations"
        trials_list.append(trials[0])

        return new_response_dict, trials[0]
    else:
        return new_response_dict


if __name__ == '__main__':

    import os, pickle

    PREFIX_DIR = '/allen/programs/braintv/workgroups/cortexmodels/michaelbu/DeepNetworkPhysiology'
    RESPONSE_CACHE = 'bob_ns_response_alltrials_dict.pkl'
    RESPONSE_CACHE = os.path.join(PREFIX_DIR, RESPONSE_CACHE)
    LAYER_RESPONSE_CACHE = 'bob_ns_response_dict_by_layer_alltrials.pkl'
    LAYER_RESPONSE_CACHE = os.path.join(PREFIX_DIR, LAYER_RESPONSE_CACHE)

    if os.path.exists(RESPONSE_CACHE):
        print "Opening cached response_dict"
        with open(RESPONSE_CACHE, 'r') as f:
            response_dict = pickle.load(f)
    else:
        print "Computing response_dict, saving to cache"
        response_dict = compute_response_dict_from_bob(all_trials=True)
        with open(RESPONSE_CACHE, 'w') as f:
            pickle.dump(response_dict, f)

    response_dict = response_dict_by_layer_from_depth(response_dict)
    with open(LAYER_RESPONSE_CACHE, 'w') as f:
        pickle.dump(response_dict, f)