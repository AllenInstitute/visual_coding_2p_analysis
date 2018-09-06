import sys
import numpy as np

from scipy.misc import imresize

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.stimulus_info import BrainObservatoryMonitor
from allensdk.brain_observatory.stimulus_info import make_display_mask

Y, X = np.where(make_display_mask().T)

YMIN, YMAX = np.min(Y), np.max(Y)
XMIN, XMAX = np.min(X), np.max(X)

if sys.platform=='darwin':
    MANIFEST_FILE='/Users/michaelbu/Code/BrainObservatory/boc/manifest.json'
if sys.platform=='linux2':
    MANIFEST_FILE='/home/michaelbu/Data/BrainObservatory/boc/manifest.json'

SESSION_DICT = {'locally_sparse_noise': 'three_session_C', 
                'natural_scenes':       'three_session_B',
                'natural_movie_one':    'three_session_A',
                'natural_movie_two':    'three_session_C',
                'natural_movie_three':  'three_session_A',
                'drifting_gratings':    'three_session_A',
                'static_gratings':      'three_session_B'}



NEW_SIZE = (200, 256)


def get_template(stimulus_type):

    session_type = SESSION_DICT[stimulus_type]
    boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)

    expt_C_id = boc.get_ophys_experiments(session_types=[session_type])[0]['id']
    expt_C = boc.get_ophys_experiment_data(expt_C_id)

    stim_template = expt_C.get_stimulus_template(stimulus_type)
    return stim_template



def get_image_stimulus(stimulus_type='locally_sparse_noise', new_size=NEW_SIZE, static=True):

    """stimulus_type should be one of 
            'locally_sparse_noise'
            'natural_scenes'
            'natural_movie_one'
            'natural_movie_two'
            'natural_movie_three'"""

    lsn_template = get_template(stimulus_type)
    bom = BrainObservatoryMonitor()

    lsn_stim = []
    for x in lsn_template:
        if stimulus_type=='locally_sparse_noise':
            im = bom.lsn_image_to_screen(x, origin='upper')
        elif stimulus_type=='natural_scenes':
            im = bom.natural_scene_image_to_screen(x, origin='upper')
        elif stimulus_type.split('_')[1]=='movie':
            im = bom.natural_movie_image_to_screen(x, origin='upper')
        im = im[YMIN:YMAX, XMIN:XMAX]
        # print im.shape
        im = imresize(im, new_size, interp='nearest')  # maybe should use 'bilinear' for pixel distortion
        lsn_stim.append(im)

    lsn_stim = np.stack(lsn_stim, axis=0)

    if not static:
        pass

    return lsn_stim


def get_table(stimulus_type):

    session_type = SESSION_DICT[stimulus_type]
    boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)

    expt_C_id = boc.get_ophys_experiments(session_types=[session_type])[0]['id']
    expt_C = boc.get_ophys_experiment_data(expt_C_id)

    stim_template = expt_C.get_stimulus_table(stimulus_type)
    return stim_template


def get_grating_stimulus(stimulus_type='static_gratings', new_size=NEW_SIZE, static=True, unique=True):

    stim_table = get_table(stimulus_type)

    bom = BrainObservatoryMonitor()


    if stimulus_type=='static_gratings':

        stim_table = stim_table[['orientation', 'spatial_frequency', 'phase']]
        if unique:
            stim_table = stim_table.drop_duplicates()

        stim = []
        for i, row in stim_table.iterrows():
            im = bom.grating_to_screen(row.phase, row.spatial_frequency, row.orientation)
            im = im[YMIN:YMAX, XMIN:XMAX]
            # print im.shape
            im = imresize(im, new_size, interp='nearest')  # maybe should use 'bilinear' for pixel distortion
            stim.append(im)

        stim = np.stack(stim, axis=0)

    
    elif stimulus_type=='drifting_gratings':
        stim_table = stim_table[['orientation', 'temporal_frequency', 'blank_sweep']]
        if unique:
            stim_table = stim_table.drop_duplicates()

    return stim, stim_table

def main():

    import matplotlib.pyplot as plt
    
    # lsn_stim = get_image_stimulus('locally_sparse_noise')

    # plt.imshow(lsn_stim[0], interpolation='nearest')

    # print lsn_stim.shape

    # ns_stim = get_image_stimulus('natural_scenes')

    # plt.figure()
    # plt.imshow(ns_stim[0])

    # plt.figure()
    # plt.imshow(make_display_mask().T)

    # mov_stim = get_image_stimulus('natural_movie_one')

    # plt.imshow(mov_stim[0])

    sg_stim = get_grating_stimulus('static_gratings')

    plt.imshow(sg_stim[0])

    plt.show()



if __name__ == '__main__':

    main()
