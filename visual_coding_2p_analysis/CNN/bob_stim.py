import numpy as np

from stim_rep import get_image_stimulus, get_grating_stimulus, get_template


def add_channels(input_tensor, num_channels=3):

    return np.repeat(input_tensor[...,np.newaxis], num_channels, axis=-1)

def get_stim(new_size):

    print "Constructing stimulus"
    ns_stim = get_image_stimulus('natural_scenes', new_size=new_size)
    ns_stim = add_channels(ns_stim)
    ns_stim = np.vstack([ns_stim, 127*np.ones((1,)+ns_stim.shape[1:], dtype=np.uint8)])

    return ns_stim
