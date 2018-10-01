#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:50:27 2018

@author: saskiad

snippet from nicain
"""

from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise as LocallySparseNoiseBase
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stimulus_info
import h5py
import core

compare_or_base = 'compare'
oeid = 652339241
cell_specimen_id = 662258048
manifest_path = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file=manifest_path)
data_set = boc.get_ophys_experiment_data(oeid)
cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]

class LocallySparseNoise(LocallySparseNoiseBase):
    
    @staticmethod
    def from_analysis_file(data_set, analysis_file, stimulus):
        lsn = LocallySparseNoise(data_set, stimulus)

        lsn.populate_stimulus_table()

        if stimulus == stimulus_info.LOCALLY_SPARSE_NOISE:
            stimulus_suffix = stimulus_info.LOCALLY_SPARSE_NOISE_SHORT
        elif stimulus == stimulus_info.LOCALLY_SPARSE_NOISE_4DEG:
            stimulus_suffix = stimulus_info.LOCALLY_SPARSE_NOISE_4DEG_SHORT
        elif stimulus == stimulus_info.LOCALLY_SPARSE_NOISE_8DEG:
            stimulus_suffix = stimulus_info.LOCALLY_SPARSE_NOISE_8DEG_SHORT

        with h5py.File(analysis_file, "r") as f:
            lsn._cell_index_receptive_field_analysis_data = LocallySparseNoise.read_cell_index_receptive_field_analysis(f, stimulus)

        return lsn

analysis_file = '/Volumes/aibs/technology/nicholasc/pipeline/brain_observatory/GH-141/%s/results/%s/%s_analysis.h5' % (compare_or_base, oeid, oeid)
lsn = LocallySparseNoise.from_analysis_file(data_set, 
                                            analysis_file,
                                            stimulus_info.LOCALLY_SPARSE_NOISE_4DEG) 

lsn.plot_receptive_field_analysis_data(cell_index)