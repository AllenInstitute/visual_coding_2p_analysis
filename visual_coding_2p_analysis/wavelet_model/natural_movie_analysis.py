from __future__ import print_function

import numpy as np
from collections import deque
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
from scipy.ndimage.interpolation import zoom
import os
import sys
import copy
from scipy.ndimage.filters import gaussian_laplace
from scipy.misc import imsave
from visual_coding_2p_analysis.l0_analysis import L0_analysis
# from L0_analysis import L0_analysis
from scipy.ndimage.filters import gaussian_filter1d
import os.path

# from progress.bar import Bar

import cPickle as pickle


m = si.BrainObservatoryMonitor()
# drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'
# manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/platform_boc_pre_2018_3_16/manifest.json')


# sys.path.append('/home/michaelo/swdb_2017_tools/projects/deconvolution-inference/')
# sys.path.append('/home/michaelo/swdb_2017_tools/projects/deconvolution-inference/OASIS')
# import ca_tools as tools

##################################

# TODO:

# take vector of centers and crop
# whiten before/after average

##################################


class natural_movie_analysis:
    def __init__(self, experiment_id, downsample=.25, include_scenes=True, movie_list=None, load_data=True, event_min_size=2., median_filter_1=5401):

        self.visual_area = boc.get_ophys_experiments(experiment_container_ids=[experiment_id])[0]['targeted_structure']


        # if experiment_id not in good_exps.keys():
        #     raise Exception('No eye-tracking for this experiment')
        if movie_list is None:
            if include_scenes:
                self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'natural_scenes']
            else:
                self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']
        else:
            self._movie_names = movie_list

        sessions = [x['id'] for x in boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=self._movie_names)]
        self.datasets = [boc.get_ophys_experiment_data(ophys_experiment_id=s) for s in sessions]

        sessions_all = [x['id'] for x in boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'natural_scenes'])]
        self.datasets_all = [boc.get_ophys_experiment_data(ophys_experiment_id=s) for s in sessions_all]

        self.experiment_id = experiment_id
        self.downsample = downsample
        self.event_min_size = event_min_size
        self.median_filter_1 = median_filter_1
        
        # self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']
        # self._movie_names = ['natural_scenes']

        self._movie_warps = {}

        # self._whitened_movie_warps = {}
        self._movie_sample_list = self._get_movie_sample_indexes(self.datasets, self._movie_names)

        self._movie_sample_list_all = self._get_movie_sample_indexes(self.datasets_all, ['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'natural_scenes'])

        # calculate pixels per degree
        self.pixperdeg = 1 / m.pixels_to_visual_degrees(n=1)
        self.mask = m.get_mask()
        self.mask_min = [x.min() for x in np.where(self.mask)]
        self.mask_max = [x.max() for x in np.where(self.mask)]

        self._dffs = None
        self._pupil_size = None
        self._running_speed = None
        self._pupil_locs = None
        self._shift_locs = None
        self._pupil_locs_all = None
        self._shift_locs_all = None
        self._min_max_shift = None
        self._mean_shift_loc = None
        self._corrected_frame_numbers = None
        self._cell_indicies = None
        self._cell_ids = None
        self._motion_correction = None


        if load_data:
            self.print('Loading and processing stimuli', end='', flush=True)
            for mn in self._movie_names:
                self.print('.', end='', flush=True)
                try:
                    with open('/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + mn + '_' + str(self.downsample) + '.pickle', 'rb') as handle:
                        tmp_warp = pickle.load(handle)
                except:
                    for ds in self.datasets:
                        try:
                            tmp_movie = self._get_stimulus_template(ds, mn)
                            # bar = Bar('Processing ' + movie_name, max=len(tmp_movie))
                            tmp = self.warp_movie_to_screen(tmp_movie[0], mn)
                            tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                            for i in range(len(tmp_movie)):
                                tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], mn)
                                # bar.next()
                            with open('/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + mn + '_' + str(self.downsample) + '.pickle', 'wb') as handle:
                                pickle.dump(tmp_warp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            break
                        except:
                            continue
                tmp_warp = np.float32(tmp_warp)
                tmp_warp /= 255
                tmp_warp -= 0.5
                self._movie_warps[mn] = tmp_warp
            self.print('done!', end='', flush=True)

            # try:
            #     with open('/allen/aibs/mat/michaelo/events_' + str(hash(str(self.experiment_id) + str(self._movie_names))) + '.pickle', 'rb') as handle:
            #         self._events = pickle.load(handle)
            # except:
            self._events = None
            self._events = self.events

        self._STA = None
        self._MA = None

        self.chunk = 500

        masks = self.datasets[0].get_roi_mask_array()[self.cell_indicies[0]]
        self.cell_centers = [np.median(np.where(ma), axis=1)/512 for ma in masks]

    @property
    def cell_ids(self):
        if self._cell_ids is None:
            cell_ids = set(self.datasets[0].get_cell_specimen_ids())
            for ds in self.datasets[1:]:
                cell_ids = cell_ids.intersection(set(ds.get_cell_specimen_ids()))
            self._cell_ids = list(cell_ids)
        return self._cell_ids

    @property
    def cell_indicies(self):
        if self._cell_indicies is None:
            self._cell_indicies = [ds.get_cell_specimen_indices(self.cell_ids) for ds in self.datasets]
        return self._cell_indicies

    def _get_movie_sample_indexes(self, datasets, movie_names):
        movie_sample_list = []
        for dataset in datasets:
            movies_used = []
            stim_range = []
            frames = []
            for movie_name in movie_names:
                try:
                    stim_table = dataset.get_stimulus_table(movie_name)

                    if movie_name is 'natural_scenes':
                        frame_index = list(stim_table['frame'] + 1)
                    else:
                        frame_index = list(stim_table['frame'])

                    # frame_index = list(stim_table['frame'])
                    frame_starts = list(stim_table['start'])
                    frame_ends = list(stim_table['end'])
                    # stim_range.append((frame_starts[0], np.maximum(frame_starts[-1] + 1, frame_ends[-1])))
                    # stim_range.append(np.unique(np.concatenate([np.arange(fs,fe+1) for fs, fe in zip(frame_starts,frame_ends)])))
                    ranges = []
                    idxs = []
                    for fs,fe,fi in zip(frame_starts, frame_ends, frame_index):
                        tmp_r = np.arange(fs,fe+1)
                        tmp_i = np.tile(fi, [len(tmp_r)])
                        ranges.append(tmp_r)
                        idxs.append(tmp_i)
                    ranges = np.concatenate(ranges)[::-1]
                    idxs = np.concatenate(idxs)[::-1]
                    uidx = np.unique(ranges, return_index=True)[1][::-1]
                    stim_range.append(ranges[uidx][::-1])
                    frames.append(idxs[uidx][::-1])
                    movies_used.append(movie_name)
                    twoidx = np.where(diff(ranges)==2)[0]
                    stim_range = np.insert(stim_range, twoidx+1, stim_range[twoidx+1]-1)
                    frames = np.insert(frames, twoidx+1, frames[twoidx])
                except:
                    continue
            movie_sample_list.append([movies_used, stim_range, frames])
        return movie_sample_list

    @property
    def pupil_locs(self):
        if self._pupil_locs is None:
            pupil_locs = [dataset.get_pupil_location()[1] * self.pixperdeg * self.downsample for dataset in self.datasets]

            pupil_loc_list = []
            for (sl, ms) in zip(pupil_locs, self._movie_sample_list):
                pupil_loc_list.append([sl[mss] for mss in ms[1]])

            self._pupil_locs = pupil_loc_list
        return self._pupil_locs

    @property
    def shift_locs(self):
        if self._shift_locs is None:
            shift_locs = []
            for p in self.pupil_locs:
                shift_locs.append([(pp - self.mean_shift_loc)[:, ::-1]*[1, -1] for pp in p])

            self._shift_locs = shift_locs
        return self._shift_locs


    @property
    def pupil_locs_all(self):
        if self._pupil_locs_all is None:
            pupil_locs = [dataset.get_pupil_location()[1] * self.pixperdeg * self.downsample for dataset in self.datasets_all]

            pupil_loc_list = []
            for (sl, ms) in zip(pupil_locs, self._movie_sample_list_all):
                pupil_loc_list.append([sl[mss] for mss in ms[1]])

            self._pupil_locs_all = pupil_loc_list
        return self._pupil_locs_all

    @property
    def shift_locs_all(self):
        if self._shift_locs_all is None:
            shift_locs = []
            for p in self.pupil_locs_all:
                shift_locs.append([(pp - self.mean_shift_loc)[:, ::-1]*[1, -1] for pp in p])

            self._shift_locs_all = shift_locs
        return self._shift_locs_all

    @property
    def mean_shift_loc(self):
        if self._mean_shift_loc is None:
            concat_pupil_locs = np.concatenate([np.concatenate(p, axis=0) for p in self.pupil_locs_all], axis=0)
            self._mean_shift_loc = np.nanmean(concat_pupil_locs, axis=0)
        return self._mean_shift_loc


    @property
    def min_max_shift(self):
        if self._min_max_shift is None:
            concat_shift_locs = np.concatenate([np.concatenate(s, axis=0) for s in self.shift_locs_all], axis=0)
            self._min_max_shift = (np.int32(np.nanmin(concat_shift_locs, axis=0)), np.int32(np.nanmax(concat_shift_locs, axis=0)))

        return self._min_max_shift

    def warp_movie_to_screen(self, image, movie_name):
        if movie_name is 'natural_scenes':
            movie_warp = m.natural_scene_image_to_screen(image,  origin='upper')
        else:
            movie_warp = m.natural_movie_image_to_screen(image,  origin='upper')

        return zoom(movie_warp[self.mask_min[0]:self.mask_max[0], self.mask_min[1]:self.mask_max[1]], [self.downsample, self.downsample], order=1)

    def _make_shifted_stim(self, original_stim, shift_locations, frame_numbers):
        '''
        make shifted stimuli

        '''

        sh = original_stim.shape

        # make larger stim defined by maximum shifts with a little extra slack
        shift_stim_shape = (len(shift_locations), sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]), sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]))

        shift_stim = 128*np.ones(shift_stim_shape, dtype='uint8')

        shift_locations = shift_locations + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
        good_shift_locations = ~np.isnan(shift_locations[:, 0])

        for i in range(len(shift_locations)):
            if good_shift_locations[i]:
                shift_stim[i, -sh[1]/2 + np.int32(shift_locations[i, 0]):np.int32(shift_locations[i, 0]) + sh[1]/2,
                              -sh[2]/2 + np.int32(shift_locations[i, 1]):np.int32(shift_locations[i, 1]) + sh[2]/2] = original_stim[frame_numbers[i]]

        return shift_stim

    def whiten_frame(self, frame, sigma):
        if sigma > 0:
            return -gaussian_laplace(frame, [sigma, sigma])
        else:
            return frame

    def _make_shifted_stim_resp_generator(self, original_stim, shift_locations, frame_numbers, dff, sigma=0):
        '''
        make shifted stimuli

        '''
        chunk = self.chunk

        sh = original_stim.shape

        idx = range(0, len(frame_numbers), chunk)

        for cut in idx:

            sl = shift_locations[cut:cut+chunk]
            fn = frame_numbers[cut:cut+chunk]
            cdff = dff[:, cut:cut+chunk]
            # make larger stim defined by maximum shifts with a little extra slack
            shift_stim_shape = (len(sl), sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 3, sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 3)

            original_stim = (np.float32(original_stim)/255) - 0.5

            shift_stim = np.zeros(shift_stim_shape, dtype='float32')

            orig_stim = np.zeros((len(sl), original_stim.shape[1], original_stim.shape[2]), dtype='float32')

            sl = sl + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
            good_shift_locations = ~np.isnan(sl[:, 0])

            for i in range(len(sl)):
                if good_shift_locations[i]:
                    shift_stim[i, -sh[1]/2 + np.int32(sl[i, 0]):np.int32(sl[i, 0]) + sh[1]/2,
                                  -sh[2]/2 + np.int32(sl[i, 1]):np.int32(sl[i, 1]) + sh[2]/2] = self.whiten_frame(original_stim[fn[i]], sigma)
                orig_stim[i] = original_stim[fn[i]]

            yield shift_stim, orig_stim, cdff

    def get_all_shifted_stims(self):
        all_shifted_stims = []
        for (ds, msl, sl, cfn) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers):
            shifted_stims = []
            for (movie_name, sl2, cfn2) in zip(msl[0], sl, cfn):
                if movie_name not in self._movie_warps.keys():
                    tmp_movie = self._get_stimulus_template(ds, movie_name)
                    tmp = self.warp_movie_to_screen(tmp_movie[0], movie_name)
                    tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                    for i in range(len(tmp)):
                        tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], movie_name)
                    self._movie_warps[movie_name] = tmp_warp
                shifted_stims.append(self._make_shifted_stim(self._movie_warps[movie_name], sl2, cfn2))
            all_shifted_stims.append(shifted_stims)
        return all_shifted_stims

    def vectorize_data_simple(self, delays):

        stim_out = []
        cfn_dict = {}
        idx_dict = {}
        cnt = 0
        for i, mn in enumerate(self._movie_names):
            stim_out.append(self._movie_warps[mn])
            cfn_dict[mn] = cnt
            idx_dict[mn] = i
            cnt += len(self._movie_warps[mn])

        stim_out = np.concatenate(stim_out, axis=0)

        events_out = []
        frame_numbers_out = []
        running_speed_out = []
        mov_index_out = []
        shifts_out = []
        weights_out = []
        dffs_out = []

        # for (msl, cfn, ev, sl) in zip(self._movie_sample_list, self.corrected_frame_numbers, self.events, self.shift_locs):
        #     for (movie_name, cfn2, ev2, sl2) in zip(msl[0], cfn, ev, sl):
        #         events_out.append(ev2)
        #         shifts_out.append(sl2)
        #         frame_numbers_out.append(np.array(cfn2) + cfn_dict[movie_name])

        for (msl, cfn, ev,  dff, ru) in zip(self._movie_sample_list, self.corrected_frame_numbers, self.events, self.dffs, self.running_speed):
            for (movie_name, movidx, cfn2, ev2, dff2, ru2) in zip(msl[0], msl[1], cfn, ev, dff,ru):
                ev2[:, :delays[-1]] = 0
                events_out.append(ev2)
                tmp_weights = np.ones(ev2[0].shape)
                tmp_weights[:delays[-1]] = 0
                cuts = np.where(np.diff(movidx)>2)[0]
                for cc in cuts:
                    tmp_weights[delays + cc + 1] = 0
                weights_out.append(tmp_weights)
                frame_numbers_out.append(np.array(cfn2) + cfn_dict[movie_name])
                mov_index_out.append(np.ones(len(cfn2)) * idx_dict[movie_name])
                dffs_out.append(dff2)
                running_speed_out.append(ru2)


        events_out = np.concatenate(events_out, axis=1)
        dffs_out = np.concatenate(dffs_out, axis=1)
        weights_out = np.concatenate(weights_out, axis=0)
        frame_numbers_out = np.concatenate(frame_numbers_out, axis=0)
        mov_index_out = np.concatenate(mov_index_out, axis=0)
        running_speed_out = np.concatenate(running_speed_out, axis=0)



        out_stim = stim_out

        return (out_stim, events_out, frame_numbers_out, weights_out, dffs_out, running_speed_out, mov_index_out, self.cell_centers)

    def vectorize_data(self, delays, correct_eye_pos=False, crop=False):

        stim_out = []
        cfn_dict = {}
        idx_dict = {}
        cnt = 0
        for i, mn in enumerate(self._movie_names):
            stim_out.append(self._movie_warps[mn])
            cfn_dict[mn] = cnt
            idx_dict[mn] = i
            cnt += len(self._movie_warps[mn])

        stim_out = np.concatenate(stim_out, axis=0)

        events_out = []
        frame_numbers_out = []
        running_speed_out = []
        mov_index_out = []
        shifts_out = []
        weights_out = []
        dffs_out = []
        pupil_size_out = []
        motion_correction_out = []

        # for (msl, cfn, ev, sl) in zip(self._movie_sample_list, self.corrected_frame_numbers, self.events, self.shift_locs):
        #     for (movie_name, cfn2, ev2, sl2) in zip(msl[0], cfn, ev, sl):
        #         events_out.append(ev2)
        #         shifts_out.append(sl2)
        #         frame_numbers_out.append(np.array(cfn2) + cfn_dict[movie_name])

        for (msl, cfn, ev, sl, dff, pu, ru, mc) in zip(self._movie_sample_list, self.corrected_frame_numbers, self.events, self.shift_locs, self.dffs, self.pupil_size, self.running_speed, self.motion_correction):
            for (movie_name, cfn2, ev2, sl2, dff2, pu2, ru2, mc2) in zip(msl[0], cfn, ev, sl, dff, pu, ru, mc):
                ev2[:, :delays[-1]] = 0
                events_out.append(ev2)
                shifts_out.append(sl2)
                tmp_weights = np.ones(ev2[0].shape)
                tmp_weights[:delays[-1]] = 0
                weights_out.append(tmp_weights)
                frame_numbers_out.append(np.array(cfn2) + cfn_dict[movie_name])
                mov_index_out.append(np.ones(len(cfn2)) * idx_dict[movie_name])
                dffs_out.append(dff2)
                pupil_size_out.append(pu2)
                running_speed_out.append(ru2)
                motion_correction_out.append(mc2)

        events_out = np.concatenate(events_out, axis=1)
        dffs_out = np.concatenate(dffs_out, axis=1)
        weights_out = np.concatenate(weights_out, axis=0)
        shifts_out = np.concatenate(shifts_out, axis=0)
        frame_numbers_out = np.concatenate(frame_numbers_out, axis=0)
        mov_index_out = np.concatenate(mov_index_out, axis=0)
        pupil_size_out = np.concatenate(pupil_size_out, axis=0)
        running_speed_out = np.concatenate(running_speed_out, axis=0)
        motion_correction_out = np.concatenate(motion_correction_out, axis=0)


        if correct_eye_pos:
            sh = stim_out.shape
            shift_stim_shape = (len(shifts_out),
                                sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 3,
                                sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 3)


            out_stim = np.zeros(shift_stim_shape, dtype='float32')


            shifts = shifts_out + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
            good_shift_locations = ~np.isnan(shifts[:, 0])
            for dd in delays:
                weights_out[np.minimum(np.where(np.isnan(shifts[:,0]))[0] + dd, len(weights_out)-1)] = 0

            or1 = 0
            or2 = 999
            oc1 = 0
            oc2 = 999

            for i in range(len(shifts)):
                if good_shift_locations[i]:
                    # print(-sh[1]/2 + np.int32(shifts[i, 0]))
                    # print(np.int32(shifts[i, 0]) + sh[1]/2)

                    r1 = -sh[1]/2 + np.int32(shifts[i, 0])
                    r2 = np.int32(shifts[i, 0]) + sh[1]/2
                    c1 = -sh[2]/2 + np.int32(shifts[i, 1])
                    c2 = np.int32(shifts[i, 1]) + sh[2]/2

                    or1 = np.maximum(or1, r1)
                    or2 = np.minimum(or2, r2)
                    oc1 = np.maximum(oc1, c1)
                    oc2 = np.minimum(oc2, c2)
                    out_stim[i, r1:r2, c1:c2] = stim_out[frame_numbers_out[i]]

        else:
            out_stim = stim_out

        if crop:
            out_stim = out_stim[:,or1:or2, oc1:oc2]

        return (out_stim, events_out, frame_numbers_out, weights_out, shifts_out, dffs_out, pupil_size_out, running_speed_out, mov_index_out, motion_correction_out, self.cell_centers)



    def keras_generator(self, delays=7, batch_size=400, cell=0, scale=5, flatten=True, center=None, crop_size=None, shuffle=True, color_chan=False, log_transform_events=True, correct_eye_pos=False, gaussian_filter=0):
        from keras.engine.training import _standardize_input_data, _make_batches, _standardize_sample_weights
        
        if type(cell) is int:
            cell = [cell]

        if type(delays) is int:
            delays = range(delays)

        (stim, events, frame_numbers, weights, shifts) = self.vectorize_data(delays)


        evidx = np.where(events)[0]
        print(str(len(frame_numbers)) + ' Samples')
        print(str(len(evidx)) + ' Events')

        if correct_eye_pos:
            sh = stim.shape
            shift_stim_shape = (len(shifts),
                                sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 3,
                                sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 3)


            out_stim = np.zeros(shift_stim_shape, dtype='float32')


            shifts = shifts + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
            good_shift_locations = ~np.isnan(shifts[:, 0])
            for dd in delays:
                weights[np.minimum(np.where(np.isnan(shifts[:,0]))[0] + dd, len(weights)-1)] = 0

            for i in range(len(shifts)):
                if good_shift_locations[i]:
                    # print(-sh[1]/2 + np.int32(shifts[i, 0]))
                    # print(np.int32(shifts[i, 0]) + sh[1]/2)
                    out_stim[i, -sh[1]/2 + np.int32(shifts[i, 0]):np.int32(shifts[i, 0]) + sh[1]/2,
                                -sh[2]/2 + np.int32(shifts[i, 1]):np.int32(shifts[i, 1]) + sh[2]/2] = stim[frame_numbers[i]]

            stim = out_stim
            frame_numbers_i = np.arange(len(frame_numbers))
        else:
            frame_numbers_i = frame_numbers

        if color_chan:
            stim = stim[:, None, :, :]

        if crop_size is not None and center is not None:
            crop_range = np.arange(-crop_size/2, crop_size/2)
            stim = stim[:, (center[0]-crop_size/2):(center[0]+crop_size/2), (center[1]-crop_size/2):(center[1]+crop_size/2)]

        if flatten:
            stim = stim.reshape(stim.shape[0], -1)

        events = np.asarray(events)
        events = events[cell].T * scale


        if log_transform_events:
            events = np.log(1 + events)


        if gaussian_filter > 0:
            events = gaussian_filter1d(events, gaussian_filter)

        index_array = np.arange(events.shape[0])

        tlist = [1, 0] + list(range(2, np.ndim(stim) + 1))
        batches = _make_batches(events.shape[0], batch_size)
        while 1:
            if shuffle:
                np.random.shuffle(index_array)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                frame_numbers_b = frame_numbers[batch_ids]
                batch_ids_stim = [frame_numbers_i[np.maximum(0, batch_ids - d)] for d in delays]
                x_batch = _standardize_input_data(stim[batch_ids_stim, :].transpose(tlist), ['x_batch'])

                y_batch = _standardize_input_data(events[batch_ids, :], ['y_batch'])

                w_batch = weights[batch_ids]

                w_batch[frame_numbers_b < delays[-1]] = 0.
                w_batch = _standardize_sample_weights(w_batch, ['w_batch'])
                yield (x_batch, y_batch, w_batch)


    def compute_STA(self, event_type='OASIS', delays=7, pre_whiten=True, post_whiten=False, sigma=5, subtract_mean=True, write_images=False, write_delay=1, write_corrected=True):

        if event_type not in self.events.keys():
            raise ValueError('Please specifiy one of the following for event_type: ' + str(self.events.keys()))

        if self._STA is None:
            STA = list(np.zeros(delays, dtype='float32'))
            STAc = list(np.zeros(delays, dtype='float32'))
            MA = list(np.zeros(delays, dtype='float32'))
            MAc = list(np.zeros(delays, dtype='float32'))
            count = list(np.zeros(delays, dtype='float32'))
            scount = list(np.zeros(delays, dtype='float32'))

            movie_dict = self._movie_warps

            for (ds, msl, sl, cfn, dff, ci) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers, self.events[event_type], self.cell_indicies):
                for (movie_name, sl2, cfn2, dff2) in zip(msl[0], sl, cfn, dff):

                    if movie_name not in movie_dict.keys():
                        tmp_movie = self._get_stimulus_template(ds, movie_name)

                        tmp = self.warp_movie_to_screen(tmp_movie[0], movie_name)
                        tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                        for i in range(len(tmp_movie)):
                            tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], movie_name)

                        with open('/allen/programs/braintv/workgroups/cortexmodels/michaelo/' + movie_name + '_' + str(self.downsample) + '.pickle', 'wb') as handle:
                            pickle.dump(tmp_warp, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        movie_dict[movie_name] = tmp_warp

                    if pre_whiten:
                        ssg = self._make_shifted_stim_resp_generator(movie_dict[movie_name], sl2, cfn2, dff2, sigma)
                    else:
                        ssg = self._make_shifted_stim_resp_generator(movie_dict[movie_name], sl2, cfn2, dff2)

                    for tmp_warp, orig, dff3 in ssg:
                        for d in range(delays):
                            STAc[d] += np.tensordot(dff3[:, d:], tmp_warp[:(len(tmp_warp)-d)][None, ...], axes=[1, 1])[:, 0]
                            STA[d] += np.tensordot(dff3[:, d:], orig[:(len(orig)-d)][None, ...], axes=[1, 1])[:, 0]
                            MAc[d] += np.sum(tmp_warp[:(len(tmp_warp)-d)], axis=0)
                            MA[d] += np.sum(orig[:(len(orig)-d)], axis=0)
                            scount[d] += np.sum(dff3[:, d:], axis=1)
                            count[d] += dff3[:, d:].shape[1]

            self._STAc = [mm/sc[:, None, None] for mm, sc in zip(STAc, scount)]
            self._STA = [mm/sc[:, None, None] for mm, sc in zip(STA, scount)]
            self._MA = [mm/cc for mm, cc in zip(MA, count)]
            self._MAc = [mm/cc for mm, cc in zip(MAc, count)]

        out_sta = copy.deepcopy(self._STA)
        out_stac = copy.deepcopy(self._STAc)

        if subtract_mean:
            out_sta = [x - y[None, ...] for x, y in zip(out_sta, self._MA)]
            out_stac = [x - y[None, ...] for x, y in zip(out_stac, self._MAc)]

        if post_whiten:
            for o in range(len(out_sta)):
                for on in range(out_sta[o].shape[0]):
                    out_sta[o][on] = -gaussian_laplace(out_sta[o][on], [sigma, sigma])
                    out_stac[o][on] = -gaussian_laplace(out_stac[o][on], [sigma, sigma])

        if write_images:
            for on in range(out_sta[write_delay].shape[0]):
                outname = '/tmp/'


        return out_sta, out_stac

    def _get_stimulus_template(self, dataset, stim_name):
        out = dataset.get_stimulus_template(stim_name)
        if stim_name is 'natural_scenes':
            out = np.vstack([128*np.ones((1, out.shape[1], out.shape[2]), dtype='uint8'), out])
        return out

    @property
    def dffs(self):
        if self._dffs is None:
            dffs = [dataset.get_dff_traces()[1][ci, :] for dataset, ci in zip(self.datasets, self.cell_indicies)]

            dffs_list = []
            for (d, ms) in zip(dffs, self._movie_sample_list):
                dffs_list.append([d[:, mss] for mss in ms[1]])

            self._dffs = dffs_list
        return self._dffs

    @property
    def pupil_size(self):
        if self._pupil_size is None:
            pupil_size = [dataset.get_pupil_size()[1] for dataset in self.datasets]

            pupil_size_list = []
            for (d, ms) in zip(pupil_size, self._movie_sample_list):
                pupil_size_list.append([d[mss] for mss in ms[1]])

            self._pupil_size = pupil_size_list
        return self._pupil_size


    @property
    def running_speed(self):
        if self._running_speed is None:
            running_speed = [dataset.get_running_speed()[0] for dataset in self.datasets]

            running_speed_list = []
            for (d, ms) in zip(running_speed, self._movie_sample_list):
                running_speed_list.append([d[mss] for mss in ms[1]])

            self._running_speed = running_speed_list
        return self._running_speed

    @property
    def motion_correction(self):
        if self._motion_correction is None:
            motion_correction = [np.array(dataset.get_motion_correction()[['x_motion', 'y_motion']]) for dataset in self.datasets]
            motion_correction_list = []
            for (d, ms) in zip(motion_correction, self._movie_sample_list):
                motion_correction_list.append([d[mss] for mss in ms[1]])

            self._motion_correction = motion_correction_list
        return self._motion_correction

    def _apply_deconvolution(self, dff):
        out = tools.ca_deconvolution(dff[0], l0=True)

        for d in dff[1:]:
            tmp = tools.ca_deconvolution(d, l0=True)
            for k in tmp.keys():
                out[k] = np.vstack([out[k], tmp[k]])
        return out

    @property
    def events(self):
        if self._events is None:
            # events = [L0_analysis(dataset.get_dff_traces()[1][ci, :]).get_events() for dataset, ci in zip(self.datasets, self.cell_indicies)]
            events = []
            for dataset, ci in zip(self.datasets, self.cell_indicies):
                # expid = dataset.get_metadata()['ophys_experiment_id']
                # fname = '/allen/aibs/mat/michaelo/' + str(expid) + '_events.npz'
                # if os.path.isfile(fname):
                #     events.append(np.load(fname)['ev'][ci, :])
                # else:
                events.append(L0_analysis(dataset, event_min_size=self.event_min_size, median_filter_1 = self.median_filter_1).get_events()[ci, :])
            # events = [L0_analysis(dataset).get_events()[ci, :] for dataset, ci in zip(self.datasets, self.cell_indicies)]

            events_list = []
            for (d, ms) in zip(events, self._movie_sample_list):
                events_list.append([d[:, mss] for mss in ms[1]])

            self._events = events_list

            # with open('/allen/aibs/mat/michaelo/events_' + str(hash(str(self.experiment_id) + str(self._movie_names))) + '.pickle', 'wb') as handle:
            #     pickle.dump(self._events, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self._events

    @property
    def corrected_frame_numbers(self):
        if self._corrected_frame_numbers is None:
            corrected_frame_numbers = []
            for (dataset, ms) in zip(self.datasets, self._movie_sample_list):
                movie_cfn = ms[2]
                corrected_frame_numbers.append(movie_cfn)
            self._corrected_frame_numbers = corrected_frame_numbers
        return self._corrected_frame_numbers


    def print(self, *args, **kwargs):
        if sys.version_info[:2] < (3, 3):
            flush = kwargs.pop('flush', False)
            print(*args, **kwargs)
            if flush:
                file = kwargs.get('file', sys.stdout)
                file.flush() if file is not None else sys.stdout.flush()
        else:
            print(*args, **kwargs)
