from __future__ import print_function
import numpy as np
# from FastLZeroSpikeInference import fast
# from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter
import sys
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import cPickle as pickle
import warnings
import os

# l0 = fast.arfpop
medfilt = lambda x, s: median_filter(x, s, mode='constant')

class L0_analysis:
    """
    Class for performing L0 event detection using an automatically determined
    lambda value. lambda is chosen by finding smallest lambda such that the size of
    the smallest detected event is greater or equal to event_min_size*robust std of noise.
    If such a lambda cannot be found, it uses the largest lambda that returns some non-zero
    values.

    Parameters
    ----------
    dataset : a dataset object (returned from get_ophys_experiment_data) or ophys_experiment_id or raw data
    event_min_size : smallest allowable event in units of noise std [default: 1.0]
    noise_scale : dff traces are rescaled so that the noise std is this value [default: 0.1]
    median_filter_1 : the length of the window for long time scale median filter detrending to
                      estimate dff from corrected_fluorescence_traces [default: 2001]
    median_filter_2 : the length of the window for short time scale median filter detrending [default: 101]
    halflife_ms : half-life of the indicator in ms, used to override lookup [default: None]
    sample_rate_hz : sampling rate of data in Hz
    genotype : genotype of cell line (use if passing raw data without specifying the halflife_ms)
    cache_directory : directory to cache estimated dffs and events
    manifest_file : Brain Observatory manifest to use

    Attributes
    ----------
    noise_stds : estimates of the std of the noise for each trace
    lambdas : chosen lambda for each trace
    gamma : the gamma decay constant calculated from the half-life
    dff_traces : detrended df/f traces

    Examples
    --------
    >>> l0a = L0_analysis(dataset)
    >>> events = l0a.get_events()

    """
    def __init__(self, dataset,
                       manifest_file='/allen/aibs/technology/allensdk_data/platform_boc_pre_2018_3_16/manifest_file.json',
                       event_min_size=2., noise_scale=.1, median_filter_1=5401, median_filter_2=101, halflife_ms=None,
                       sample_rate_hz=30, genotype='Unknown', L0_constrain=False,
                       cache_directory='/allen/aibs/technology/allensdk_data/platform_events_pre_2018_3_19/'):


        if type(dataset) is int:
            if manifest_file is None:
                boc = BrainObservatoryCache()
            else:
                boc = BrainObservatoryCache(manifest_file=manifest_file)
            dataset = boc.get_ophys_experiment_data(ophys_experiment_id=dataset)

        try:
            self.metadata = dataset.get_metadata()
            # dff_traces = dataset.get_dff_traces()[1]
            self.corrected_fluorescence_traces = dataset.get_corrected_fluorescence_traces()[1]
        except:
            self.metadata = {'genotype':genotype, 'ophys_experiment_id':999}
            self.corrected_fluorescence_traces = dataset


        self.sample_rate_hz = sample_rate_hz
        self.event_min_size = event_min_size
        self.noise_scale = noise_scale

        if halflife_ms is None:
            self.halflife = self.get_halflife()
        else:
            self.halflife = halflife_ms

        self.median_filter_1 = median_filter_1
        self.median_filter_2 = median_filter_2
        self.L0_constrain = L0_constrain
        self.cache_directory = cache_directory

        self._dff_traces = None
        self._fit_params = None

        self._gamma = None

        self.lambdas = []
        self.l0_func = None

    @property
    def l0(self):
        if self.l0_func is None:
            from FastLZeroSpikeInference import fast
            self.l0_func = fast.arfpop

        return self.l0_func

    @property
    def evfile(self):
        return os.path.join(self.cache_directory, str(self.metadata['ophys_experiment_id']) +  '_' +
                                                  str(hash(str(self.event_min_size) +
                                                  str(self.noise_scale) +
                                                  str(self.median_filter_1) +
                                                  str(self.median_filter_2) +
                                                  str(self.halflife) +
                                                  str(self.sample_rate_hz) +
                                                  str(self.L0_constrain))) + '_events.npz')

    @property
    def dff_file(self):
        return os.path.join(self.cache_directory, str(self.metadata['ophys_experiment_id']) +  '_' +
                                                  str(hash(str(self.noise_scale) +
                                                  str(self.median_filter_1) +
                                                  str(self.median_filter_2) +
                                                  str(self.halflife) +
                                                  str(self.sample_rate_hz))) + '_dff.npz')

    @property
    def dff_traces(self):
        if self._dff_traces is None and os.path.isfile(self.dff_file):
            self._dff_traces = np.load(self.dff_file)['dff']
        elif self._dff_traces is None:
            self.print('Median filter detrending in progress', end='', flush=True)
            self.noise_stds = []
            dff_traces = np.copy(self.corrected_fluorescence_traces)

            N = dff_traces.shape[0]
            num_small_baseline_frames = np.zeros((N, ), dtype=int)

            for dff in dff_traces:

                sigma_f = self.noise_std(dff)

                # long timescale median filter for baseline subtraction
                tf = medfilt(dff, self.median_filter_1)
                dff -= tf
                dff /= np.maximum(tf, sigma_f)

                num_small_baseline_frames[n] = np.sum(tf <= sigma_f)

                sigma_dff = self.noise_std(dff)
                self.noise_stds.append(sigma_dff)

                # short timescale detrending
                tf = medfilt(dff, self.median_filter_2)
                tf = np.minimum(tf, 2.5*sigma_dff)
                dff -= tf

                self.print('.', end='', flush=True)

            self._dff_traces = dff_traces
            np.savez(self.dff_file, dff=dff_traces, num_low_baseline=num_small_baseline_frames)
            self.print('done!')
        return self._dff_traces

    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = np.exp(-np.log(2)*1000/(self.halflife*self.sample_rate_hz))
        return self._gamma


    def get_halflife(self):
        genotype = self.metadata['genotype']

        if 'Cux2' in genotype and 'Ai93' in genotype:
            return 239
        elif 'Emx1' in genotype and 'Ai96' in genotype:
            return 436
        elif 'tetO' in genotype and '6s' in genotype:
            return 348
        elif 'Emx1' in genotype and 'Ai94' in genotype:
            return 649
        elif 'Emx1' in genotype and 'Ai93' in genotype:
            return 315
        elif 'Ai93' in genotype:
            return 315
        else:
            warnings.warn('Genotype is unknown, assuming halflife of 315 ms')
            return 315


    def noise_std(self, x, filt_length=31):
        if any(np.isnan(x)):
            return np.NaN
        x = x - medfilt(x, filt_length)
        # first pass removing big pos peak outliers
        x = x[x< 1.5*np.abs(x.min())]
        rstd = self.robust_std(x)
        # second pass removing remaining pos and neg peak outliers
        x = x[abs(x) < 2.5*rstd]
        return self.robust_std(x)


    def robust_std(self, x):
        '''
        Robust estimate of std
        '''
        MAD = np.median(np.abs(x - np.median(x)))
        return 1.4826*MAD


    def get_events(self, event_min_size=None):
        if event_min_size is not None:
            self.event_min_size = event_min_size

        if os.path.isfile(self.evfile):
            events = np.load(self.evfile)['ev']
        else:

            self.print('Calculating events in progress', flush=True)

            events = []
            for n, dff in enumerate(self.dff_traces):
                dff *= self.noise_scale / self.noise_stds[n]

                if any(np.isnan(dff)):
                    tmp = np.NaN*np.zeros(dff.shape)
                    self._lambdas.append(np.NaN)
                else:
                    tmp = dff[:]

                    (tmp, l) = self.bracket(tmp, self.noise_scale, 0, self.noise_scale, .0001, self.event_min_size)

                    events.append(tmp * self.noise_stds[n] / self.noise_scale)
                    self.lambdas.append(l)
                self.print('.', end='', flush=True)
            events = np.array(events)
            np.savez(self.evfile, ev=events)
            self.print('done!')
        return np.array(events)

    def bracket(self, dff, n, s1, step, step_min, event_min_size, bisect=False):
        l = s1 + step
        if l < step:
            l = step
            s1 += step
        print(l)
        tmp = self.l0(dff, self.gamma, l, self.L0_constrain)['pos_spike_mag']

        if len(tmp[tmp > 0]) == 0 and bisect is True:
            return self.bracket(dff, n, s1 - 5*step, step, step_min, event_min_size)

        if step == step_min:
            if np.min(tmp[tmp > 0]) > n * event_min_size and bisect is True:
                return self.bracket(dff, n, s1 - 5*step, step, step_min, event_min_size)
            else:
                while len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) < n * event_min_size:
                    lasttmp = tmp[:]
                    l += step
                    print(l)
                    tmp = self.l0(dff, self.gamma, l, self.L0_constrain)['pos_spike_mag']
                if len(tmp[tmp > 0]) == 0:
                    return (lasttmp, l-step)
                else:
                    return (tmp, l)

        if len(tmp[tmp > 0]) == 0 and bisect is False:
            return self.bracket(dff, n, s1 + .5*step - step/10, step/10, step_min, event_min_size, True)


        if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) < n * event_min_size:
            return self.bracket(dff, n, l, step, step_min, event_min_size)

        if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) > n * event_min_size and step > step_min and bisect is False:
            return self.bracket(dff, n, s1 + .5*step - step/10, step/10, step_min, event_min_size, True)


        if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) > n * event_min_size and step > step_min and bisect is True:
            return self.bracket(dff, n, s1 - 5*step, step, step_min, event_min_size)


    def print(self, *args, **kwargs):
        if sys.version_info[:2] < (3, 3):
            flush = kwargs.pop('flush', False)
            print(*args, **kwargs)
            if flush:
                file = kwargs.get('file', sys.stdout)
                file.flush() if file is not None else sys.stdout.flush()
        else:
            print(*args, **kwargs)
