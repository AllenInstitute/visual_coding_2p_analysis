import numpy as np
import itertools

class gabor_wavelet_preprocessing_linquad:
    def __init__(self, 
                 stimulus,
                 movie_hz=30,
                 temporal_window=11, 
                 spatial_frequencies=[0,2,4,8,16], 
                 temporal_frequencies=[0,2,4],
                 simple_cell_filters=True,
                 complex_cell_filters=True, 
                 angle_spacing=30,  
                 sf_gauss_ratio=0.6,
                 max_spatial_env=0.3,
                 gabor_spacing=3.5,
                 tf_gauss_ratio=10.,
                 max_temp_env=0.3,
                 include_edges=False,
                 output_nonlinearity=None,
                 zscore_by_type=True,
                 zscore_each=False,
                 means = None,
                 stds = None,
                 ):
        """ Parameters
            ----------
            movie_hz : scalar, Hz
                Temporal resolution of the stimulus (e..g. 15)
            gabor_hz : scalar, Hz
                Temporal window of the motion-energy filter (e.g. 10)
            temporal_frequencies : array-like, Hz
                Temporal frequencies of the filters for use on the stimulus
            spatial_frequencies : array-like, degrees
                Spatial frequencies for the filters
            spatial_directions : array-like, degrees
                Direction of filter motion. Degree position corresponds
                to standard unit-circle coordinates.


            sf_gauss_ratio : scalar
                The ratio of spatial frequency to gaussian s.d.
                This controls the number of cycles in a filter
            max_spatial_env : scalar
                Defines the maximum s.d. of the gaussian
            gabor_spacing : scalar
                Defines the spacing between spatial gabors
                (in s.d. units)
            tf_gauss_ratio : scalar
                The ratio of temporal frequency to gaussian s.d.
                This controls the number of temporal cycles
            max_temp_env : scalar
                Defines the maximum s.d. of the temporal gaussian
            aspect_ratio : scalar, horizontal/vertical
                The image aspect ratio. This ensures full image
                coverage for non-square images (e.g. 16:9)
            include_edges : bool
                Determines whether to include filters at the edge
                of the image which might be partially outside the
                stimulus field-of-view
        """

        _, self.vertical_dim, self.horizontal_dim = stimulus.shape
        self.aspect_ratio = self.horizontal_dim/float(self.vertical_dim)
        self.stimulus = stimulus.reshape(stimulus.shape[0], -1)
        self.movie_hz = movie_hz
        if temporal_window is None:
            self.temporal_window = int(movie_hz*(2./3.))
        else:
            self.temporal_window = temporal_window
        self.spatial_frequencies = spatial_frequencies
        self.temporal_frequencies = temporal_frequencies
        self.spatial_directions = np.arange(0, 360, angle_spacing)
        self.sf_gauss_ratio = sf_gauss_ratio
        self.max_spatial_env = max_spatial_env
        self.gabor_spacing = gabor_spacing
        self.tf_gauss_ratio = tf_gauss_ratio
        self.max_temp_env = max_temp_env
        self.include_edges = include_edges
        self.simple_cell_filters = simple_cell_filters
        self.complex_cell_filters = complex_cell_filters
        self.zscore_by_type = zscore_by_type
        self.zscore_each = zscore_each
        self._means = means
        self._stds = stds
        assert (not zscore_by_type and not zscore_each) or (zscore_by_type is not zscore_each),  'Choose only one way to zscore, each channel or by type'

    @property
    def means(self):
        if self._means is None:
            _ = self.compute_filter_responses()
        return self._means

    @property
    def stds(self):
        if self._stds is None:
            _ = self.compute_filter_responses()
        return self._stds


    @property
    def param_list(self):
        """Parametrize a motion-energy pyramid that tiles the stimulus.

        Returns
        -------
        gabor_parameters : np.array, (nfilters, 7)
            Parameters that defined the motion-energy filter
            Each of the `nfilters` has the following parameters:
                * centerx,centery : horizontal and vertical position
                * direction       : direction of motion
                * spatial_freq    : spatial frequency
                * spatial_env     : spatial envelope (gaussian s.d.)
                * temporal_freq   : temporal frequency
                * temporal_env    : temporal envelope (gaussian s.d.)

        Notes
        -----
        Same method as Nishimoto, et al., 2011.
        """

        spatial_frequencies = np.asarray(self.spatial_frequencies).astype(np.float)
        spatial_directions = np.asarray(self.spatial_directions).astype(np.float)
        temporal_frequencies = np.asarray(self.temporal_frequencies).astype(np.float)
        include_edges = int(self.include_edges)

        # normalize temporal frequency to wavelet size
        temporal_frequencies = temporal_frequencies*(self.temporal_window/float(self.movie_hz))

        # We have to deal with zero frequency spatial filters differently
        include_local_dc = True if 0 in spatial_frequencies else False
        spatial_frequencies = np.asarray([t for t in spatial_frequencies if t != 0])

        # add temporal envelope max
        params = list(itertools.product(spatial_frequencies, spatial_directions))

        phases = []
        # 0,1: +/- sin
        # 2,3: +/- cos
        # 4: (sin**2 + cos**2)**.5
        if self.simple_cell_filters:
            phases = phases + [0, 1]

        if self.complex_cell_filters:
            phases = phases + [2, 3]

        gabor_parameters = []

        for spatial_freq, spatial_direction in params:
            spatial_env = min(self.compute_envelope(spatial_freq, self.sf_gauss_ratio), self.max_spatial_env)

            # compute the number of gaussians that will fit in the FOV
            vertical_space = np.floor(((1.0 - spatial_env*self.gabor_spacing)/(self.gabor_spacing*spatial_env))/2.0)
            horizontal_space = np.floor(((self.aspect_ratio - spatial_env*self.gabor_spacing)/(self.gabor_spacing*spatial_env))/2.0)

            # include the edges of screen?
            vertical_space = max(vertical_space, 0) + include_edges
            horizontal_space = max(horizontal_space, 0) + include_edges

            # get the spatial gabor locations
            ycenters = spatial_env*self.gabor_spacing*np.arange(-vertical_space, vertical_space+1) + 0.5
            xcenters = spatial_env*self.gabor_spacing*np.arange(-horizontal_space, horizontal_space+1) + self.aspect_ratio/2.

            for ii, (cx, cy) in enumerate(itertools.product(xcenters,ycenters)):
                for temp_freq in temporal_frequencies:
                    temp_env = min(self.compute_envelope(temp_freq, self.tf_gauss_ratio), self.max_temp_env)

                    if temp_freq == 0 and spatial_direction >= 180:
                        # 0Hz temporal filter doesn't have motion, so
                        # 0 and 180 degrees orientations are the same filters
                        continue
                    for phase in phases:
                        gabor_parameters.append([cx,
                                                 cy,
                                                 spatial_direction,
                                                 spatial_freq,
                                                 spatial_env,
                                                 temp_freq,
                                                 temp_env,
                                                 phase, 
                                                 0,
                                                 spatial_freq])

                    if spatial_direction == 0 and include_local_dc:
                        # add local 0 spatial frequency non-directional temporal filter
                        for phase in set(phases) - set([0,2]): # only use cos simple filters since sin are all 0
                            gabor_parameters.append([cx,
                                                     cy,
                                                     spatial_direction,
                                                     0, # zero spatial freq
                                                     spatial_env,
                                                     temp_freq,
                                                     temp_env,
                                                     phase, 
                                                     1,
                                                     spatial_freq])

        gabor_parameters = np.asarray(gabor_parameters)
        return gabor_parameters

    def compute_envelope(self, freq, ratio):
        return np.inf if freq == 0 else (1.0/freq)*ratio

    def compute_filter_responses(self):
        """Compute the motion-energy filters' response to the stimuli.

        Parameters
        ----------
        stimulus : 3D np.array (n, vdim, hdim)
            The movie frames.
        stimulus_fps : scalar
            The temporal frequency of the stimulus
        gabor_temporal_window : scalar, None
            The number of frames in one filter.
            If None, it defaults to floor(2/3) of `stimulus_fps`
            Similar to Nishimoto, 2011.

        quadrature_combination : function, optional
            Specifies how to combine the channel reponses quadratures.
            The function must take the sin and cos as arguments in order.
            Defaults to: (sin^2 + cos^2)^1/2
        output_nonlinearity : function, optional
            Passes the channels (after `quadrature_combination`) through a
            non-linearity. The function input is the (`n`,`nfilters`) array.
            Defaults to: ln(x + 1)
        dozscore : bool, optional
            Whether to z-score the channel responses in time

        moten_pyramid_parameters: dict
            See :func:`mk_moten_pyramid_params` for details on parameters
            specifiying a motion-energy pyramid.

        Returns
        -------
        filter_responses : np.array, (n, nfilters)
        """

        # stimulus = stimulus.reshape(stimulus.shape[0], -1)

        # if gabor_temporal_window is None:
        #     gabor_temporal_window = int(stimulus_fps*(2./3.))

        # gabor_parameters = mk_moten_pyramid_params(stimulus_fps,
        #                                            gabor_temporal_window,
        #                                            aspect_ratio=aspect_ratio,
        #                                            **moten_pyramid_parameters)

        channels = []

        for idx, gabor_param in enumerate(self.param_list):
            gabor = self.make_3d_gabor((self.horizontal_dim, self.vertical_dim, self.temporal_window),
                                        *gabor_param[:-3],
                                        aspect_ratio=self.aspect_ratio)
            gabor0, gabor90, tgabor0, tgabor90 = gabor
            phase = gabor_param[7]

            channel_sin, channel_cos = self.dotdelay_frames(gabor0, gabor90,
                                                            tgabor0, tgabor90)

            channel_sin, channel_cos = np.float32(channel_sin), np.float32(channel_cos)
            
            if phase == 0:
                channels.append(channel_sin)
            elif phase == 1:
                channels.append(channel_cos)
            elif phase == 2:
                channels.append(channel_sin**2)
            elif phase == 3:
                channels.append(channel_cos**2)

        channels = np.asarray(channels).T

        if self.zscore_each:
            # from scipy.stats import zscore
            if self._means is None:
                self._means = np.mean(channels, axis=0)
            if self._stds is None:
                self._stds = np.std(channels, axis=0)
            channels -= self._means
            channels /= self._stds
        elif self.zscore_by_type:
            if self._means is None or self._stds is None:
                usf = np.unique(self.param_list[:,9])
                utf = np.unique(self.param_list[:,5])
                ugg = np.unique(self.param_list[:,8])
                self._means = np.zeros(len(self.param_list))
                self._stds = np.zeros(len(self.param_list))
                for sf in usf:
                    for tf in utf:
                        for gg in ugg:
                            idx_l = np.bitwise_and(np.bitwise_and(np.bitwise_and(self.param_list[:,9] == sf, self.param_list[:,5]==tf), np.bitwise_or(self.param_list[:,7]==0, self.param_list[:,7]==1)), self.param_list[:,8]==gg)
                            idx_q = np.bitwise_and(np.bitwise_and(np.bitwise_and(self.param_list[:,9] == sf, self.param_list[:,5]==tf), np.bitwise_or(self.param_list[:,7]==2, self.param_list[:,7]==3)), self.param_list[:,8]==gg)
                            self._means[idx_l] = np.mean(channels[:,idx_l])
                            self._means[idx_q] = np.mean(channels[:,idx_q])
                            self._stds[idx_l] = np.sqrt(np.mean((channels[:,idx_l] - self._means[idx_l])**2))
                            self._stds[idx_q] = np.sqrt(np.mean((channels[:,idx_q] - self._means[idx_q])**2))
            channels -= self._means
            channels /= self._stds
        return channels

    def dotdelay_frames(self, 
                        spatial_gabor_sin, 
                        spatial_gabor_cos,
                        temporal_gabor_sin, 
                        temporal_gabor_cos,
                        masklimit=0.001):
        '''Convolve the motion-energy filter with a stimulus

        Parameters
        ----------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair

        temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
            Temporal gabor quadrature pair

        stimulus : 2D np.array (n, vdim*hdim)
            The movie frames with the spatial dimension collapsed.

        Returns
        -------
        channel_sin, channel_cos : np.ndarray, (n, )
            The filter response to the stimulus at each time point
            The quadrature pair can be combined: (x^2 + y^2)^0.5
        '''

        gabor_sin, gabor_cos = self.dotspatial_frames(spatial_gabor_sin, spatial_gabor_cos, masklimit=masklimit)
        gabor_prod = np.c_[gabor_sin, gabor_cos]


        temporal_gabors = np.asarray([temporal_gabor_sin,
                                      temporal_gabor_cos])

        # dot the product with the temporal gabors
        outs = np.dot(gabor_prod[:, [0]], temporal_gabors[[1]]) + np.dot(gabor_prod[:, [1]], temporal_gabors[[0]])
        outc = np.dot(-gabor_prod[:, [0]], temporal_gabors[[0]]) + np.dot(gabor_prod[:, [1]], temporal_gabors[[1]])

        # sum across delays
        nouts = np.zeros_like(outs)
        noutc = np.zeros_like(outc)
        tdxc = int(np.ceil(outs.shape[1]/2.0))
        delays = np.arange(outs.shape[1])-tdxc +1
        for ddx, num in enumerate(delays):
            if num == 0:
                nouts[:, ddx] = outs[:,ddx]
                noutc[:, ddx] = outc[:,ddx]
            elif num > 0:
                nouts[num:, ddx] = outs[:-num,ddx]
                noutc[num:, ddx] = outc[:-num,ddx]
            elif num < 0:
                nouts[:num, ddx] = outs[abs(num):,ddx]
                noutc[:num, ddx] = outc[abs(num):,ddx]

        channel_sin = nouts.sum(-1)
        channel_cos = noutc.sum(-1)
        return channel_sin, channel_cos


    def dotspatial_frames(self, 
                          spatial_gabor_sin,
                          spatial_gabor_cos,
                          masklimit=0.001):

        '''Dot the spatial gabor filters filter with the stimuli

        Parameters
        ----------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair
        stimuli : 2D np.array (n, vdim*hdim)
            The movie frames with the spatial dimension collapsed.
        masklimit : float-like
            Threshold to find the non-zero filter region

        Returns
        -------
        channel_sin, channel_cos : np.ndarray, (n, )
            The filter response to each stimulus
            The quadrature pair can be combined: (x^2 + y^2)^0.5
        '''
        gabors = np.asarray([spatial_gabor_sin.ravel(),
                             spatial_gabor_cos.ravel()])

        # dot the gabors with the stimuli
        mask = np.abs(gabors).sum(0) > masklimit
        gabor_prod = np.dot(gabors[:,mask].squeeze(), self.stimulus.T[mask].squeeze()).T
        gabor_sin, gabor_cos = gabor_prod[:,0], gabor_prod[:,1]
        return gabor_sin, gabor_cos

    def make_3d_gaussian(self, 
                      xyt,
                      centerx,
                      centery,
                      direction,
                      spatial_freq,
                      spatial_env,
                      temporal_freq,
                      temporal_env,
                      aspect_ratio,
                      ):

        '''Make a motion-energy filter.

        A motion-energy filter is a 3D gabor with
        two spatial and one temporal dimension.
        Each dimension is defined by two sine waves which
        differ in phase by 90 degrees. The sine waves are
        then multiplied by a gaussian.

        Parameters
        ----------
        xyt : array-like, (hdim, vdim, tdim)
            Defines the 3D field-of-view of the filter
            `hdim` : horizontal dimension size
            `vdim` : vertical dimension size
            `tdim` : temporal dimension size
        centerx, centery : float
            Horizontal and vertical position in space, respectively.
            The image center is (0.5,0.5) for square aspect ratios
        direction : float, degrees
            Direction of spatial motion
        spatial_freq : float
            Spatial frequency
        spatial_env : float
            Spatial envelope (s.d. of the gaussian)
        temporal_freq : float
            Temporal frequency
        temporal_env : float
            Temporal envelope (s.d. of gaussian)
        aspect_ratio : float-like,
            Useful for preserving the spatial gabors circular even
            when images have non-square aspect ratios. For example,
            a 16:9 image would have `aspect_ratio`=16/9.

        Returns
        -------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair. `spatial_gabor_cos` has
            a 90 degree phase offset relative to `spatial_gabor_sin`

        temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
            Temporal gabor quadrature pair. `temporal_gabor_cos` has
            a 90 degree phase offset relative to `temporal_gabor_sin`

        Notes
        -----
        Same method as Nishimoto, et al., 2011.
        '''

        szx, szy, szt = np.asarray(xyt).astype(np.int)

        dx = np.linspace(0,aspect_ratio,szx, endpoint=True)
        dy = np.linspace(0,1,szy, endpoint=True)
        dt = np.linspace(0,1,szt, endpoint=False)
        mdt = np.mean(dt)
        ixs, iys = np.meshgrid(dx,dy)

        fx = -spatial_freq*np.cos(direction/180.*np.pi)*2*np.pi
        fy = spatial_freq*np.sin(direction/180.*np.pi)*2*np.pi
        ft = np.real(temporal_freq)*2*np.pi

        # spatial filters
        spatial_gaussian = np.exp(-((ixs - centerx)**2 + (iys - centery)**2)/(2*spatial_env**2))

        # spatial_grating_sin = np.sin((ixs - centerx)*fx + (iys - centery)*fy)
        # spatial_grating_cos = np.cos((ixs - centerx)*fx + (iys - centery)*fy)

        spatial_gabor_sin = spatial_gaussian # * spatial_grating_sin
        spatial_gabor_cos = spatial_gaussian # * spatial_grating_cos

        ##############################
        temporal_gaussian = np.exp(-(dt - mdt)**2/(2*temporal_env**2))
        # temporal_grating_sin = np.sin((dt - 0.5)*ft)
        # temporal_grating_cos = np.cos((dt - 0.5)*ft)

        temporal_gabor_sin = temporal_gaussian # *temporal_grating_sin
        temporal_gabor_cos = temporal_gaussian # *temporal_grating_cos

        return spatial_gabor_sin, spatial_gabor_cos, temporal_gabor_sin, temporal_gabor_cos


    def make_3d_gabor(self, 
                      xyt,
                      centerx,
                      centery,
                      direction,
                      spatial_freq,
                      spatial_env,
                      temporal_freq,
                      temporal_env,
                      aspect_ratio,
                      ):

        '''Make a motion-energy filter.

        A motion-energy filter is a 3D gabor with
        two spatial and one temporal dimension.
        Each dimension is defined by two sine waves which
        differ in phase by 90 degrees. The sine waves are
        then multiplied by a gaussian.

        Parameters
        ----------
        xyt : array-like, (hdim, vdim, tdim)
            Defines the 3D field-of-view of the filter
            `hdim` : horizontal dimension size
            `vdim` : vertical dimension size
            `tdim` : temporal dimension size
        centerx, centery : float
            Horizontal and vertical position in space, respectively.
            The image center is (0.5,0.5) for square aspect ratios
        direction : float, degrees
            Direction of spatial motion
        spatial_freq : float
            Spatial frequency
        spatial_env : float
            Spatial envelope (s.d. of the gaussian)
        temporal_freq : float
            Temporal frequency
        temporal_env : float
            Temporal envelope (s.d. of gaussian)
        aspect_ratio : float-like,
            Useful for preserving the spatial gabors circular even
            when images have non-square aspect ratios. For example,
            a 16:9 image would have `aspect_ratio`=16/9.

        Returns
        -------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair. `spatial_gabor_cos` has
            a 90 degree phase offset relative to `spatial_gabor_sin`

        temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
            Temporal gabor quadrature pair. `temporal_gabor_cos` has
            a 90 degree phase offset relative to `temporal_gabor_sin`

        Notes
        -----
        Same method as Nishimoto, et al., 2011.
        '''

        szx, szy, szt = np.asarray(xyt).astype(np.int)

        dx = np.linspace(0,aspect_ratio,szx, endpoint=True)
        dy = np.linspace(0,1,szy, endpoint=True)
        dt = np.linspace(0,1,szt, endpoint=False)
        mdt = np.mean(dt)
        ixs, iys = np.meshgrid(dx,dy)

        fx = -spatial_freq*np.cos(direction/180.*np.pi)*2*np.pi
        fy = spatial_freq*np.sin(direction/180.*np.pi)*2*np.pi
        ft = np.real(temporal_freq)*2*np.pi

        # spatial filters
        spatial_gaussian = np.exp(-((ixs - centerx)**2 + (iys - centery)**2)/(2*spatial_env**2))

        spatial_grating_sin = np.sin((ixs - centerx)*fx + (iys - centery)*fy)
        spatial_grating_cos = np.cos((ixs - centerx)*fx + (iys - centery)*fy)

        spatial_gabor_sin = spatial_gaussian * spatial_grating_sin
        spatial_gabor_cos = spatial_gaussian * spatial_grating_cos

        ##############################
        temporal_gaussian = np.exp(-(dt - mdt)**2/(2*temporal_env**2))
        temporal_grating_sin = np.sin((dt - mdt)*ft)
        temporal_grating_cos = np.cos((dt - mdt)*ft)

        temporal_gabor_sin = temporal_gaussian*temporal_grating_sin
        temporal_gabor_cos = temporal_gaussian*temporal_grating_cos

        return spatial_gabor_sin, spatial_gabor_cos, temporal_gabor_sin, temporal_gabor_cos

    def get_all_filters(self):
        simple_cell_filters = []
        simple_cell_idx = []

        complex_cell_filters = []
        complex_cell_idx = []

        for idx, gabor_param in enumerate(self.param_list):
            gabor = self.make_3d_gabor((self.horizontal_dim, self.vertical_dim, self.temporal_window),
                                        *gabor_param[:-3],
                                        aspect_ratio=self.aspect_ratio)
            gabor0, gabor90, tgabor0, tgabor90 = gabor
            phase = gabor_param[7]

            gabor_3d_sin = self.make_spatiotemporal_gabor(-gabor90, gabor0, tgabor0, tgabor90)
            gabor_3d_cos = self.make_spatiotemporal_gabor(gabor0, gabor90, tgabor0, tgabor90)

            if phase == 0:
                simple_cell_filters.append(gabor_3d_sin)
                simple_cell_idx.append(idx)
            elif phase == 1:
                simple_cell_filters.append(gabor_3d_cos)
                simple_cell_idx.append(idx)
            elif phase == 2:
                complex_cell_filters.append(gabor_3d_sin)
                complex_cell_idx.append(idx)
            elif phase == 3:
                complex_cell_filters.append(gabor_3d_cos)
                complex_cell_idx.append(idx)

        self.simple_cell_idx = np.array(simple_cell_idx)
        self.complex_cell_idx = np.array(complex_cell_idx)
        return np.array(simple_cell_filters),  np.array(complex_cell_filters), np.array(simple_cell_idx), np.array(complex_cell_idx)

    def get_all_envelopes(self):
        simple_cell_filters = []
        simple_cell_idx = []

        complex_cell_filters = []
        complex_cell_idx = []

        for idx, gabor_param in enumerate(self.param_list):
            gabor = self.make_3d_gaussian((self.horizontal_dim, self.vertical_dim, self.temporal_window),
                                        *gabor_param[:-3],
                                        aspect_ratio=self.aspect_ratio)
            gabor0, gabor90, tgabor0, tgabor90 = gabor
            phase = gabor_param[7]

            gabor_3d_sin = self.make_spatiotemporal_gabor(-gabor90, gabor0, tgabor0, tgabor90)
            # gabor_3d_cos = self.make_spatiotemporal_gabor(gabor0, gabor90, tgabor0, tgabor90)

            if phase == 0:
                simple_cell_filters.append(gabor_3d_sin)
                simple_cell_idx.append(idx)
            elif phase == 1:
                simple_cell_filters.append(gabor_3d_sin)
                simple_cell_idx.append(idx)
            elif phase == 2:
                complex_cell_filters.append(gabor_3d_sin)
                complex_cell_idx.append(idx)
            elif phase == 3:
                complex_cell_filters.append(gabor_3d_sin)
                complex_cell_idx.append(idx)

        self.simple_cell_idx = np.array(simple_cell_idx)
        self.complex_cell_idx = np.array(complex_cell_idx)
        return np.array(simple_cell_filters),  np.array(complex_cell_filters), np.array(simple_cell_idx), np.array(complex_cell_idx)


    def make_spatiotemporal_gabor(self,
                                  spatial_gabor_sin, 
                                  spatial_gabor_cos,
                                  temporal_gabor_sin, 
                                  temporal_gabor_cos):
        '''Make 3D motion-energy filter defined by the spatial and temporal gabors.

        Takes the output of :func:`mk_3d_gabor` and constructs the 3D filter.
        This is useful for visualization.

        Parameters
        ----------
        spatial_gabor_sin, spatial_gabor_cos : np.array, (vdim,hdim)
            Spatial gabor quadrature pair
        temporal_gabor_sin, temporal_gabor_cos : np.array, (tdim)
            Temporal gabor quadrature pair

        Returns
        -------
        motion_energy_filter : np.array, (vdim, hdim, tdim)
            The 3D motion-energy filter

        '''
        a = np.dot(-spatial_gabor_sin.ravel()[...,None], temporal_gabor_sin[...,None].T)
        b = np.dot(spatial_gabor_cos.ravel()[...,None], temporal_gabor_cos[...,None].T)
        x,y = spatial_gabor_sin.shape
        t = temporal_gabor_sin.shape[0]
        return (a+b).reshape(x,y,t).transpose(2,0,1)


    def get_preprocessing_as_keras_model(self, delays):
        from keras.models import Sequential
        from kfs.layers.decode import SpatioTemporalFilter, Rescale
        from keras.layers import Flatten, Dense
        # from keras.layers import ZeroPadding1D

        if type(delays) is int:
            delays = np.arange(delays)

        number_of_features = len(self.param_list)

        simple_filters, complex_filters, _, _ = self.get_all_filters()

        s1, s2, s3, s4 = simple_filters.shape
        c1, c2, c3, c4 = complex_filters.shape
        simple_filters = simple_filters.reshape(s1, s2, s3*s4, 1).transpose(1, 3, 2, 0)
        complex_filters = complex_filters.reshape(c1, c2, c3*c4, 1).transpose(1, 3, 2, 0)

        self.newidx = np.concatenate([self.simple_cell_idx, self.complex_cell_idx])

        filts = np.concatenate([simple_filters, complex_filters], axis=-1)[::-1]

        model = Sequential()
        model.add(SpatioTemporalFilter(nb_simple=s1, nb_complex=c1, filter_delays=s2, input_shape=(len(delays)+self.temporal_window-1, s3*s4)))
        if self.zscore_by_type or self.zscore_each:
            model.add(Rescale(means=self.means[self.newidx], stds=self.stds[self.newidx]))
        
        # model.add(Flatten())

        model.compile(loss='mse', optimizer='sgd')

        model.layers[0].set_weights([filts])

        self.model = model

        return model

    def get_simple_preprocessing_as_keras_model(self, delays):
        from keras.models import Sequential
        from kfs.layers.decode import SpatioTemporalFilterSimple, Rescale
        from keras.layers import Flatten, Dense, Activation
        # from keras.layers import ZeroPadding1D

        if type(delays) is int:
            delays = np.arange(delays)

        number_of_features = len(self.param_list)

        simple_filters, complex_filters, _, _ = self.get_all_filters()

        s1, s2, s3, s4 = simple_filters.shape
        c1, c2, c3, c4 = complex_filters.shape
        simple_filters = simple_filters.reshape(s1, s2, s3*s4, 1).transpose(1, 3, 2, 0)
        complex_filters = complex_filters.reshape(c1, c2, c3*c4, 1).transpose(1, 3, 2, 0)

        self.newidx = self.simple_cell_idx

        filts = simple_filters[::-1]

        model = Sequential()
        model.add(Activation('tanh', input_shape=(len(delays)+self.temporal_window-1, s3*s4)))
        model.add(SpatioTemporalFilterSimple(nb_simple=s1, filter_delays=s2))
        if self.zscore_by_type or self.zscore_each:
            model.add(Rescale(means=self.means[self.newidx], stds=self.stds[self.newidx]))
        
        # model.add(Flatten())

        model.compile(loss='mse', optimizer='sgd')

        model.layers[1].set_weights([filts])

        self.model = model

        return model

    def get_complex_preprocessing_as_keras_model(self, delays):
        from keras.models import Sequential
        from kfs.layers.decode import SpatioTemporalFilterComplex, Rescale
        from keras.layers import Flatten, Dense, Activation
        # from keras.layers import ZeroPadding1D

        if type(delays) is int:
            delays = np.arange(delays)

        number_of_features = len(self.param_list)

        simple_filters, complex_filters, _, _ = self.get_all_filters()

        s1, s2, s3, s4 = simple_filters.shape
        c1, c2, c3, c4 = complex_filters.shape
        simple_filters = simple_filters.reshape(s1, s2, s3*s4, 1).transpose(1, 3, 2, 0)
        complex_filters = complex_filters.reshape(c1, c2, c3*c4, 1).transpose(1, 3, 2, 0)

        self.newidx = self.complex_cell_idx

        filts = complex_filters[::-1]

        model = Sequential()
        model.add(Activation('tanh', input_shape=(len(delays)+self.temporal_window-1, s3*s4)))
        model.add(SpatioTemporalFilterComplex(nb_complex=c1, filter_delays=s2))
        if self.zscore_by_type or self.zscore_each:
            model.add(Rescale(means=self.means[self.newidx], stds=self.stds[self.newidx]))
        
        # model.add(Flatten())

        model.compile(loss='mse', optimizer='sgd')

        model.layers[1].set_weights([filts])

        self.model = model

        return model

    def get_preprocessing_from_keras_model(self, delays=1):
        from kfs.generators import time_delay_generator2

        if type(delays) is int:
            delays = np.arange(delays)

        model = self.get_preprocessing_as_keras_model(delays)
        gen = time_delay_generator2(self.stimulus, None, self.temporal_window, delays, 1, shuffle=False)
        out = model.predict_generator(gen, self.stimulus.shape[0])

        return out

    def make_keras_model(self, weights, bias, delays=1):
        from keras.layers import Flatten, Dense
        model = self.get_preprocessing_as_keras_model(delays)
        model.add(Flatten())
        model.add(Dense(weights.shape[-1]))
        model.compile(loss='mse', optimizer='sgd')

        weights = weights[:,self.newidx]
        weights = weights.reshape((-1, weights.shape[-1]))

        model.layers[-1].set_weights([weights, bias])
        return model

    def make_keras_model_simple(self, weights, bias, delays=1):
        from keras.layers import Flatten, Dense
        model = self.get_simple_preprocessing_as_keras_model(delays)
        model.add(Flatten())
        model.add(Dense(weights.shape[-1]))
        model.compile(loss='mse', optimizer='sgd')

        weights = weights[:,self.simple_cell_idx]
        weights = weights.reshape((-1, weights.shape[-1]))

        model.layers[-1].set_weights([weights, bias])
        return model


    def make_keras_model_complex(self, weights, bias, delays=1):
        from keras.layers import Flatten, Dense
        model = self.get_complex_preprocessing_as_keras_model(delays)
        model.add(Flatten())
        model.add(Dense(weights.shape[-1]))
        model.compile(loss='mse', optimizer='sgd')

        weights = weights[:,self.complex_cell_idx]
        weights = weights.reshape((-1, weights.shape[-1]))

        model.layers[-1].set_weights([weights, bias])
        return model

    def reset_weights(self, model, weights, bias, idx):
        weights = weights[:,idx]
        weights = weights.reshape((-1, weights.shape[-1]))

        model.layers[-1].set_weights([weights, bias])
        return model
