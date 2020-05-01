"""
Created on Tuesday Sept 11 2018

@author: marinag
"""
import os
import h5py
import platform
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class LazyLoadable(object):

    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.name):
            setattr(obj, self.name, self.calculate(obj))
        return getattr(obj, self.name)


class OpenScopePredictiveCodingDataset(object):
    # TODO getter methods no longer need to set attributes directly

    def __init__(self, experiment_id, cache_dir=None, **kwargs):
        """Initialize openscope predictive coding ophys experiment dataset.
            Loads experiment data from cache_dir, including dF/F traces, roi masks, stimulus metadata, running speed, licks, rewards, and metadata.

            All available experiment data is read upon initialization of the object.
            Data can be accessed directly as attributes of the class (ex: dff_traces = dataset.dff_traces) or by using functions (ex: dff_traces = dataset.get_dff_traces()

        Parameters
        ----------
        experiment_id : ophys experiment ID
        cache_dir : directory where data files are located
        """
        self.experiment_id = experiment_id
        self.cache_dir = cache_dir
        self.cache_dir = self.get_cache_dir()

    def get_cache_dir(self):
        if self.cache_dir is None:
            if platform.system() == 'Linux':
                cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
            else:
                cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
            logger.info('using default cache_dir: {}'.format(cache_dir))
        else:
            cache_dir = self.cache_dir
        self.cache_dir = cache_dir
        return self.cache_dir

    def get_analysis_folder(self):
        candidates = [file for file in os.listdir(self.cache_dir) if str(self.experiment_id) in file]
        if len(candidates) == 1:
            self._analysis_folder = candidates[0]
        elif len(candidates) < 1:
            raise OSError('unable to locate analysis folder for experiment {} in {}'.format(self.experiment_id, self.cache_dir))
        elif len(candidates) > 1:
            raise OSError('{} contains multiple possible analysis folders: {}'.format(self.cache_dir, candidates))
        return self._analysis_folder
    analysis_folder = LazyLoadable('_analysis_folder', get_analysis_folder)

    def get_analysis_dir(self):
        self._analysis_dir = os.path.join(self.cache_dir, self.analysis_folder)
        return self._analysis_dir
    analysis_dir = LazyLoadable('_analysis_dir', get_analysis_dir)

    def get_metadata(self):
        metadata = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df')
        metadata['depth'] = ['deep' if depth > 250 else 'superficial' for depth in metadata.imaging_depth.values]
        metadata['location'] = metadata.targeted_structure + '_' + metadata.depth
        self._metadata = metadata
        return self._metadata
    metadata = LazyLoadable('_metadata', get_metadata)

    def get_timestamps(self):
        self._timestamps = pd.read_hdf(os.path.join(self.analysis_dir, 'timestamps.h5'), key='df')
        return self._timestamps
    timestamps = LazyLoadable('_timestamps', get_timestamps)

    def get_timestamps_stimulus(self):
        self._timestamps_stimulus = self.timestamps['stimulus_frames']['timestamps']
        return self._timestamps_stimulus
    timestamps_stimulus = LazyLoadable('_timestamps_stimulus', get_timestamps_stimulus)

    def get_timestamps_ophys(self):
        self._timestamps_ophys = self.timestamps['ophys_frames']['timestamps']
        return self._timestamps_ophys
    timestamps_ophys = LazyLoadable('_timestamps_ophys', get_timestamps_ophys)

    def get_stimulus_table(self):
        import openscope_predictive_coding.ophys.response_analysis.response_processing as rp
        stimulus_table = pd.read_hdf(os.path.join(self.analysis_dir, 'stimulus_table.h5'), key='df')
        stimulus_table.index.name = 'stimulus_presentations_id'
        # Average running speed on each flash
        stimulus_running_speed = stimulus_table.apply(
            lambda row: rp.trace_average(
                self.running_speed['speed'].values,
                self.running_speed['time'].values,
                row["start_time"],
                row["start_time"] + 0.25, ), axis=1, )
        stimulus_table["mean_running_speed"] = stimulus_running_speed
        self._stimulus_table = stimulus_table
        return self._stimulus_table
    stimulus_table = LazyLoadable('_stimulus_table', get_stimulus_table)

    def get_stimulus_name(self):
        stimulus_table = self.get_stimulus_table()
        self._stimulus_name = stimulus_table.stimulus_key.values[0]
        return self.stimulus_name
    stimulus_name = LazyLoadable('_stimulus_name', get_stimulus_name)


    def get_stimulus_block_table(self):
        gb = self.stimulus_table.groupby('session_block_name')
        mins = gb.apply(lambda x: x['start_frame'].min())
        maxs = gb.apply(lambda x: x['start_frame'].max())
        block_df = pd.DataFrame()
        block_df['block_name'] = mins.keys()
        block_df['start_frame'] = mins.values
        block_df['end_frame'] = maxs.values
        block_df['start_time'] = [self.timestamps_stimulus[frame] for frame in block_df.start_frame.values]
        block_df['end_time'] = [self.timestamps_stimulus[frame] for frame in block_df.end_frame.values]
        block_df.to_hdf(os.path.join(self.analysis_dir, 'block_df.h5'), key='df')
        self._stimulus_block_table = block_df
        return self._stimulus_block_table
    stimulus_block_table = LazyLoadable('_stimulus_block_table', get_stimulus_block_table)


    # def get_stimulus_template(self):
    #     with h5py.File(os.path.join(self.analysis_dir, 'stimulus_template.h5'), 'r') as stimulus_template_file:
    #         self._stimulus_template = np.asarray(stimulus_template_file['data'])
    #     return self._stimulus_template
    # stimulus_template = LazyLoadable('_stimulus_template', get_stimulus_template)

    # def get_stimulus_metadata(self):
    #     self._stimulus_metadata = pd.read_hdf(
    #         os.path.join(self.analysis_dir, 'stimulus_metadata.h5'),
    #         key='df' )
    #     self._stimulus_metadata = self._stimulus_metadata.drop(columns='image_category')
    #     return self._stimulus_metadata
    # stimulus_metadata = LazyLoadable('_stimulus_metadata', get_stimulus_metadata)
    #
    def get_running_speed(self):
        self._running_speed = pd.read_hdf(os.path.join(self.analysis_dir, 'running_speed.h5'), key='df')
        return self._running_speed
    running_speed = LazyLoadable('_running_speed', get_running_speed)


    def get_dff_traces(self):
        with h5py.File(os.path.join(self.analysis_dir, 'dff_traces.h5'), 'r') as dff_traces_file:
            dff_traces = pd.DataFrame(columns=['dff'], index = dff_traces_file.keys())
            for key in dff_traces_file.keys():
                dff_traces.at[key, 'dff'] = np.asarray(dff_traces_file[key])
            dff_traces.index = [int(cell_specimen_id) for cell_specimen_id in dff_traces.index.values]
            dff_traces.index.name = 'cell_specimen_id'
        self._dff_traces = dff_traces
        return self._dff_traces
    dff_traces = LazyLoadable('_dff_traces', get_dff_traces)

    def get_dff_traces_array(self):
        with h5py.File(os.path.join(self.analysis_dir, 'dff_traces.h5'), 'r') as dff_traces_file:
            dff_traces = []
            for key in dff_traces_file.keys():
                dff_traces.append(np.asarray(dff_traces_file[key]))
        self._dff_traces_array = np.asarray(dff_traces)
        return self._dff_traces_array
    dff_traces_array = LazyLoadable('_dff_traces_array', get_dff_traces_array)

    def get_corrected_fluorescence_traces(self):
        data = h5py.File(os.path.join(self.analysis_dir, 'corrected_fluorescence_traces.h5'), 'r')
        corrected_fluorescence_traces = []
        for key in data.keys():
            corrected_fluorescence_traces.append(np.asarray(data[key]))
        self._corrected_fluorescence_traces = np.asarray(corrected_fluorescence_traces)
        return self._corrected_fluorescence_traces
    corrected_fluorescence_traces = LazyLoadable('_corrected_fluorescence_traces', get_corrected_fluorescence_traces)

    def get_neuropil_traces(self):
        with h5py.File(os.path.join(self.analysis_dir, 'neuropil_traces.h5'), 'r') as neuropil_traces_file:
            neuropil_traces = []
            for key in neuropil_traces_file.keys():
                neuropil_traces.append(np.asarray(neuropil_traces_file[key]))
        self._neuropil_traces = np.asarray(neuropil_traces)
        return self._neuropil_traces
    neuropil_traces = LazyLoadable('_neuropilf_traces', get_neuropil_traces)

    def get_events_array(self):
        events_folder = os.path.join(self.cache_dir, 'events')
        if os.path.exists(events_folder):
            events_file = [file for file in os.listdir(events_folder) if str(self.experiment_id) + '_events.npz' in file]
            if len(events_file) > 0:
                print('getting L0 events')
                f = np.load(os.path.join(events_folder, events_file[0]))
                events = np.asarray(f['ev'])
                ## put smoothing here? ##
                f.close()
                if events.shape[1] > self.timestamps_ophys.shape[0]:
                    difference = self.timestamps_ophys.shape[0] - events.shape[1]
                    print('length of ophys timestamps <  length of events by', str(difference),
                                'frames , truncating events')
                    events = events[:, :self.timestamps_ophys.shape[0]]
            else:
                print('no events for this experiment')
                events = None
        else:
            print('no events for this experiment')
            events = None
        self._events_array = events
        return self._events_array

    events_array = LazyLoadable('_events_array', get_events_array)

    def get_events(self):
        self._events = pd.DataFrame({'events': [x for x in self.events_array]}, index=pd.Index(self.cell_specimen_ids, name='cell_specimen_id'))
        return self._events
    events = LazyLoadable('_events', get_events)


    def get_roi_metrics(self):
        self._roi_metrics = pd.read_hdf(os.path.join(self.analysis_dir, 'roi_metrics.h5'), key='df')
        return self._roi_metrics
    roi_metrics = LazyLoadable('_roi_metrics', get_roi_metrics)

    def get_roi_mask_dict(self):
        f = h5py.File(os.path.join(self.analysis_dir, 'roi_masks.h5'), 'r')
        roi_mask_dict = {}
        for key in f.keys():
            roi_mask_dict[key] = np.asarray(f[key])
        f.close()
        self._roi_mask_dict = roi_mask_dict
        return self._roi_mask_dict
    roi_mask_dict = LazyLoadable('_roi_mask_dict', get_roi_mask_dict)

    def get_roi_mask_array(self):
        w, h = self.roi_mask_dict[list(self.roi_mask_dict.keys())[0]].shape
        roi_mask_array = np.empty((len(list(self.roi_mask_dict.keys())), w, h))
        for cell_specimen_id in list(self.roi_mask_dict.keys()):
            cell_index = self.get_cell_index_for_cell_specimen_id(int(cell_specimen_id))
            roi_mask_array[cell_index] = self.roi_mask_dict[cell_specimen_id]
        self._roi_mask_array = roi_mask_array
        return self._roi_mask_array
    roi_mask_array = LazyLoadable('_roi_mask_array', get_roi_mask_array)

    def get_max_projection(self):
        with h5py.File(os.path.join(self.analysis_dir, 'max_projection.h5'), 'r') as max_projection_file:
            self._max_projection = np.asarray(max_projection_file['data'])
        return self._max_projection
    max_projection = LazyLoadable('_max_projection', get_max_projection)

    def get_average_image(self):
        with h5py.File(os.path.join(self.analysis_dir, 'average_image.h5'), 'r') as average_image_file:
            self._average_image = np.asarray(average_image_file['data'])
        return self._average_image
    average_image = LazyLoadable('_average_image', get_average_image)

    def get_red_channel_image(self):
        import tifffile
        # from PIL import Image
        red_image_file = [file for file in os.listdir(self.analysis_dir) if 'red' in file and '.tif' in file]
        if len(red_image_file) > 0:
            red_image_file_path = os.path.join(self.analysis_dir, red_image_file[0])
            self._red_channel_image = tifffile.imread(red_image_file_path)
            # self._red_channel_image = Image.open(red_image_file_path)
        else:
            print('no red channel image for', self.experiment_id)
            self._red_channel_image = None
        return self._red_channel_image
    red_channel_image = LazyLoadable('_red_channel_image', get_red_channel_image)

    def get_motion_correction(self):
        self._motion_correction = pd.read_hdf(
            os.path.join(self.analysis_dir, 'motion_correction.h5'),
            key='df')
        return self._motion_correction
    motion_correction = LazyLoadable('_motion_correction', get_motion_correction)

    def get_cell_specimen_ids(self):
        self._cell_specimen_ids = np.sort(self.roi_metrics.cell_specimen_id.values)
        return self._cell_specimen_ids
    cell_specimen_ids = LazyLoadable('_cell_specimen_ids', get_cell_specimen_ids)

    def get_cell_indices(self):
        self._cell_indices = np.sort(self.roi_metrics.cell_index.values)
        return self._cell_indices
    cell_indices = LazyLoadable('_cell_indices', get_cell_indices)

    def get_cell_specimen_id_for_cell_index(self, cell_index):
        return self.cell_specimen_ids[cell_index]

    def get_cell_index_for_cell_specimen_id(self, cell_specimen_id):
        return np.where(self.cell_specimen_ids == cell_specimen_id)[0][0]

    def get_stimulus_template(self):
        import openscope_predictive_coding.ophys.dataset.stimulus_processing as sp
        stimulus_table = self.stimulus_table[self.stimulus_table.session_block_name == 'randomized_control_pre']
        stimulus_template = {}
        for image_id in stimulus_table.image_id.unique():
            stimulus_template[int(image_id)] = sp.get_image_array(image_id, self.cache_dir)
        self._stimulus_template = stimulus_template
        return self._stimulus_template
    stimulus_template = LazyLoadable('_stimulus_template', get_stimulus_template)


    @classmethod
    def construct_and_load(cls, experiment_id, cache_dir=None, **kwargs):
        ''' Instantiate a VisualBehaviorOphysDataset and load its data

        Parameters
        ----------
        experiment_id : int
            identifier for this experiment
        cache_dir : str
            filesystem path to directory containing this experiment's

        '''

        obj = cls(experiment_id, cache_dir=cache_dir, **kwargs)

        obj.get_analysis_dir()
        obj.get_metadata()
        obj.get_timestamps()
        obj.get_timestamps_ophys()
        obj.get_timestamps_stimulus()
        obj.get_stimulus_table()
        obj.get_stimulus_name()
        obj.get_stimulus_block_table()
        # obj.get_stimulus_template()
        # obj.get_stimulus_metadata()
        obj.get_running_speed()
        obj.get_dff_traces()
        obj.get_dff_traces_array()
        obj.get_events_array()
        obj.get_events()
        obj.get_corrected_fluorescence_traces()
        obj.get_neuropil_traces()
        obj.get_roi_metrics()
        obj.get_roi_mask_dict()
        obj.get_roi_mask_array()
        obj.get_cell_specimen_ids()
        obj.get_cell_indices()
        obj.get_max_projection()
        obj.get_average_image()
        obj.get_red_channel_image()
        obj.get_motion_correction()

        return obj
