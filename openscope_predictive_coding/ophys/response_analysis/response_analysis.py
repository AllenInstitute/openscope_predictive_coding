"""
Created on Sunday July 15 2018

@author: marinag
"""
from __future__ import print_function
import os
import numpy as np
import pandas as pd

from openscope_predictive_coding.ophys.response_analysis import utilities as ut


class ResponseAnalysis(object):
    """ Contains methods for organizing responses by visual stimulus presentation in a DataFrame called 'response_df'.

    A segment of the dF/F trace for each cell is extracted for each sweep in the stimulus_table in a +/- x seconds window (the 'sweep_window') around the start_time.
    The mean_response for each cell is taken in a 500ms window after the change time (the 'response_window').

    Parameters
    ----------
    dataset: OpenscopePredictiveCodingDataset instance
    overwrite_analysis_files: Boolean, if True will create and overwrite response analysis files.
    This can be used if new functionality is added to the ResponseAnalysis class to modify existing structures or make new ones.
    If False, will load existing analysis files from dataset.analysis_dir, or generate and save them if none exist.
    """

    def __init__(self, dataset, overwrite_analysis_files=False, preload_response_dfs=True):
        self.dataset = dataset
        self.overwrite_analysis_files = overwrite_analysis_files
        self.sweep_window = [-2, 2]  # time, in seconds, around start time to extract portion of cell trace
        self.response_window_duration = 0.5  # window, in seconds, over which to take the mean for a given stimulus sweep
        self.response_window = [np.abs(self.sweep_window[0]), np.abs(self.sweep_window[0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        self.baseline_window = np.asarray(
            self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        # self.stimulus_duration = self.dataset.task_parameters['stimulus_duration'].values[0]
        # self.blank_duration = self.dataset.task_parameters['blank_duration'].values[0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]

        # self.get_response_df()
        self.get_block_df()
        self.get_oddball_block()
        self.get_image_ids()
        if preload_response_dfs:
            self.get_response_df_dict()


    def get_block_df(self):
        gb = self.dataset.stimulus_table.groupby('session_block_name')
        mins = gb.apply(lambda x: x['start_frame'].min())
        maxs = gb.apply(lambda x: x['start_frame'].max())
        block_df = pd.DataFrame()
        block_df['block_name'] = mins.keys()
        block_df['start_frame'] = mins.values
        block_df['end_frame'] = maxs.values
        block_df['start_time'] = [self.dataset.timestamps_stimulus[frame] for frame in block_df.start_frame.values]
        block_df['end_time'] = [self.dataset.timestamps_stimulus[frame] for frame in block_df.end_frame.values]
        block_df.to_hdf(os.path.join(self.dataset.analysis_dir, 'block_df.h5'), key='df')
        self.block_df = block_df
        return self.block_df

    def get_sequence_images(self):
        stimulus_table = self.dataset.stimulus_table.copy()
        oddball_block = stimulus_table[stimulus_table.session_block_name == 'oddball']
        sequence_images = list(oddball_block.image_id.values[:4])
        self.sequence_images = sequence_images
        return self.sequence_images

    def get_oddball_images(self):
        ob = self.oddball_block.copy()
        sequence_images = self.get_sequence_images()
        oddball_images = ob[ob.image_id.isin(sequence_images) == False].image_id.unique()
        self.oddball_images = list(np.sort(oddball_images))
        return self.oddball_images

    def get_image_ids(self):
        """
        :return: returns list of image_ids where the first 4 are the sequence images and the last 10 are the oddball images
        """
        sequence_images = self.get_sequence_images()
        oddball_images = self.get_oddball_images()
        self.image_ids = list(sequence_images) + list(oddball_images)
        return self.image_ids

    def get_stimulus_duration(self, session_block_name):
        """
        This needs to be not hard coded, encode how these numbers were derived from the stim table
        :param session_block_name:
        :return:
        """
        if session_block_name == 'oddball':
            stimulus_duration = 0.23
        elif 'control' in session_block_name:
            stimulus_duration = 0.23
        elif 'movie' in session_block_name:
            stimulus_duration = 30.
        elif 'occlusion' in session_block_name:
            stimulus_duration = 0.5
        return stimulus_duration

    def create_stimulus_block(self, session_block_name):
        stimulus_table = self.dataset.stimulus_table.copy()
        block = stimulus_table[stimulus_table.session_block_name == session_block_name]
        sequence_images = self.get_sequence_images()
        # label oddball images
        block['oddball'] = False
        indices = block[block.image_id.isin(sequence_images) == False].index
        block.loc[indices, 'oddball'] = True
        if session_block_name == 'oddball':
            # add boolean for sequence start
            block['sequence_start'] = False
            indices = block[block.image_id.isin([block.image_id.values[0]]) == True].index
            block.loc[indices, 'sequence_start'] = True
            # label all images of a sequence preceeding a violation frame as True
            block['violation_sequence'] = False
            indices = block[block.oddball == True].index
            block.loc[indices, 'violation_sequence'] = True
            block.loc[indices - 1, 'violation_sequence'] = True
            block.loc[indices - 2, 'violation_sequence'] = True
            block.loc[indices - 3, 'violation_sequence'] = True
        print('saving', session_block_name, 'block')
        block.to_hdf(os.path.join(self.dataset.analysis_dir, session_block_name + '_block.h5'), key='df')
        return block

    def get_stimulus_block(self, session_block_name):
        if (self.overwrite_analysis_files is True) or (
                session_block_name + '_block.h5' not in os.listdir(os.path.join(self.dataset.analysis_dir))):
            print('creating ' + session_block_name + ' block')
            if session_block_name is 'oddball':
                block = self.create_oddball_block()
            else:
                block = self.create_stimulus_block(session_block_name)
        elif (self.overwrite_analysis_files is False) and (
                session_block_name + '_block.h5' in os.listdir(os.path.join(self.dataset.analysis_dir))):
            print('loading ' + session_block_name + ' block')
            block = pd.read_hdf(os.path.join(self.dataset.analysis_dir, session_block_name + '_block.h5'), key='df')
        return block

    def create_oddball_block(self):
        stimulus_table = self.dataset.stimulus_table.copy()
        oddball_block = stimulus_table[stimulus_table.session_block_name == 'oddball']
        sequence_images = self.get_sequence_images()
        # label oddball images
        oddball_block['oddball'] = False
        indices = oddball_block[oddball_block.image_id.isin(sequence_images) == False].index
        oddball_block.loc[indices, 'oddball'] = True
        # add boolean for sequence start
        oddball_block['sequence_start'] = False
        indices = oddball_block[oddball_block.image_id.isin([oddball_block.image_id.values[0]]) == True].index
        oddball_block.loc[indices, 'sequence_start'] = True
        # label all images of a sequence preceeding a violation frame as True
        oddball_block['violation_sequence'] = False
        indices = oddball_block[oddball_block.oddball == True].index
        oddball_block.loc[indices, 'violation_sequence'] = True
        oddball_block.loc[indices - 1, 'violation_sequence'] = True
        oddball_block.loc[indices - 2, 'violation_sequence'] = True
        oddball_block.loc[indices - 3, 'violation_sequence'] = True
        print('saving oddball block')
        oddball_block.to_hdf(os.path.join(self.dataset.analysis_dir, 'oddball_block.h5'), key='df')
        self.oddball_block = oddball_block
        return self.oddball_block

    def get_oddball_block(self):
        file_path = os.path.join(self.dataset.analysis_dir, 'oddball_block.h5')
        if self.overwrite_analysis_files is True:
            print('overwriting oddball block')
            if os.path.exists(file_path):
                os.remove(file_path)
            self.oddball_block = self.create_oddball_block()
        if 'oddball_block.h5' not in os.listdir(os.path.join(self.dataset.analysis_dir)):
            print('creating oddball block')
            self.oddball_block = self.create_oddball_block()
        elif (self.overwrite_analysis_files is False) and (
            'oddball_block.h5' in os.listdir(os.path.join(self.dataset.analysis_dir))):
            # print('loading oddball block')
            self.oddball_sblock = pd.read_hdf(os.path.join(self.dataset.analysis_dir, 'oddball_block.h5'), key='df')
        return self.oddball_block

    def generate_response_df(self, session_block_name):
        stimulus_block = self.get_stimulus_block(session_block_name)
        print('generating response dataframe for', session_block_name)
        stimulus_duration = self.get_stimulus_duration(session_block_name)
        sweep_window = [self.sweep_window[0], self.sweep_window[1] + stimulus_duration]
        if 'movie' in session_block_name:
            stimulus_block = stimulus_block[stimulus_block.data_file_index==0]
            sweep_window = [0, stimulus_duration]
        df_list = []
        for cell_index in self.dataset.cell_indices:
            cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell_index)
            cell_trace = self.dataset.dff_traces[cell_index, :]
            for sweep in stimulus_block.sweep.values:
                start_time = stimulus_block[stimulus_block.sweep == sweep].start_time.values[0]

                trace, timestamps = ut.get_trace_around_timepoint(start_time, cell_trace,
                                                                  self.dataset.timestamps_ophys,
                                                                  sweep_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, self.response_window, self.ophys_frame_rate)
                baseline_response = ut.get_mean_in_window(trace, self.baseline_window, self.ophys_frame_rate)
                p_value = ut.get_p_val(trace, self.response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(trace, self.response_window, self.baseline_window,
                                                           self.ophys_frame_rate)
                df_list.append(
                    [sweep, cell_index, cell_specimen_id, trace, timestamps, mean_response, baseline_response,
                     p_value,
                     sd_over_baseline])  # , running_speed_trace, running_speed_timestamps, mean_running_speed])

        columns = ['sweep', 'cell_index', 'cell_specimen_id', 'trace', 'timestamps', 'mean_response',
                   'baseline_response',
                   'p_value',
                   'sd_over_baseline']  # , 'running_speed_trace', 'running_speed_timestamps', 'mean_running_speed']
        response_df = pd.DataFrame(df_list, columns=columns)
        if session_block_name != 'oddball':
            response_df = response_df.merge(stimulus_block, on='sweep')
        return response_df

    def get_response_df_path(self, session_block_name):
        path = os.path.join(self.dataset.analysis_dir, session_block_name + '_response_df.h5')
        return path

    def save_response_df(self, response_df, session_block_name):
        print('saving response dataframe for', session_block_name)
        response_df.to_hdf(self.get_response_df_path(session_block_name), key='df')

    def get_response_df(self, session_block_name):
        if self.overwrite_analysis_files:
            print('overwriting response dataframe for', session_block_name)
            if os.path.exists(self.get_response_df_path(session_block_name)):
                os.remove(self.get_response_df_path(session_block_name))
            response_df = self.generate_response_df(session_block_name)
            self.save_response_df(response_df, session_block_name)
        else:
            if os.path.exists(self.get_response_df_path(session_block_name)):
                print('loading response dataframe for', session_block_name)
                response_df = pd.read_hdf(self.get_response_df_path(session_block_name), key='df')
            else:
                response_df = self.generate_response_df(session_block_name)
                self.save_response_df(response_df, session_block_name)
        if session_block_name == 'oddball':
            if 'oddball' not in response_df.keys():
                print('merging oddball response with stim block')
                stimulus_block = self.get_stimulus_block(session_block_name)
                response_df = response_df.merge(stimulus_block, on='sweep')
        return response_df

    def get_response_df_dict(self):
        response_df_dict = {}
        for session_block_name in self.dataset.stimulus_table.session_block_name.unique():
            if 'movie' in session_block_name:
                df = self.get_response_df(session_block_name)
                response_df_dict[session_block_name] = df
            else:
                try:
                    df = self.get_response_df(session_block_name)
                    response_df_dict[session_block_name] = df
                except:
                    print('failed to generate response dataframe for', session_block_name)
        self.response_df_dict = response_df_dict
        return self.response_df_dict



