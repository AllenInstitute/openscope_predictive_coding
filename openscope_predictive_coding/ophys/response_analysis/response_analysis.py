"""
Created on Sunday July 15 2018

@author: marinag
"""

import os
import numpy as np
import pandas as pd

from visual_behavior.ophys.response_analysis import utilities as ut


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

    def __init__(self, dataset, overwrite_analysis_files=False):
        self.dataset = dataset
        self.overwrite_analysis_files = overwrite_analysis_files
        self.sweep_window = [-1, 2]  # time, in seconds, around start time to extract portion of cell trace
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

    def get_response_df_path(self):
        path = os.path.join(self.dataset.analysis_dir, 'response_df.h5')
        return path

    def generate_response_df(self):
        print('generating response dataframe')
        # running_speed = self.dataset.running_speed.running_speed.values
        df_list = []
        for cell_index in self.dataset.cell_indices:
            cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell_index)
            cell_trace = self.dataset.dff_traces[cell_index, :]
            for sweep in self.dataset.stimulus_table.sweep.values:
                start_time = self.dataset.stimulus_table[self.dataset.stimulus_table.sweep == sweep].start_time.values[0]

                trace, timestamps = ut.get_trace_around_timepoint(start_time, cell_trace,
                                                                  self.dataset.timestamps_ophys,
                                                                  self.sweep_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, self.response_window, self.ophys_frame_rate)
                baseline_response = ut.get_mean_in_window(trace, self.baseline_window, self.ophys_frame_rate)
                p_value = ut.get_p_val(trace, self.response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(trace, self.response_window, self.baseline_window,
                                                           self.ophys_frame_rate)

                # # this is redundant because its the same for every cell. do we want to keep this?
                # running_speed_trace, running_speed_timestamps = ut.get_trace_around_timepoint(change_time,
                #                                                                               running_speed,
                #                                                                               self.dataset.timestamps_stimulus,
                #                                                                               self.sweep_window,
                #                                                                               self.stimulus_frame_rate)
                # mean_running_speed = ut.get_mean_in_window(running_speed_trace, self.response_window,
                #                                            self.stimulus_frame_rate)

                df_list.append(
                    [sweep, cell_index, cell_specimen_id, trace, timestamps, mean_response, baseline_response,
                     p_value, sd_over_baseline]) #, running_speed_trace, running_speed_timestamps, mean_running_speed])

        columns = ['sweep', 'cell_index', 'cell_specimen_id', 'trace', 'timestamps', 'mean_response', 'baseline_response',
                   'p_value', 'sd_over_baseline'] #, 'running_speed_trace', 'running_speed_timestamps', 'mean_running_speed']
        response_df = pd.DataFrame(df_list, columns=columns)
        # response_df = response_df.merge(self.dataset.stimulus_table, on='sweep')
        return response_df

    def save_response_df(self, response_df):
        print('saving response dataframe')
        response_df.to_hdf(self.get_response_df_path(), key='df', format='fixed')

    def get_response_df(self):
        if self.overwrite_analysis_files:
            print('overwriting analysis files')
            self.response_df = self.generate_response_df()
            self.save_response_df(self.response_df)
        else:
            if os.path.exists(self.get_response_df_path()):
                print('loading response dataframe')
                self.response_df = pd.read_hdf(self.get_response_df_path(), key='df', format='fixed')
            else:
                self.response_df = self.generate_response_df()
                self.save_response_df(self.response_df)
        return self.response_df

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
        block_df.to_hdf(os.path.join(self.dataset.analysis_dir, 'block_df.h5'), key='df', format='fixed')
        self.block_df = block_df
        return self.block_df

    def create_oddball_block(self):
        stimulus_table = self.dataset.stimulus_table.copy()
        oddball_block = stimulus_table[stimulus_table.session_block_name == 'oddball']
        sequence_images = list(oddball_block.image_id.values[:4])
        # label oddball images
        oddball_block['oddball'] = False
        indices = oddball_block[oddball_block.image_id.isin(sequence_images) == False].index
        for index in indices:
            oddball_block.loc[index, 'oddball'] = True
        # add boolean for sequence start
        oddball_block['sequence_start'] = False
        indices = oddball_block[oddball_block.image_id.isin([oddball_block.image_id.values[0]]) == True].index
        for index in indices:
            oddball_block.loc[index, 'sequence_start'] = True
        # label all images of a sequence preceeding a violation frame as True
        oddball_block['violation_sequence'] = False
        indices = oddball_block[oddball_block.oddball == True].index
        for index in indices:
            oddball_block.loc[index, 'violation_sequence'] = True
            oddball_block.loc[index - 1, 'violation_sequence'] = True
            oddball_block.loc[index - 2, 'violation_sequence'] = True
            oddball_block.loc[index - 3, 'violation_sequence'] = True
        self.oddball_block = oddball_block
        return self.oddball_block

    def get_oddball_block(self):
        if (self.overwrite_analysis_files is True) or ('oddball_block.h5' not in os.path.join(self.dataset.analysis_dir)):
            print('creating oddball block')
            self.oddball_block = self.create_oddball_block()
            print('saving oddball block')
            self.oddball_block.to_hdf(os.path.join(self.dataset.analysis_dir, 'oddball_block.h5'), key='df', format='fixed')
        elif (self.overwrite_analysis_files is False) and ('oddball_block.h5' in os.path.join(self.dataset.analysis_dir)):
            print('loading oddball block')
            self.oddball_block = pd.read_hdf(os.path.join(self.dataset.analysis_dir, 'oddball_block.h5'), key='df', format='fixed')
        return self.oddball_block
