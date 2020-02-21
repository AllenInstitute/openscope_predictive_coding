"""
Created on Sunday July 15 2018

@author: marinag
"""
from __future__ import print_function
import os
import numpy as np
import pandas as pd

from openscope_predictive_coding.ophys.response_analysis import utilities as ut
import openscope_predictive_coding.ophys.response_analysis.response_processing as rp


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

    def __init__(self, dataset, overwrite_analysis_files=False, preload_response_dfs=False, use_events=False):
        self.dataset = dataset
        self.overwrite_analysis_files = overwrite_analysis_files
        self.preload_response_dfs = preload_response_dfs
        self.use_events = use_events
        if self.use_events:
            self.suffix = '_events'
        else:
            self.suffix = ''
        self.sweep_window = [-2, 2]  # time, in seconds, around start time to extract portion of cell trace
        self.response_window_duration = 0.5  # window, in seconds, over which to take the mean for a given stimulus sweep
        self.response_window = [np.abs(self.sweep_window[0]), np.abs(self.sweep_window[0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        self.baseline_window = np.asarray(
            self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]

        self.get_block_df()
        self.get_image_ids()
        self.get_oddball_images()
        self.get_sequence_images()
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

    def add_spontaneous_epochs_to_block_df(self):
        block_df = self.get_block_df()
        block_df.sort_values(axis=0, by='start_frame', inplace=True)
        block_df = block_df.reset_index().drop(columns='index')
        new_block_df = pd.DataFrame(columns=block_df.columns)
        i = 0
        for row in range(len(block_df))[:-1]:
            new_block_df.loc[i] = block_df.loc[row].values
            start_frame = block_df.iloc[row].end_frame
            if row == len(block_df) - 1:
                end_frame = len(dataset.timestamps_stimulus)
            else:
                end_frame = block_df.iloc[row + 1].start_frame
            start_time = dataset.timestamps_stimulus[start_frame]
            end_time = dataset.timestamps_stimulus[end_frame]
            data = ['spontaneous', start_frame, end_frame, start_time, end_time]
            i += 1
            new_block_df.loc[i] = data
            i += 1
        return new_block_df

    def get_sequence_images(self):
        stimulus_table = self.dataset.stimulus_table.copy()
        oddball_block = stimulus_table[stimulus_table.session_block_name == 'oddball']
        sequence_images = list(oddball_block.image_id.values[:4]) #relying on the fact that the first 3 images are always sequence images
        self.sequence_images = sequence_images
        return self.sequence_images

    def get_oddball_images(self):
        stimulus_table = self.dataset.stimulus_table.copy()
        ob = stimulus_table[stimulus_table.session_block_name == 'oddball']
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
        stimulus_table.index.name = 'stimulus_presentations_id'
        block = stimulus_table[stimulus_table.session_block_name == session_block_name]
        sequence_images = self.get_sequence_images()
        # label oddball images
        block['oddball'] = False
        indices = block[block.image_id.isin(sequence_images) == False].index
        block.at[indices, 'oddball'] = True
        if session_block_name == 'oddball':
            # add boolean for sequence start
            block.at[:,'sequence_start'] = False
            indices = block[block.image_id.isin([block.image_id.values[0]]) == True].index
            block.at[indices, 'sequence_start'] = True
            # label all images of a sequence preceeding a violation frame as True
            block['violation_sequence'] = False
            indices = block[block.oddball == True].index
            block.at[indices, 'violation_sequence'] = True
            block.at[indices - 1, 'violation_sequence'] = True
            block.at[indices - 2, 'violation_sequence'] = True
            block.at[indices - 3, 'violation_sequence'] = True
        return block

    def get_stimulus_block(self, session_block_name):
        block = self.create_stimulus_block(session_block_name)
        if self.overwrite_analysis_files:
            print('saving', session_block_name, 'block')
            block.to_hdf(os.path.join(self.dataset.analysis_dir, session_block_name + '_block.h5'), key='df')
        return block

    def generate_response_df(self, session_block_name):
        stimulus_block = self.get_stimulus_block(session_block_name)
        print('generating response dataframe for', session_block_name, self.suffix)
        response_xr = rp.stimulus_response_xr(self, stimulus_block, response_analysis_params=None, use_events=self.use_events)
        response_df = rp.stimulus_response_df(response_xr)
        return response_df

    def get_response_df_path(self, session_block_name):
        path = os.path.join(self.dataset.analysis_dir, session_block_name + '_response_df'+self.suffix+'.h5')
        return path

    def save_response_df(self, response_df, session_block_name):
        print('saving response dataframe for', session_block_name)
        try:
            response_df.to_hdf(self.get_response_df_path(session_block_name), key='df')
        except:
            print('**couldnt save response df for', session_block_name)

    def get_response_df(self, session_block_name):
        if self.overwrite_analysis_files:
            print('overwriting response dataframe for', session_block_name)
            if os.path.exists(self.get_response_df_path(session_block_name)):
                os.remove(self.get_response_df_path(session_block_name))
            response_df = self.generate_response_df(session_block_name)
            self.save_response_df(response_df, session_block_name)
        elif self.preload_response_dfs:
            if os.path.exists(self.get_response_df_path(session_block_name)):
                print('loading response dataframe for', session_block_name)
                response_df = pd.read_hdf(self.get_response_df_path(session_block_name), key='df')
            else:
                print('**couldnt load response dataframe for', session_block_name)
        else:
            response_df = self.generate_response_df(session_block_name)
        stimulus_block = self.get_stimulus_block(session_block_name)
        response_df = response_df.merge(stimulus_block, on='stimulus_presentations_id')
        if session_block_name == 'transition_control':
            response_df['second_in_sequence'] = [
                True if response_df.iloc[row].stimulus_key[1] == response_df.iloc[row].image_id else False
                for row in range(0, len(response_df))]
        return response_df

    def get_response_df_dict(self):
        response_df_dict = {}
        for session_block_name in self.dataset.stimulus_table.session_block_name.unique():
            # if 'movie' in session_block_name:
            #     df = self.get_response_df(session_block_name)
            #     response_df_dict[session_block_name] = df
            # else:
            #     try:
            df = self.get_response_df(session_block_name)
            response_df_dict[session_block_name] = df
                # except:
                #     print('failed to generate response dataframe for', session_block_name)
        self.response_df_dict = response_df_dict
        return self.response_df_dict



