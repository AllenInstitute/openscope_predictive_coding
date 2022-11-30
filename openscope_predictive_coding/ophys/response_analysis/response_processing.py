## adapted from @nickponvert
## For calculating trial and flash responses
import sys
import os
import numpy as np
import math
import pandas as pd
from scipy import stats
import itertools
import xarray as xr

OPHYS_FRAME_RATE = 31.


def get_default_stimulus_response_params():
    '''
        Get default parameters for computing stimulus_response_xr
        including the window around each stimulus presentation start_time to take a snippet of the dF/F trace for each cell,
        the duration of time after each start_time to take the mean_response,
        and the duration of time before each start_time to take the baseline_response.
        Args:
            None
        Returns
            (dict) dict of response window params for computing stimulus_response_xr
        '''
    stimulus_response_params = {
        "window_around_timepoint_seconds": [-2, 2],
        "response_window_duration_seconds": 0.5,
        "baseline_window_duration_seconds": 0.5
    }
    return stimulus_response_params


def index_of_nearest_value(sample_times, event_times):
    '''
    The index of the nearest sample time for each event time.
    Args:
        sample_times (np.ndarray of float): sorted 1-d vector of sample timestamps
        event_times (np.ndarray of float): 1-d vector of event timestamps
    Returns
        (np.ndarray of int) nearest sample time index for each event time
    '''
    insertion_ind = np.searchsorted(sample_times, event_times)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = sample_times[insertion_ind] - event_times
    ind_minus_one_diff = np.abs(sample_times[np.clip(insertion_ind - 1, 0, np.inf).astype(int)] - event_times)
    return insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)


def eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset):
    '''
    Extract trace for each cell, for each event-relative window.
    Args:
        dff_traces (np.ndarray): shape (nSamples, nCells) with dff traces for each cell
        event_indices (np.ndarray): 1-d array of shape (nEvents) with closest sample ind for each event
        start_ind_offset (int): Where to start the window relative to each event ind
        end_ind_offset (int): Where to end the window relative to each event ind
    Returns:
        sliced_dataout (np.ndarray): shape (nSamples, nEvents, nCells)
    '''
    all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[:, None]
    sliced_dataout = dff_traces_arr.T[all_inds]
    return sliced_dataout


def slice_inds_and_offsets(ophys_times, event_times, window_around_timepoint_seconds, frame_rate=None):
    '''
    Get nearest indices to event times, plus ind offsets for slicing out a window around the event from the trace.
    Args:
        ophys_times (np.array): timestamps of ophys frames
        event_times (np.array): timestamps of events around which to slice windows
        window_around_timepoint_seconds (list): [start_offset, end_offset] for window
        frame_rate (float): we shouldn't need this. leave none to infer from the ophys timestamps
    '''
    if frame_rate is None:
        frame_rate = 1 / np.diff(ophys_times).mean()
    event_indices = index_of_nearest_value(ophys_times, event_times)
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * frame_rate
    start_ind_offset = int(window_around_timepoint_seconds[0] * frame_rate)
    end_ind_offset = int(start_ind_offset + trace_len)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / frame_rate
    return event_indices, start_ind_offset, end_ind_offset, trace_timebase


def add_spontaneous_epochs_to_block_df(block_df, timestamps_stimulus):
    block_df.sort_values(axis=0, by='start_frame', inplace=True)
    block_df = block_df.reset_index().drop(columns='index')
    new_block_df = pd.DataFrame(columns = block_df.columns)
    i = 0
    for row in range(len(block_df))[:-1]:
        new_block_df.loc[i] = block_df.loc[row].values
        start_frame = block_df.iloc[row].end_frame
        if row == len(block_df)-1:
            end_frame = len(dataset.timestamps_stimulus)
        else:
            end_frame = block_df.iloc[row+1].start_frame
        start_time = timestamps_stimulus[start_frame]
        end_time = timestamps_stimulus[end_frame]
        data = ['spontaneous', start_frame, end_frame, start_time, end_time]
        i+=1
        new_block_df.loc[i] = data
        i+=1
    return new_block_df


def get_spontaneous_frames(analysis):
    '''
        Returns a list of the frames that occur during the before and after spontaneous windows.
    Args:
        analysis (object): ResponseAnalysis class instance
    Returns:
        spontaneous_inds (np.array): indices of ophys frames during the spontaneous period
    '''
    ophys_timestamps = analysis.dataset.timestamps_ophys
    block_df = add_spontaneous_epochs_to_block_df(analysis.dataset.get_stimulus_block_table(), analysis.dataset.timestamps_stimulus)
    spontaneous_blocks = block_df[block_df.block_name == 'spontaneous'].reset_index().drop(columns='index')

    row = 0
    spontaneous_frames = []
    for row in range(len(spontaneous_blocks)):
        start_index = index_of_nearest_value(ophys_timestamps, spontaneous_blocks.iloc[row].start_time)
        end_index = index_of_nearest_value(ophys_timestamps, spontaneous_blocks.iloc[row].end_time)
        block_frames = list(np.arange(start_index, end_index, 1))
        spontaneous_frames = np.hstack((spontaneous_frames, block_frames))

    return spontaneous_frames

def filter_events_array(trace_arr, scale=2):
    from scipy import stats
    filt = stats.halfnorm(loc=0, scale=scale).pdf(np.arange(20))
    filtered_arr = np.empty(trace_arr.shape)
    for ind_cell in range(trace_arr.shape[0]):
        this_trace = trace_arr[ind_cell, :]
        this_trace_filtered = np.convolve(this_trace, filt)[:len(this_trace)]
        filtered_arr[ind_cell, :] = this_trace_filtered
    return filtered_arr

def get_p_value_from_shuffled_spontaneous(analysis,
                                          mean_responses,
                                          dff_traces_arr,
                                          response_window_duration,
                                          ophys_frame_rate=None,
                                          number_of_shuffles=10000):
    '''
    Args:
        analysis (object): ResponseAnalysis class instance
        mean_responses (xarray.DataArray): Mean response values, shape (nConditions, nCells)
        dff_traces_arr (np.array): Dff values, shape (nSamples, nCells)
        response_window_duration (int): Number of frames averaged to produce mean response values
        number_of_shuffles (int): Number of shuffles of spontaneous activity used to produce the p-value
    Returns:
        p_values (xarray.DataArray): p-value for each response mean, shape (nConditions, nCells)
    '''

    spontaneous_frames = get_spontaneous_frames(analysis)
    shuffled_spont_inds = np.random.choice(spontaneous_frames, number_of_shuffles).astype(int)

    if ophys_frame_rate is None:
        ophys_frame_rate = int(1 / np.diff(analysis.dataset.timestamps_ophys).mean())

    trace_len = np.round(response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    spont_traces = eventlocked_traces(dff_traces_arr, shuffled_spont_inds, start_ind_offset, end_ind_offset)
    spont_mean = spont_traces.mean(axis=0)  # Returns (nShuffles, nCells)

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    spont_mean_sorted = np.sort(spont_mean, axis=0)
    response_insertion_ind = np.empty(mean_responses.data.shape)
    for ind_cell in range(mean_responses.data.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(spont_mean_sorted[:, ind_cell],
                                                              mean_responses.data[:, ind_cell])

    proportion_spont_larger_than_sample = 1 - (response_insertion_ind / number_of_shuffles)
    result = xr.DataArray(data=proportion_spont_larger_than_sample,
                          coords=mean_responses.coords)
    return result


def stimulus_response_xr(analysis, stimulus_block, response_analysis_params=None, use_events=False):
    dataset = analysis.dataset
    if response_analysis_params is None:
        response_analysis_params = get_default_stimulus_response_params()

    # dff_traces_arr = dataset.dff_traces_array
    if use_events:
        traces = np.stack(dataset.events['events'].values)
        traces = filter_events_array(traces, scale=2)
    else:
        traces = np.stack(dataset.dff_traces['dff'].values)

    event_times = stimulus_block['start_time'].values
    event_indices = index_of_nearest_value(dataset.timestamps_ophys, event_times)

    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=dataset.timestamps_ophys,
        event_times=event_times,
        window_around_timepoint_seconds=response_analysis_params['window_around_timepoint_seconds']
    )
    sliced_dataout = eventlocked_traces(traces, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "stimulus_presentations_id", "cell_specimen_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "stimulus_presentations_id": stimulus_block.index.values,
            "cell_specimen_id": dataset.cell_specimen_ids
        }
    )

    response_range = [0, response_analysis_params['response_window_duration_seconds']]
    baseline_range = [-1 * response_analysis_params['baseline_window_duration_seconds'], 0]

    mean_response = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*response_range)}].mean(['eventlocked_timestamps'])
    mean_baseline = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*baseline_range)}].mean(['eventlocked_timestamps'])
    max_response = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*response_range)}].max(['eventlocked_timestamps'])
    min_response = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*response_range)}].min(['eventlocked_timestamps'])
    summed_response = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*response_range)}].sum(['eventlocked_timestamps'])
    
    
    divisor = max_response - min_response
    trace = eventlocked_traces_xr.loc[{'eventlocked_timestamps': slice(*response_range)}]
#     normalized_trace = 2*np.divide(trace - min_response, divisor,out = np.zeros((trace.shape),dtype=np.float32),where = divisor!=0) - 1
    
    # import pdb; pdb.set_trace()
    p_values = get_p_value_from_shuffled_spontaneous(analysis,
                                                 mean_response,
                                                 traces,
                                                 response_analysis_params['response_window_duration_seconds'])
        


    result = xr.Dataset({
        'eventlocked_traces': trace,
#         'normalized_trace': normalized_trace,
        'summed_response': summed_response,
        'max_response': max_response,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
        'p_value': p_values
    })

    return result

def stimulus_response_df(stimulus_response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = stimulus_response_xr['eventlocked_traces']
    mean_response = stimulus_response_xr['mean_response']
    summed_response = stimulus_response_xr['summed_response']
    max_response = stimulus_response_xr['max_response']
    mean_baseline = stimulus_response_xr['mean_baseline']
    summed_response = stimulus_response_xr['summed_response']
    p_vals = stimulus_response_xr['p_value']
    stacked_traces = traces.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_sum = summed_response.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_max = max_response.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_pval = p_vals.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)
    
    df = pd.DataFrame({
        'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],
        'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'summed_response': summed_response.data,
        'max_response': stacked_max.data,
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'summed_response': stacked_sum.data,
        'p_value': stacked_pval
    })
    return df


def trace_average(values, timestamps, start_time, stop_time):
    values_this_range = values[((timestamps >= start_time) & (timestamps < stop_time))]
    return values_this_range.mean()


