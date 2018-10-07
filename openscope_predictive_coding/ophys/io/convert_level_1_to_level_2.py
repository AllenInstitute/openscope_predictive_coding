from __future__ import print_function

"""
Created on Tuesday Sept 11 2018

@author: marinag
"""

import os
import h5py
import json
import shutil
import platform
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib

matplotlib.use('Agg')

import matplotlib.image as mpimg
#
# from ...translator import foraging2, foraging
# from ...translator.core import create_extended_dataframe
# from ..sync.process_sync import get_sync_data
# from ..plotting.summary_figures import save_figure, plot_roi_validation
# from .lims_database import LimsDatabase


# relative import doesnt work on cluster
# from visual_behavior.translator import foraging2, foraging
# from visual_behavior.translator.core import create_extended_dataframe
from openscope_predictive_coding.ophys.sync.process_sync import get_sync_data
from openscope_predictive_coding.ophys.plotting.summary_figures import save_figure, plot_roi_validation
from openscope_predictive_coding.ophys.io.lims_database import LimsDatabase


def save_data_as_h5(data, name, analysis_dir):
    f = h5py.File(os.path.join(analysis_dir, name + '.h5'), 'w')
    f.create_dataset('data', data=data)
    f.close()


def save_dataframe_as_h5(df, name, analysis_dir):
    df.to_hdf(os.path.join(analysis_dir, name + '.h5'), key='df', format='fixed')


def get_cache_dir(cache_dir=None):
    if not cache_dir:
        if platform.system() == 'Linux':
            cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
        else:
            cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
        return cache_dir
    else:
        return cache_dir


def get_lims_data(lims_id):
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type', value=lims_data.experiment_name.values[0])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def get_analysis_folder_name(lims_data):
    date = str(lims_data.experiment_date.values[0])[:10].split('-')
    analysis_folder_name = str(lims_data.lims_id.values[0]) + '_' + \
                           str(lims_data.external_specimen_id.values[0]) + '_' + date[0][2:] + date[1] + date[2] + '_' + \
                           lims_data.structure.values[0] + '_' + str(lims_data.depth.values[0]) + '_' + \
                           lims_data.specimen_driver_line.values[0].split('-')[0] + '_' + lims_data.rig.values[0][3:5] + \
                           lims_data.rig.values[0][6] + '_' + lims_data.session_type.values[0]
    return analysis_folder_name


def get_mouse_id(lims_data):
    mouse_id = int(lims_data.external_specimen_id.values[0])
    return mouse_id


def get_experiment_date(lims_data):
    experiment_date = str(lims_data.experiment_date.values[0])[:10].split('-')
    return experiment_date


def get_analysis_dir(lims_data, cache_dir=None, cache_on_lims_data=True):
    cache_dir = get_cache_dir(cache_dir=cache_dir)

    if 'analysis_dir' in lims_data.columns:
        return lims_data['analysis_dir'].values[0]

    analysis_dir = os.path.join(cache_dir, get_analysis_folder_name(lims_data))
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    if cache_on_lims_data:
        lims_data.insert(loc=2, column='analysis_dir', value=analysis_dir)
    return analysis_dir


def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir


def get_ophys_experiment_dir(lims_data):
    lims_id = get_lims_id(lims_data)
    ophys_session_dir = get_ophys_session_dir(lims_data)
    ophys_experiment_dir = os.path.join(ophys_session_dir, 'ophys_experiment_' + str(lims_id))
    return ophys_experiment_dir


def get_demix_dir(lims_data):
    ophys_experiment_dir = get_ophys_experiment_dir(lims_data)
    demix_dir = os.path.join(ophys_experiment_dir, 'demix')
    return demix_dir


def get_processed_dir(lims_data):
    ophys_experiment_dir = get_ophys_experiment_dir(lims_data)
    processed_dir = os.path.join(ophys_experiment_dir, 'processed')
    return processed_dir


def get_segmentation_dir(lims_data):
    processed_dir = get_processed_dir(lims_data)
    segmentation_folder = [file for file in os.listdir(processed_dir) if 'segmentation' in file]
    segmentation_dir = os.path.join(processed_dir, segmentation_folder[0])
    return segmentation_dir


def get_sync_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    analysis_dir = get_analysis_dir(lims_data)
    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file][0]
    sync_path = os.path.join(ophys_session_dir, sync_file)
    if sync_file not in os.listdir(analysis_dir):
        print('moving ', sync_file, ' to analysis dir')  # flake8: noqa: E999
        shutil.copy2(sync_path, os.path.join(analysis_dir, sync_file))
    return sync_path


def get_timestamps(lims_data):
    sync_data = get_sync_data(lims_data)
    timestamps = pd.DataFrame(sync_data)
    return timestamps


def get_timestamps_stimulus(timestamps):
    timestamps_stimulus = timestamps['stimulus_frames']['timestamps']
    return timestamps_stimulus


def get_timestamps_ophys(timestamps):
    timestamps_ophys = timestamps['ophys_frames']['timestamps']
    return timestamps_ophys


def get_metadata(lims_data, timestamps):
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    timestamps_ophys = get_timestamps_ophys(timestamps)
    metadata = OrderedDict()
    metadata['ophys_experiment_id'] = lims_data['experiment_id'].values[0]
    if lims_data.parent_session_id.values[0]:
        metadata['experiment_container_id'] = int(lims_data.parent_session_id.values[0])
    else:
        metadata['experiment_container_id'] = None
    metadata['targeted_structure'] = lims_data.structure.values[0]
    metadata['imaging_depth'] = int(lims_data.depth.values[0])
    metadata['cre_line'] = lims_data['specimen_driver_line'].values[0].split(';')[0]
    if len(lims_data['specimen_driver_line'].values[0].split(';')) > 1:
        metadata['reporter_line'] = lims_data['specimen_driver_line'].values[0].split(';')[1] + ';' + \
                                    lims_data['specimen_reporter_line'].values[0].split('(')[0]
    else:
        metadata['reporter_line'] = lims_data['specimen_reporter_line'].values[0].split('(')[0]
    metadata['full_genotype'] = metadata['cre_line'] + ';' + metadata['reporter_line']
    metadata['session_type'] = 'behavior_session_' + lims_data.session_type.values[0][-1]
    metadata['donor_id'] = int(lims_data.external_specimen_id.values[0])
    metadata['experiment_date'] = str(lims_data.experiment_date.values[0])[:10]
    metadata['donor_id'] = int(lims_data.external_specimen_id.values[0])
    metadata['specimen_id'] = int(lims_data.specimen_id.values[0])
    # metadata['session_name'] = lims_data.session_name.values[0]
    # metadata['session_id'] = int(lims_data.session_id.values[0])
    # metadata['project_id'] = lims_data.project_id.values[0]
    # metadata['rig'] = lims_data.rig.values[0]
    metadata['ophys_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_ophys)), 0)
    metadata['stimulus_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_stimulus)), 0)
    # metadata['eye_tracking_frame_rate'] = np.round(1 / np.mean(np.diff(self.timestamps_eye_tracking)),1)
    metadata = pd.DataFrame(metadata, index=[metadata['ophys_experiment_id']])
    return metadata


def save_metadata(metadata, lims_data):
    save_dataframe_as_h5(metadata, 'metadata', get_analysis_dir(lims_data))


def get_stimulus_pkl_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    pkl_file = [file for file in os.listdir(ophys_session_dir) if 'stim.pkl' in file]
    stimulus_pkl_path = os.path.join(ophys_session_dir, pkl_file[0])
    return stimulus_pkl_path


def get_pkl(lims_data):
    stimulus_pkl_path = get_stimulus_pkl_path(lims_data)
    pkl_file = os.path.basename(stimulus_pkl_path)
    analysis_dir = get_analysis_dir(lims_data)
    if pkl_file not in os.listdir(analysis_dir):
        print('moving ', pkl_file, ' to analysis dir')
        shutil.copy2(stimulus_pkl_path, os.path.join(analysis_dir, pkl_file))
    print('getting stimulus data from pkl')
    pkl = pd.read_pickle(stimulus_pkl_path)
    return pkl

#
# def calc_deriv(x, time):
#     dx = np.diff(x)
#     dt = np.diff(time)
#     dxdt_rt = np.hstack((np.nan, dx / dt))
#     dxdt_lt = np.hstack((dx / dt, np.nan))
#
#     dxdt = np.vstack((dxdt_rt, dxdt_lt))
#
#     dxdt = np.nanmean(dxdt, axis=0)
#
#     return dxdt
#
#
# def rad_to_dist(speed_rad_per_s):
#     wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
#     running_radius = 0.5 * (
#         2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
#     running_speed_cm_per_sec = np.pi * speed_rad_per_s * running_radius / 180.
#     return running_speed_cm_per_sec
#
#
# def load_running_speed(pkl, smooth=False, time=None):
#     if time is None:
#         print('`time` not passed. using vsync from pkl file')
#         time = load_time(data)
#
#     dx_raw = np.array(data['dx'])
#     dx = medfilt(dx_raw, kernel_size=5)  # remove big, single frame spikes in encoder values
#     dx = np.cumsum(dx)  # wheel rotations
#
#     time = time[:len(dx)]
#
#     speed = calc_deriv(dx, time)
#     speed = rad_to_dist(speed)
#
#     if smooth:
#         # running_speed_cm_per_sec = pd.rolling_mean(running_speed_cm_per_sec, window=6)
#         raise NotImplementedError
#
#     # accel = calc_deriv(speed, time)
#     # jerk = calc_deriv(accel, time)
#
#     running_speed = pd.DataFrame({
#         'time': time,
#         'frame': range(len(time)),
#         'speed': speed,
#         'dx': dx_raw,
#         'v_sig': data['vsig'],
#         'v_in': data['vin'],
#         # 'acceleration (cm/s^2)': accel,
#         # 'jerk (cm/s^3)': jerk,
#     })
#     return running_speed

def get_stimulus_table(lims_data, timestamps_stimulus):
    import openscope_predictive_coding.utilities as utilities
    # get pkl file from analysis_dir because opc function requires 'stimB' to be in file name
    stimulus_pkl_path = get_stimulus_pkl_path(lims_data)
    pkl_file = os.path.basename(stimulus_pkl_path)
    analysis_dir = get_analysis_dir(lims_data)
    pkl_path = os.path.join(analysis_dir, pkl_file)

    data = utilities.pickle_file_to_interval_table(pkl_path)
    stimulus_table = data[
        ['start_frame', 'end_frame_inclusive', 'session_block_name', 'image_id', 'repeat', 'fraction_occlusion',
         'duration', 'session_type', 'stimulus_key', 'data_file_index', 'data_file_name', 'frame_list']]
    start_time = [timestamps_stimulus[start_frame] for start_frame in stimulus_table.start_frame.values]
    stimulus_table.insert(loc=2, column='start_time', value=start_time)
    end_time = [timestamps_stimulus[end_frame] for end_frame in stimulus_table.end_frame_inclusive.values]
    stimulus_table.insert(loc=3, column='end_time', value=end_time)
    sweeps = np.arange(0,len(stimulus_table),1)
    stimulus_table.insert(loc=0, column='sweep', value=sweeps)

    stimulus_table = stimulus_table.reset_index()
    # stimulus_table = stimulus_table.drop(columns=['lk'])

    return stimulus_table


def save_stimulus_table(stimulus_table, lims_data):
    save_dataframe_as_h5(stimulus_table, 'stimulus_table', get_analysis_dir(lims_data))


#
# def save_core_data_components(core_data, lims_data, timestamps_stimulus):
#     rewards = core_data['rewards']
#     save_dataframe_as_h5(rewards, 'rewards', get_analysis_dir(lims_data))
#
#     running = core_data['running']
#     running_speed = running.rename(columns={'speed': 'running_speed'})
#     # filter to get rid of encoder spikes
#     # happens in 645086795, 645362806
#     from scipy.signal import medfilt
#     running_speed['running_speed'] = medfilt(running_speed.running_speed.values, kernel_size=5)
#     save_dataframe_as_h5(running_speed, 'running_speed', get_analysis_dir(lims_data))
#
#     licks = core_data['licks']
#     save_dataframe_as_h5(licks, 'licks', get_analysis_dir(lims_data))
#
#     stimulus_table = core_data['visual_stimuli'][:-10]  # ignore last 10 flashes
#     # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
#     stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
#     stimulus_table = stimulus_table.rename(columns={'frame': 'start_frame', 'time': 'start_time'})
#     start_time = [timestamps_stimulus[start_frame] for start_frame in stimulus_table.start_frame.values]
#     stimulus_table.start_time = start_time
#     end_time = [timestamps_stimulus[end_frame] for end_frame in stimulus_table.end_frame.values]
#     stimulus_table.insert(loc=4, column='end_time', value=end_time)
#     save_dataframe_as_h5(stimulus_table, 'stimulus_table', get_analysis_dir(lims_data))
#
#     task_parameters = get_task_parameters(core_data)
#     save_dataframe_as_h5(task_parameters, 'task_parameters', get_analysis_dir(lims_data))
#
#
# def get_visual_stimulus_data(pkl):
#     stimulus_template, stimulus_metadata = foraging.extract_images.get_image_data(pkl['image_dict'])
#     stimulus_metadata = pd.DataFrame(stimulus_metadata)
#     return stimulus_template, stimulus_metadata
#
#
# def save_visual_stimulus_data(stimulus_template, stimulus_metadata, lims_data):
#     save_dataframe_as_h5(stimulus_metadata, 'stimulus_metadata', get_analysis_dir(lims_data))
#     save_data_as_h5(stimulus_template, 'stimulus_template', get_analysis_dir(lims_data))


def parse_mask_string(mask_string):
    # convert ruby json array ouput to python 2D array
    # needed for segmentation output prior to 10/10/17 due to change in how masks were saved
    mask = []
    row_length = -1
    for i in range(1, len(mask_string) - 1):
        c = mask_string[i]
        if c == '{':
            row = []
        elif c == '}':
            mask.append(row)
            if row_length < 1:
                row_length = len(row)
        elif c == 'f':
            row.append(False)
        elif c == 't':
            row.append(True)
    return np.asarray(mask)


def get_input_extract_traces_json(lims_data):
    processed_dir = get_processed_dir(lims_data)
    json_file = [file for file in os.listdir(processed_dir) if 'input_extract_traces.json' in file]
    json_path = os.path.join(processed_dir, json_file[0])
    with open(json_path, 'r') as w:
        jin = json.load(w)
    return jin


def get_roi_locations(lims_data):
    jin = get_input_extract_traces_json(lims_data)
    rois = jin["rois"]
    # get data out of json and into dataframe
    roi_locations_list = []
    for i in range(len(rois)):
        roi = rois[i]
        if roi['mask'][0] == '{':
            mask = parse_mask_string(roi['mask'])
        else:
            mask = roi["mask"]
        roi_locations_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], mask])
    roi_locations = pd.DataFrame(data=roi_locations_list, columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'])
    return roi_locations


def add_cell_specimen_ids_to_roi_metrics(roi_metrics, roi_locations):
    # add roi ids to objectlist
    ids = []
    for row in roi_metrics.index:
        minx = roi_metrics.iloc[row][' minx']
        miny = roi_metrics.iloc[row][' miny']
        id = roi_locations[(roi_locations.x == minx) & (roi_locations.y == miny)].id.values[0]
        ids.append(id)
    roi_metrics['cell_specimen_id'] = ids
    return roi_metrics


def get_roi_metrics(lims_data):
    # objectlist.txt contains metrics associated with segmentation masks
    segmentation_dir = get_segmentation_dir(lims_data)
    roi_metrics = pd.read_csv(os.path.join(segmentation_dir, 'objectlist.txt'))
    # get roi_locations and add unfiltered cell index
    roi_locations = get_roi_locations(lims_data)
    roi_names = np.sort(roi_locations.id.values)
    roi_locations['unfiltered_cell_index'] = [np.where(roi_names == id)[0][0] for id in roi_locations.id.values]
    # add cell ids to roi_metrics from roi_locations
    roi_metrics = add_cell_specimen_ids_to_roi_metrics(roi_metrics, roi_locations)
    # merge roi_metrics and roi_locations
    roi_metrics['id'] = roi_metrics.cell_specimen_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on='id')
    # remove invalid roi_metrics
    roi_metrics = roi_metrics[roi_metrics.valid == True]
    # add filtered cell index
    cell_index = [np.where(np.sort(roi_metrics.cell_specimen_id.values) == id)[0][0] for id in
                  roi_metrics.cell_specimen_id.values]
    roi_metrics['cell_index'] = cell_index
    return roi_metrics


def save_roi_metrics(roi_metrics, lims_data):
    save_dataframe_as_h5(roi_metrics, 'roi_metrics', get_analysis_dir(lims_data))


def get_cell_specimen_ids(roi_metrics):
    cell_specimen_ids = np.sort(roi_metrics.cell_specimen_id.values)
    return cell_specimen_ids


def get_cell_indices(roi_metrics):
    cell_indices = np.sort(roi_metrics.cell_index.values)
    return cell_indices


def get_cell_specimen_id_for_cell_index(cell_index, cell_specimen_ids):
    cell_specimen_id = cell_specimen_ids[cell_index]
    return cell_specimen_id


def get_cell_index_for_cell_specimen_id(cell_specimen_id, cell_specimen_ids):
    cell_index = np.where(cell_specimen_ids == cell_specimen_id)[0][0]
    return cell_index


def get_roi_masks(roi_metrics, lims_data):
    # make roi_dict with ids as keys and roi_mask_array
    jin = get_input_extract_traces_json(lims_data)
    h = jin["image"]["height"]
    w = jin["image"]["width"]
    cell_specimen_ids = get_cell_specimen_ids(roi_metrics)
    roi_masks = {}
    for i, id in enumerate(cell_specimen_ids):
        m = roi_metrics[roi_metrics.id == id].iloc[0]
        mask = np.asarray(m['mask'])
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
        roi_masks[int(id)] = binary_mask
    return roi_masks


def save_roi_masks(roi_masks, lims_data):
    f = h5py.File(os.path.join(get_analysis_dir(lims_data), 'roi_masks.h5'), 'w')
    for id, roi_mask in roi_masks.items():
        f.create_dataset(str(id), data=roi_mask)
    f.close()


def get_dff_traces(roi_metrics, lims_data):
    dff_path = os.path.join(get_ophys_experiment_dir(lims_data), str(get_lims_id(lims_data)) + '_dff.h5')
    g = h5py.File(dff_path)
    dff_traces = np.asarray(g['data'])
    valid_roi_indices = np.sort(roi_metrics.unfiltered_cell_index.values)
    dff_traces = dff_traces[valid_roi_indices]
    print('length of traces:', dff_traces.shape[1])
    print('number of segmented cells:', dff_traces.shape[0])
    return dff_traces


def save_dff_traces(dff_traces, roi_metrics, lims_data):
    traces_path = os.path.join(get_analysis_dir(lims_data), 'dff_traces.h5')
    f = h5py.File(traces_path, 'w')
    for i, index in enumerate(get_cell_specimen_ids(roi_metrics)):
        f.create_dataset(str(index), data=dff_traces[i])
    f.close()


def save_timestamps(timestamps, dff_traces, lims_data):
    # remove spurious frames at end of ophys session - known issue with Scientifica data
    if dff_traces.shape[1] < timestamps['ophys_frames']['timestamps'].shape[0]:
        difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
        print('length of ophys timestamps >  length of traces by', str(difference),
              'frames , truncating ophys timestamps')
        timestamps['ophys_frames']['timestamps'] = timestamps['ophys_frames']['timestamps'][:dff_traces.shape[1]]
    # account for dropped ophys frames - a rare but unfortunate issue
    if dff_traces.shape[1] > timestamps['ophys_frames']['timestamps'].shape[0]:
        difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
        print('length of ophys timestamps <  length of traces by', str(difference),
              'frames , truncating traces')
        dff_traces = dff_traces[:, :timestamps['ophys_frames']['timestamps'].shape[0]]
        roi_metrics = get_roi_metrics(lims_data)
        save_dff_traces(dff_traces, roi_metrics, lims_data)
    # make sure length of timestamps equals length of running traces
    # running_speed = core_data['running'].speed.values
    # if len(running_speed) < timestamps['stimulus_frames']['timestamps'].shape[0]:
    #     timestamps['stimulus_frames']['timestamps'] = timestamps['stimulus_frames']['timestamps'][:len(running_speed)]
    save_dataframe_as_h5(timestamps, 'timestamps', get_analysis_dir(lims_data))


def get_motion_correction(lims_data):
    csv_file = [file for file in os.listdir(get_processed_dir(lims_data)) if file.endswith('.csv')]
    csv_file = os.path.join(get_processed_dir(lims_data), csv_file[0])
    csv = pd.read_csv(csv_file, header=None)
    motion_correction = pd.DataFrame()
    motion_correction['x_corr'] = csv[1].values
    motion_correction['y_corr'] = csv[2].values
    return motion_correction


def save_motion_correction(motion_correction, lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    save_dataframe_as_h5(motion_correction, 'motion_correction', analysis_dir)


def get_max_projection(lims_data):
    # max_projection = mpimg.imread(os.path.join(get_processed_dir(lims_data), 'max_downsample_4Hz_0.png'))
    max_projection = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'maxInt_a13a.png'))
    return max_projection


def save_max_projection(max_projection, lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    save_data_as_h5(max_projection, 'max_projection', analysis_dir)
    mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'max_intensity_projection.png'), arr=max_projection,
                 cmap='gray')


def get_roi_validation(lims_data):
    roi_validation = plot_roi_validation(lims_data)
    return roi_validation


def save_roi_validation(roi_validation, lims_data):
    analysis_dir = get_analysis_dir(lims_data)

    for roi in roi_validation:
        fig = roi['fig']
        index = roi['index']
        id = roi['id']
        cell_index = roi['cell_index']

        save_figure(fig, (20, 10), analysis_dir, 'roi_validation',
                    str(index) + '_' + str(id) + '_' + str(cell_index))


def convert_level_1_to_level_2(lims_id, cache_dir=None):
    print('converting', lims_id)
    lims_data = get_lims_data(lims_id)

    get_analysis_dir(lims_data, cache_on_lims_data=True, cache_dir=cache_dir)

    timestamps = get_timestamps(lims_data)

    metadata = get_metadata(lims_data, timestamps)
    save_metadata(metadata, lims_data)

    pkl = get_pkl(lims_data)
    timestamps_stimulus = get_timestamps_stimulus(timestamps)

    stimulus_table = get_stimulus_table(lims_data, timestamps_stimulus)
    save_stimulus_table(stimulus_table, lims_data)
    # core_data = get_core_data(pkl, timestamps_stimulus)
    # save_core_data_components(core_data, lims_data, timestamps_stimulus)
    #
    # trials = get_trials(core_data)
    # save_trials(trials, lims_data)

    # stimulus_template, stimulus_metadata = get_visual_stimulus_data(pkl)
    # save_visual_stimulus_data(stimulus_template, stimulus_metadata, lims_data)

    roi_metrics = get_roi_metrics(lims_data)
    save_roi_metrics(roi_metrics, lims_data)

    roi_masks = get_roi_masks(roi_metrics, lims_data)
    save_roi_masks(roi_masks, lims_data)

    dff_traces = get_dff_traces(roi_metrics, lims_data)
    save_dff_traces(dff_traces, roi_metrics, lims_data)

    save_timestamps(timestamps, dff_traces, lims_data)

    motion_correction = get_motion_correction(lims_data)
    save_motion_correction(motion_correction, lims_data)

    max_projection = get_max_projection(lims_data)
    save_max_projection(max_projection, lims_data)

    # import matplotlib
    # matplotlib.use('Agg')

    roi_validation = get_roi_validation(lims_data)
    save_roi_validation(roi_validation, lims_data)
    print('done converting')
    #
    # ophys_data = core_data.update(
    #     dict(
    #         lims_data=lims_data,
    #         timestamps=timestamps,
    #         metadata=metadata,
    #         roi_metrics=roi_metrics,
    #         roi_masks=roi_masks,
    #         dff_traces=dff_traces,
    #         motion_correction=motion_correction,
    #         max_projection=max_projection,
    #     )
    # )
    # return ophys_data


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)

    # lims_ids = [746270939, 746271249, 750534428, 752473496, 755645715,
    #             754579284, 755000515, 755646041, 756118440,
    #             746271665, 750845430, 750846019, 752473630,
    #             755645219, 756118288, 758305436, 759037671]
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    # for experiment_id in experiment_ids:
    #     ophys_data = convert_level_1_to_level_2(int(experiment_id), cache_dir=cache_dir)
