## @nickponvert

import numpy as np
import pandas as pd
import pickle
import os


def load_pickle(pstream):
    return pickle.load(pstream, encoding="bytes")

def add_run_speed_to_trials(running_speed_df, trials):
    trial_running_speed = trials.apply(lambda row: trace_average(
        running_speed_df['running_speed'].values,
        running_speed_df['time'].values,
        row["change_time"],
        row["change_time"] + 0.25, ), axis=1, )
    trials["mean_running_speed"] = trial_running_speed
    return trials


def get_stimulus_presentations(data, stimulus_timestamps):
    stimulus_table = get_visual_stimuli_df(data, stimulus_timestamps)
    # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(
        columns={'frame': 'start_frame', 'time': 'start_time', 'flash_number': 'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[start_frame] for start_frame in stimulus_table.start_frame.values]
    end_time = []
    for end_frame in stimulus_table.end_frame.values:
        if not np.isnan(end_frame):
            end_time.append(stimulus_timestamps[int(end_frame)])
        else:
            end_time.append(float('nan'))

    stimulus_table.insert(loc=4, column='stop_time', value=end_time)
    stimulus_table.set_index('stimulus_presentations_id', inplace=True)
    stimulus_table = stimulus_table[sorted(stimulus_table.columns)]
    return stimulus_table


def get_images_dict(pkl):
    # Sometimes the source is a zipped pickle:
    metadata = {'image_set': pkl["items"]["behavior"]["stimuli"]["images"]["image_path"]}

    # Get image file name; these are encoded case-insensitive in the pickle file :/
    filename = convert_filepath_caseinsensitive(metadata['image_set'])

    image_set = load_pickle(open(filename, 'rb'))
    images = []
    images_meta = []

    ii = 0
    for cat, cat_images in image_set.items():
        for img_name, img in cat_images.items():
            meta = dict(
                image_category=cat.decode("utf-8"),
                image_name=img_name.decode("utf-8"),
                image_index=ii,
            )

            images.append(img)
            images_meta.append(meta)

            ii += 1

    images_dict = dict(
        metadata=metadata,
        images=images,
        image_attributes=images_meta,
    )

    return images_dict


def get_stimulus_templates(pkl):
    images = get_images_dict(pkl)
    image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
    return {IMAGE_SETS_REV[image_set_filename]: np.array(images['images'])}


def get_stimulus_metadata(pkl):
    images = get_images_dict(pkl)
    stimulus_index_df = pd.DataFrame(images['image_attributes'])
    image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
    stimulus_index_df['image_set'] = IMAGE_SETS_REV[image_set_filename]

    # Add an entry for omitted stimuli
    omitted_df = pd.DataFrame({'image_category': ['omitted'],
                               'image_name': ['omitted'],
                               'image_set': ['omitted'],
                               'image_index': [stimulus_index_df['image_index'].max() + 1]})
    stimulus_index_df = stimulus_index_df.append(omitted_df, ignore_index=True, sort=False)
    stimulus_index_df.set_index(['image_index'], inplace=True, drop=True)
    return stimulus_index_df


def _get_stimulus_epoch(set_log, current_set_index, start_frame, n_frames):
    try:
        next_set_event = set_log[current_set_index + 1]  # attr_name, attr_value, time, frame
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames,)

    return (start_frame, next_set_event[3])  # end frame isnt inclusive


def time_from_last(flash_times, other_times):
    last_other_index = np.searchsorted(a=other_times, v=flash_times) - 1
    time_from_last_other = flash_times - other_times[last_other_index]

    # flashes that happened before the other thing happened should return nan
    time_from_last_other[last_other_index == -1] = np.nan

    return time_from_last_other


def trace_average(values, timestamps, start_time, stop_time):
    values_this_range = values[((timestamps >= start_time) & (timestamps < stop_time))]
    return values_this_range.mean()


def add_prior_image_to_stimulus_presentations(stimulus_presentations):
    prior_image_name = [None]
    prior_image_name = prior_image_name + list(stimulus_presentations.image_name.values[:-1])
    stimulus_presentations['prior_image_name'] = prior_image_name
    return stimulus_presentations


def add_window_running_speed(running_speed, stimulus_presentations, response_params):
    window_running_speed = stimulus_presentations.apply(lambda row: trace_average(
        running_speed['running_speed'].values,
        running_speed['time'].values,
        row["start_time"] + response_params['window_around_timepoint_seconds'][0],
        row["start_time"] + response_params['window_around_timepoint_seconds'][1], ), axis=1, )
    stimulus_presentations["window_running_speed"] = window_running_speed
    return stimulus_presentations


def get_image_array(image_id, cache_dir):
    file_dir = os.path.join(cache_dir, 'stimulus_files', 'individual_image_files')
    image_array = np.load(os.path.join(file_dir, str(int(image_id)) + '.npy'))
    return image_array