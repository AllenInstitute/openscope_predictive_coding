"""
Created on Saturday July 14 2018

@author: marinag
"""

import numpy as np
from scipy import stats
import pandas as pd


def get_nearest_frame(timepoint, timestamps):
    return int(np.nanargmin(abs(timestamps - timepoint)))


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_mean_in_window(trace, window, frame_rate):
    return np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])


def get_sd_in_window(trace, window, frame_rate):
    return np.std(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])


def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)


def get_p_val(trace, response_window, frame_rate):
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p


def ptest(x, num_conditions):
    ptest = len(np.where(x < (0.05 / num_conditions))[0])
    return ptest


def get_mean_sem_trace(group):
    mean_response = np.mean(group['mean_response'])
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    mean_trace = np.mean(group['trace'])
    sem_trace = np.std(group['trace'].values) / np.sqrt(len(group['trace'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace, 'mean_responses': mean_responses})


def annotate_trial_response_df_with_pref_stim(trial_response_df):
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    mean_response = rdf.groupby(['cell_specimen_id', 'image_id']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.cell_specimen_id.values:
        cdf = m[m.cell_specimen_id==cell]
        image_index = np.where(cdf['mean_response'].values == np.max(cdf['mean_response'].values))[0][0]
        pref_image = cdf['mean_response'].index[image_index]
        trials = rdf[(rdf.cell_specimen_id == cell) & (rdf.image_id == pref_image)].index
        for trial in trials:
            rdf.loc[trial, 'pref_stim'] = True
    return rdf


def get_mean_sem(group):
    mean_response = np.nanmean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})


# def annotate_flash_response_df_with_pref_stim(fdf):
#     fdf['pref_stim'] = False
#     mean_response = fdf.groupby(['cell', 'image_name']).apply(get_mean_sem)
#     m = mean_response.unstack()
#     for cell in m.index:
#         image_index = np.where(m.loc[cell]['mean_response'].values == np.max(m.loc[cell]['mean_response'].values))[0][0]
#         pref_image = m.loc[cell]['mean_response'].index[image_index]
#         trials = fdf[(fdf.cell == cell) & (fdf.image_name == pref_image)].index
#         for trial in trials:
#             fdf.loc[trial, 'pref_stim'] = True
#     return fdf


def annotate_mean_df_with_pref_stim(mean_df, flashes=False):
    if flashes:
        image_name = 'image_id'
    else:
        image_name = 'image_id'
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False
    if 'cell_specimen_id' in mdf.keys():
        cell_label = 'cell_specimen_id'
    elif 'cell_index' in mdf.keys():
        cell_label = 'cell_index'

    for cell in mdf[cell_label].unique():
        mc = mdf[(mdf[cell_label] == cell)]
        pref_image = mc[(mc.mean_response == np.max(mc.mean_response.values))][image_name].values[0]
        row = mdf[(mdf[cell_label] == cell) & (mdf[image_name] == pref_image)].index
        mdf.loc[row, 'pref_stim'] = True
    return mdf


def get_fraction_significant_trials(group):
    fraction_significant_trials = len(group[group.p_value < 0.005]) / float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})


def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(group[group.mean_response > 0.1]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


def get_mean_df(response_df, conditions=['cell_specimen_id', 'image_id']):
    rdf = response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'mean_responses']]
    mdf = mdf.reset_index()
    if 'image_id' in mdf.keys():
        mdf = annotate_mean_df_with_pref_stim(mdf)

    fraction_significant_trials = rdf.groupby(conditions).apply(get_fraction_significant_trials)
    fraction_significant_trials = fraction_significant_trials.reset_index()
    mdf['fraction_significant_trials'] = fraction_significant_trials.fraction_significant_trials

    fraction_responsive_trials = rdf.groupby(conditions).apply(get_fraction_responsive_trials)
    fraction_responsive_trials = fraction_responsive_trials.reset_index()
    mdf['fraction_responsive_trials'] = fraction_responsive_trials.fraction_responsive_trials

    return mdf


def add_metadata_to_mean_df(mdf, metadata):
    metadata = metadata.reset_index()
    metadata = metadata.rename(columns={'ophys_experiment_id': 'experiment_id'})
    metadata = metadata.drop(columns=['ophys_frame_rate', 'stimulus_frame_rate', 'index'])
    metadata['experiment_id'] = [int(experiment_id) for experiment_id in metadata.experiment_id]
    # metadata['session_num'] = metadata.session_type.values[0][-1]
    mdf = mdf.merge(metadata, how='outer', on='experiment_id')
    return mdf


def add_repeat_to_stimulus_table(stimulus_table):
    repeat = []
    n = 0
    for i, image in enumerate(stimulus_table.image_name.values):
        if image != stimulus_table.image_name.values[i - 1]:
            n = 1
            repeat.append(n)
        else:
            n += 1
            repeat.append(n)
    stimulus_table['repeat'] = repeat
    stimulus_table['repeat'] = [int(repeat) for repeat in stimulus_table.repeat.values]
    return stimulus_table


def add_repeat_number_to_flash_response_df(flash_response_df, stimulus_table):
    stimulus_table = add_repeat_to_stimulus_table(stimulus_table)
    flash_response_df = flash_response_df.merge(stimulus_table[['flash_number','repeat']],on='flash_number')
    return flash_response_df


def add_image_block_to_stimulus_table(stimulus_table):
    stimulus_table['image_block'] = np.nan
    for image_name in stimulus_table.image_name.unique():
        block = 0
        for index in stimulus_table[stimulus_table.image_name==image_name].index.values:
            if stimulus_table.iloc[index]['repeat'] == 1:
                block +=1
            stimulus_table.loc[index,'image_block'] = int(block)
    stimulus_table['image_block'] = [int(image_block) for image_block in stimulus_table.image_block.values]
    return stimulus_table


def add_image_block_to_flash_response_df(flash_response_df, stimulus_table):
    stimulus_table = add_image_block_to_stimulus_table(stimulus_table)
    flash_response_df = flash_response_df.merge(stimulus_table[['flash_number','image_block']],on='flash_number')
    return flash_response_df


def annotate_flash_response_df_with_block_set(flash_response_df):
    fdf = flash_response_df.copy()
    fdf['block_set'] = np.nan
    block_sets = np.arange(0,np.amax(fdf.image_block.unique()),10)
    for i,block_set in enumerate(block_sets):
        if block_set != np.amax(block_sets):
            indices = fdf[(fdf.image_block>=block_sets[i])&(fdf.image_block<block_sets[i+1])].index.values
        else:
            indices = fdf[(fdf.image_block>=block_sets[i])].index.values
        for index in indices:
            fdf.loc[index,'block_set'] = i
    return fdf

def add_early_late_block_ratio_for_fdf(fdf, repeat=1, pref_stim=True):
    data = fdf[(fdf.repeat==repeat)&(fdf.pref_stim==pref_stim)]

    data['early_late_block_ratio'] = np.nan
    for cell in data.cell.unique():
        first_blocks = data[(data.cell==cell)&(data.block_set.isin([0,1]))].mean_response.mean()
        last_blocks = data[(data.cell==cell)&(data.block_set.isin([2,3]))].mean_response.mean()
        index = (last_blocks-first_blocks)/(last_blocks+first_blocks)
        ratio = first_blocks/last_blocks
        indices = data[data.cell==cell].index
        data.loc[indices,'early_late_block_index'] = index
        data.loc[indices,'early_late_block_ratio'] = ratio
    return data


def add_retrogradely_labeled_column_to_df(df, cache_dir=None):
    """
    takes any dataframe with a column for 'cell_specimen_id' and adds a new column called 'retrogradely_labeled' which is a boolean for whether the cell was tagged or not
    """
    if cache_dir is None:
        # cache_dir = r'C:\Users\marinag\Dropbox\opc_analysis'
        cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    red_label_df = pd.read_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs', 'red_label_df.h5'), key='df')
    red_label_df.reset_index(inplace=True)
    red_label_df['cell_specimen_id'] = [int(cell_specimen_id) for cell_specimen_id in red_label_df.cell_specimen_id.values]
    red_df = red_label_df[['cell_specimen_id', 'retrogradely_labeled']]
    df = pd.merge(df, red_df, left_on='cell_specimen_id', right_on='cell_specimen_id')
    return df


def get_manifest(cache_dir=None):
    """
    Loads experiment manifest file as a dataframe, listing all experiments and associated metadata
    """
    if cache_dir is None:
        # cache_dir = r'C:\Users\marinag\Dropbox\opc_analysis'
        cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    manifest_file = os.path.join(cache_dir, 'opc_production_manifest.xlsx')
    manifest = pd.read_excel(manifest_file)
    return manifest


def add_projection_pathway_to_df(df, cache_dir=None):
    """
    df: dataframe to add projection pathway information to. Dataframe must have a column 'experiment_id'.
    cache_dir: cache directory to load manifest from

    Adds columns called 'injection_area', the brain area where the retrograde tracer was injected,
    and 'projection_pathway', indicating whether retrogradely labeled cells are part of a feed forward ('FF') or feed back ('FB') pathway,
    to experiment manifest table, then merges in with provided dataframe.
    """
    manifest = get_manifest(cache_dir)
    manifest['projection_pathway'] = np.nan
    manifest.at[manifest[manifest.injection_area == 'RSP'].index.values, 'projection_pathway'] = 'FF'
    manifest.at[manifest[manifest.injection_area == 'VISp'].index.values, 'projection_pathway'] = 'FB'
    manifest.at[manifest[(manifest.imaging_area == 'VISpm') & (
    manifest.injection_area == 'RSP')].index.values, 'projection_pathway'] = 'FF'
    manifest.at[manifest[(manifest.imaging_area == 'VISpm') & (
    manifest.injection_area == 'VISp')].index.values, 'projection_pathway'] = 'FB'
    df = df.merge(manifest[['experiment_id', 'injection_area', 'pathway']], on='experiment_id')
    return data


def add_location_to_df(df):
    """
    df: dataframe containing columns 'targeted_structure' and 'imaging_depth', such as the manifest table or multi_session_summary_dfs
    Add useful columns, including 'depth' which translates 'imaging_depth' integer values in um to a string indicating superficial or deep layers,
    and 'location', a string combining the imaged area and the 'depth' string as a way to easily group data by both area and depth for analysis.
    """
    df['area'] = df.targeted_structure.values
    df['depth'] = ['deep' if depth > 250 else 'superficial' for depth in df.imaging_depth.values]
    df['location'] = None
    df['location'] = [df.iloc[row].area+'_'+df.iloc[row].depth for row in range(len(df))]
    return df



