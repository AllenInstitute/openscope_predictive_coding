"""
Created on Wednesday August 22 2018

@author: marinag
"""
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import openscope_predictive_coding.ophys.response_analysis.utilities as ut
import openscope_predictive_coding.ophys.plotting.summary_figures as sf
import seaborn as sns

# formatting
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')


def placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1], wspace=None, hspace=None, sharex=False, sharey=False):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0], dim[1],
                                                  subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]),
                                                  int(100 * xspan[0]):int(100 * xspan[1])], wspace=wspace, hspace=hspace)

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [
        fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx], sharex=share_x_with, sharey=share_y_with)
            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    filename = os.path.join(fig_dir, fig_title)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape', bbox_inches='tight')

def plot_traces_heatmap(dff_traces, ax=None, save_dir=None):
    if ax is None:
        figsize = (20, 8)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=np.percentile(dff_traces, 99))
    ax.set_ylim(0, dff_traces.shape[0])
    ax.set_xlim(0, dff_traces.shape[1])
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    cb = plt.colorbar(cax, pad=0.015)
    cb.set_label('dF/F', labelpad=3)
    if save_dir:
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'traces_heatmap')
    return ax

def plot_mean_image_response_heatmap(analysis, mean_df, title=None, ax=None, save_dir=None):
    df = mean_df.copy()
    images = np.sort(df.image_id.unique())
    images = analysis.get_image_ids()
    cell_list = []
    for image in images:
        tmp = df[(df.image_id == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cell_list = cell_list + cell_ids

    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df.cell_specimen_id == cell) & (df.image_id == image)].mean_response.values[0]
            responses.append(response)
        response_matrix[i, :] = np.asarray(responses)

    if ax is None:
        figsize = (5, 8)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=0.3, robust=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": "mean dF/F"}, ax=ax)

    if title is None:
        title = 'mean response by image'
    ax.set_title(title, va='bottom', ha='center')
    ax.set_xticks(np.arange(0,len(images),1))
    ax.set_xticklabels([int(image) for image in images], rotation=90)
    ax.set_ylabel('cells')
    interval = 10
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval))
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_image_response_heatmap')


def plot_mean_trace_heatmap(mean_df, condition='trial_type', condition_values=['go', 'catch'], ax=None, save_dir=None):
    data = mean_df[mean_df.pref_stim == True].copy()
    vmax = 0.5
    if ax is None:
        figsize = (3 * len(condition_values), 6)
        fig, ax = plt.subplots(1, len(condition_values), figsize=figsize, sharey=True)
        ax = ax.ravel()

    for i, condition_value in enumerate(condition_values):
        im_df = data[(data[condition] == condition_value)]
        if i == 0:
            order = np.argsort(im_df.mean_response.values)[::-1]
            cells = im_df.cell.unique()[order]
        len_trace = len(im_df.mean_trace.values[0])
        response_array = np.empty((len(cells), len_trace))
        for x, cell in enumerate(cells):
            tmp = im_df[im_df.cell == cell]
            if len(tmp) >= 1:
                trace = tmp.mean_trace.values[0]
            else:
                trace = np.empty((len_trace))
                trace[:] = np.nan
            response_array[x, :] = trace

        sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='magma', cbar=False)
        xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=1)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels([int(x) for x in xticklabels])
        ax[i].set_yticks(np.arange(0, response_array.shape[0], 10))
        ax[i].set_yticklabels(np.arange(0, response_array.shape[0], 10))
        ax[i].set_xlabel('time after change (s)', fontsize=16)
        ax[i].set_title(condition_value)
        ax[0].set_ylabel('cells')

    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_trace_heatmap_' + condition)


def get_upper_limit_and_intervals(dff_traces, timestamps_ophys):
    upper = np.round(dff_traces.shape[1], -3) + 1000
    interval = 5 * 60
    frame_interval = np.arange(0, len(dff_traces), interval * 31)
    time_interval = np.uint64(np.round(np.arange(timestamps_ophys[0], timestamps_ophys[-1], interval), 1))
    return upper, time_interval, frame_interval


def plot_run_speed(running_speed, timestamps_stimulus, ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(timestamps_stimulus, running_speed, color='gray')
    if label:
        ax.set_ylabel('run speed (cm/s)')
        ax.set_xlabel('time(s)')
    return ax


def format_table_data(dataset):
    table_data = dataset.metadata.copy()
    table_data = table_data[['donor_id', 'targeted_structure', 'imaging_depth', 'cre_line',
                             'experiment_date', 'session_type', 'ophys_experiment_id']]
    table_data = table_data.transpose()
    return table_data

def plot_mean_trace_with_stimulus_blocks(analysis, ax=None, save=False):
    dataset = analysis.dataset
    block_df = analysis.block_df

    colors = sns.color_palette('deep')
    if ax is None:
        figsize=(20,5)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dataset.timestamps_ophys, np.nanmean(dataset.dff_traces_array,axis=0))
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('dF/F')
    ax.set_title('population average')

    for i, block_name in enumerate(block_df.block_name.values):
        start_time = block_df[block_df.block_name==block_name].start_time.values[0]
        end_time = block_df[block_df.block_name==block_name].end_time.values[0]
        ax.axvspan(start_time, end_time, facecolor=colors[i], edgecolor='none', alpha=0.3, linewidth=0, zorder=1, label=block_name)
    ax.legend(bbox_to_anchor=(1,0.3))
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.8)
        save_figure(fig, figsize, dataset.analysis_dir, 'population_average', str(dataset.experiment_id)+'_cell_average_stimulus_blocks')
    return ax

# def plot_mean_running_trace_with_stimulus_blocks(analysis, ax=None, save=False):
#     dataset = analysis.dataset
#     block_df = analysis.block_df
#
#     colors = sns.color_palette('deep')
#     if ax is None:
#         figsize=(20,5)
#         fig, ax = plt.subplots(figsize=figsize)
#
#     ax.plot(dataset.timestamps_ophys, np.nanmean(dataset.running_speed.,axis=0))
#     ax.set_xlabel('time (seconds)')
#     ax.set_ylabel('dF/F')
#     ax.set_title('population average')
#
#     for i, block_name in enumerate(block_df.block_name.values):
#         start_time = block_df[block_df.block_name==block_name].start_time.values[0]
#         end_time = block_df[block_df.block_name==block_name].end_time.values[0]
#         ax.axvspan(start_time, end_time, facecolor=colors[i], edgecolor='none', alpha=0.3, linewidth=0, zorder=1, label=block_name)
#     ax.legend(bbox_to_anchor=(1,0.3))
#     if save:
#         fig.tight_layout()
#         plt.gcf().subplots_adjust(right=0.8)
#         save_figure(fig, figsize, dataset.analysis_dir, 'population_average', str(dataset.experiment_id)+'_cell_average_stimulus_blocks')
#     return ax


def plot_mean_neuropil_trace_with_stimulus_blocks(analysis, ax=None, save=False):
    dataset = analysis.dataset
    block_df = analysis.block_df

    colors = sns.color_palette('deep')
    if ax is None:
        figsize=(20,5)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dataset.timestamps_ophys, np.nanmean(dataset.neuropil_traces,axis=0))
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('F')
    ax.set_title('neuropil mask population average')

    for i, block_name in enumerate(block_df.block_name.values):
        start_time = block_df[block_df.block_name==block_name].start_time.values[0]
        end_time = block_df[block_df.block_name==block_name].end_time.values[0]
        ax.axvspan(start_time, end_time, facecolor=colors[i], edgecolor='none', alpha=0.3, linewidth=0, zorder=1, label=block_name)
    # ax.legend(bbox_to_anchor=(1,0.3))
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.8)
        sf.save_figure(fig, figsize, dataset.analysis_dir, 'population_average', str(dataset.experiment_id)+'_neuropil_average_stimulus_blocks')
    return ax


def plot_mean_cell_trace_with_oddballs(analysis, ax=None, save=False):
    dataset = analysis.dataset
    block_df = analysis.block_df
    odf = analysis.response_df_dict['oddball']

    colors = sns.color_palette('deep')
    if ax is None:
        figsize=(20,5)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dataset.timestamps_ophys, np.nanmean(dataset.dff_traces_array,axis=0))
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('dF/F')
    ax.set_title('population average with oddball stimuli')
    ax.set_xlim(200, 2400)

    indices = odf[(odf.cell_specimen_id==odf.cell_specimen_id.unique()[0])&(odf.oddball==True)].index
    for i, oddball_index in enumerate(indices):
        start_time = odf.iloc[oddball_index].start_time
        end_time = odf.iloc[oddball_index].end_time +3
        ax.axvspan(start_time, end_time, facecolor=colors[1], edgecolor='none', alpha=0.7, linewidth=0, zorder=1)
    # ax.legend(bbox_to_anchor=(1,0.3))
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.8)
        sf.save_figure(fig, figsize, dataset.analysis_dir, 'population_average', str(dataset.experiment_id)+'_cell_average_oddball_stimuli')
    return ax


def plot_mean_neuropil_trace_with_oddballs(analysis, ax=None, save=False):
    dataset = analysis.dataset
    block_df = analysis.block_df
    odf = analysis.response_df_dict['oddball']

    colors = sns.color_palette('deep')
    if ax is None:
        figsize=(20,5)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dataset.timestamps_ophys, np.nanmean(dataset.neuropil_traces,axis=0))
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('F')
    ax.set_title('neuropil mask population average with oddball stimuli')
    ax.set_xlim(200, 2400)

    indices = odf[(odf.cell_specimen_id==0)&(odf.oddball==True)].index
    for i, oddball_index in enumerate(indices):
        start_time = odf.iloc[oddball_index].start_time
        end_time = odf.iloc[oddball_index].end_time +3
        ax.axvspan(start_time, end_time, facecolor=colors[1], edgecolor='none', alpha=0.7, linewidth=0, zorder=1)
    # ax.legend(bbox_to_anchor=(1,0.3))
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.8)
        sf.save_figure(fig, figsize, dataset.analysis_dir, 'population_average', str(dataset.experiment_id)+'_neuropil_average_oddball_stimuli')
    return ax


def plot_mean_sequence_violation(analysis, ax=None, save=False):
    dataset = analysis.dataset
    try:
        oddball_df = analysis.response_df_dict['oddball']
    except:
        oddball_df = analysis.get_response_df('oddball')

    ophys_frame_rate = dataset.metadata.ophys_frame_rate.values[0]
    if ax is None:
        figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)

    traces = oddball_df[oddball_df.image_id == analysis.get_sequence_images()[3]].dff_trace.values
    ax = sf.plot_mean_trace(traces, ophys_frame_rate, color='b', interval_sec=0.5, ax=ax,
                            legend_label='expected')

    traces = oddball_df[oddball_df.oddball == True].dff_trace.values
    ax = sf.plot_mean_trace(traces, ophys_frame_rate, color='r', interval_sec=0.5, ax=ax,
                            legend_label='unexpected')

    ax.axvspan(len(np.mean(traces)) / 2., len(np.mean(traces)) / 2. + (0.25 * ophys_frame_rate), facecolor='gray',
               edgecolor='none', alpha=0.3,
               linewidth=0, zorder=1)
    ax.set_title('population response')
    ax.legend(loc='upper left')

    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.75)
        save_figure(fig, figsize, dataset.analysis_dir, 'experiment_summary', 'oddball_response')
        plt.close()
    return ax


def plot_mean_first_flash_response_by_image_block(analysis, save_dir=None, ax=None):
    fdf = analysis.flash_response_df.copy()
    fdf.image_block = [int(image_block) for image_block in fdf.image_block.values]
    data = fdf[(fdf.repeat==1)&(fdf.pref_stim==True)]
    mean_response = data.groupby(['cell']).apply(ut.get_mean_sem)
    mean_response = mean_response.unstack()

    cell_order = np.argsort(mean_response.mean_response.values)
    if ax is None:
        figsize = (15,5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data = data, x="image_block", y="mean_response", kind="point", hue='cell', hue_order=cell_order,
                       palette='Blues',ax = ax)
    # ax.legend(bbox_to_anchor=(1,1))
    ax.legend_.remove()
    min = mean_response.mean_response.min()
    max = mean_response.mean_response.max()
    norm = plt.Normalize(min,max)
#     norm = plt.Normalize(0,5)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(mappable=sm, ax=ax, label='mean response across blocks')
    ax.set_title('mean response to first flash of pref stim across image blocks')
    if save_dir:
        fig.tight_layout()
        save_figure(fig,figsize,save_dir,'first_flash_by_image_block',analysis.dataset.analysis_folder)
    return ax


def plot_mean_response_across_image_block_sets(data, analysis_folder, save_dir=None, ax=None):
    order = np.argsort(data[data.image_block==1].early_late_block_ratio.values)
    cell_order = data[data.image_block==1].cell.values[order]
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data = data, x="block_set", y="mean_response", kind="point", palette='RdBu',ax = ax,
                      hue='cell', hue_order=cell_order)
    # ax.legend(bbox_to_anchor=(1,1))
    ax.legend_.remove()
    min = np.amin(data.early_late_block_ratio.unique())
    max = np.amax(data.early_late_block_ratio.unique())
    norm = plt.Normalize(min,max)
#     norm = plt.Normalize(0,5)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(mappable=sm, ax=ax, label='first/last ratio')
    ax.set_title('mean response across image blocks\ncolored by ratio of first to last block')
    fig.tight_layout()
    if save_dir:
        fig.tight_layout()
        save_figure(fig,figsize,save_dir,'first_flash_by_image_block_set',analysis_folder)
    return ax


def plot_experiment_summary_figure(analysis, save_dir=None):
    interval_seconds = 600
    ophys_frame_rate = 31

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.8, 0.95), yspan=(0, .3))
    table_data = format_table_data(analysis.dataset)
    xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    xtable.scale(1.5, 3)
    ax.axis('off')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(0, .27))
    ax.imshow(analysis.dataset.max_projection, cmap='gray', vmin=0, vmax=np.amax(analysis.dataset.max_projection) / 2.)
    ax.set_title(analysis.dataset.experiment_id)
    ax.axis('off')

    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(analysis.dataset.dff_traces_array,
                                                                               analysis.dataset.timestamps_ophys)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, 0.88), yspan=(0, .3))
    ax = plot_traces_heatmap(analysis.dataset.dff_traces_array, ax=ax)
    ax.set_xticks(np.arange(0, upper_limit, interval_seconds * ophys_frame_rate))
    ax.set_xticklabels(np.arange(0, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, 0.78), yspan=(.3, .45))
    ax = plot_mean_trace_with_stimulus_blocks(analysis, ax=ax)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, 0.78), yspan=(.45, .6))
    ax = plot_mean_neuropil_trace_with_stimulus_blocks(analysis, ax=ax)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, 0.78), yspan=(.6, .75))
    ax = plot_mean_cell_trace_with_oddballs(analysis, ax=ax)
    # ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    # ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, .65), yspan=(.78, .98))
    ax = plot_mean_sequence_violation(analysis, ax=ax, save=False)

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.8), yspan=(.26, .41))
    # ax = plot_run_speed(analysis.dataset.running_speed.running_speed, analysis.dataset.timestamps_stimulus, ax=ax,
    #                     label=True)
    # ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    # ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    # ax.set_xlabel('time (seconds)')
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.8), yspan=(.37, .52))
    # ax = plot_hit_false_alarm_rates(analysis.dataset.trials, ax=ax)
    # ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    # ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    # ax.legend(loc='upper right', ncol=2, borderaxespad=0.)
    # ax.set_xlabel('time (seconds)')
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(.25, .8))
    # ax = plot_lick_raster(analysis.dataset.trials, ax=ax, save_dir=None)
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 4), xspan=(.2, .8), yspan=(.5, .8), wspace=0.35)
    # mdf = ut.get_mean_df(analysis.trial_response_df,
    #                      conditions=['cell', 'change_image_name', 'behavioral_response_type'])
    # ax = plot_mean_trace_heatmap(mdf, condition='behavioral_response_type',
    #                              condition_values=['HIT', 'MISS', 'CR', 'FA'], ax=ax, save_dir=None)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.78, 0.97), yspan=(.45, .95))
    mdf = ut.get_mean_df(analysis.response_df_dict['oddball'])
    ax = plot_mean_image_response_heatmap(analysis, mdf, title='mean image response - oddball', ax=ax, save_dir=None)
    fig.tight_layout()

    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', analysis.dataset.analysis_folder)




if __name__ == '__main__':
    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

    # experiment_id = 723037901
    # experiment_id = 712860764
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    # analysis = ResponseAnalysis(dataset)
    # plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    #
    lims_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
                647551128, 647887770, 648647430, 649118720, 649318212, 652844352,
                653053906, 653123781, 639253368, 639438856, 639769395, 639932228,
                661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
                672185644, 672584839, 685744008, 686726085, 695471168, 696136550,
                698244621, 698724265, 700914412, 701325132, 702134928, 702723649,
                692342909, 692841424, 693272975, 693862238, 712178916, 712860764,
                713525580, 714126693, 715161256, 715887497, 716327871, 716600289,
                715228642, 715887471, 716337289, 716602547, 720001924, 720793118,
                723064523, 723750115, 719321260, 719996589, 723748162, 723037901]

    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'

    for lims_id in lims_ids[10:]:
        print(lims_id)
        dataset = VisualBehaviorOphysDataset(lims_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset)
        plot_experiment_summary_figure(analysis, save_dir=cache_dir)
        print('done plotting figures')
