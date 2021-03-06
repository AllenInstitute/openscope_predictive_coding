"""
Created on Sunday July 15 2018

@author: marinag
"""
import os
import h5py
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import openscope_predictive_coding.ophys.response_analysis.utilities as ut

sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')

def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    filename = os.path.join(fig_dir, fig_title)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape')



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


def plot_cell_zoom(roi_masks, max_projection, cell_id, spacex=10, spacey=10, show_mask=False, ax=None):
    m = roi_masks[cell_id]
    (y, x) = np.where(m == 1)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    mask = np.empty(m.shape)
    mask[:] = np.nan
    mask[y, x] = 1
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.amax(max_projection)/2.)
    if show_mask:
        ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)
    ax.set_xlim(xmin - spacex, xmax + spacex)
    ax.set_ylim(ymin - spacey, ymax + spacey)
    ax.set_title('cell ' + str(cell_id))
    ax.grid(False)
    ax.axis('off')
    return ax


def plot_roi_validation(lims_data):
    from ..io import convert_level_1_to_level_2 as convert

    file_path = os.path.join(convert.get_processed_dir(lims_data), 'roi_traces.h5')
    g = h5py.File(file_path)
    roi_traces = np.asarray(g['data'])
    roi_names = np.asarray(g['roi_names'])
    g.close()

    dff_path = os.path.join(convert.get_ophys_experiment_dir(lims_data),
                            str(convert.get_lims_id(lims_data)) + '_dff.h5')
    f = h5py.File(dff_path)
    dff_traces_original = np.asarray(f['data'])
    f.close()

    roi_df = convert.get_roi_locations(lims_data)
    roi_metrics = convert.get_roi_metrics(lims_data)
    roi_masks = convert.get_roi_masks(roi_metrics, lims_data)
    dff_traces = convert.get_dff_traces(roi_metrics, lims_data)
    cell_specimen_ids = convert.get_cell_specimen_ids(roi_metrics)
    max_projection = convert.get_max_projection(lims_data)

    roi_validation = []

    for index, id in enumerate(roi_names):
        fig, ax = plt.subplots(3, 2, figsize=(20, 10))
        ax = ax.ravel()

        id = int(id)
        x = roi_df[roi_df.id == id]['x'].values[0]
        y = roi_df[roi_df.id == id]['y'].values[0]
        valid = roi_df[roi_df.id == id]['valid'].values[0]
        ax[0].imshow(roi_df[roi_df.id == id]['mask'].values[0])
        ax[0].set_title(str(id) + ', ' + str(valid) + ', x: ' + str(x) + ', y: ' + str(y))
        ax[0].grid(False)

        ax[1].plot(roi_traces[index])
        ax[1].set_title('index: ' + str(index) + ', id: ' + str(id))
        ax[1].set_ylabel('fluorescence counts')

        ax[3].plot(dff_traces_original[index])
        ax[3].set_title('index: ' + str(index) + ', id: ' + str(id))
        ax[3].set_ylabel('dF/F')

        if id in cell_specimen_ids:
            cell_index = convert.get_cell_index_for_cell_specimen_id(cell_specimen_ids, id)
            ax[2] = plot_cell_zoom(roi_masks, max_projection, id, spacex=10, spacey=10, show_mask=True, ax=ax[2])
            ax[2].grid(False)

            ax[4].imshow(max_projection, cmap='gray')
            mask = np.empty(roi_masks[id].shape)
            mask[:] = np.nan
            (y, x) = np.where(roi_masks[id] == 1)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            ax[4].imshow(mask, cmap='RdBu', alpha=0.5)
            ax[4].set_xlim(xmin - 20, xmax + 20)
            ax[4].set_ylim(ymin - 20, ymax + 20)
            ax[4].grid(False)

            ax[5].plot(dff_traces[cell_index])
            ax[5].set_title('roi index: ' + str(cell_index) + ', id: ' + str(id))
            ax[5].set_ylabel('dF/F')
            ax[5].set_xlabel('frames')
        else:
            cell_index = ''

        fig.tight_layout()
        roi_validation.append(dict(
            fig=fig,
            index=index,
            id=id,
            cell_index=cell_index,
        ))

    return roi_validation


def get_xticks_xticklabels(trace, frame_rate, interval_sec=1, window=[-2,2]):
    """
    Function that accepts a timeseries, evaluates the number of points in the trace, and converts from acquisition frames to timestamps

    :param trace: a single trace where length = the number of timepoints
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param interval_sec: interval in seconds in between labels

    :return: xticks, xticklabels = xticks in frames corresponding to timepoints in the trace, xticklabels in seconds
    """
    interval_frames = interval_sec * frame_rate
    n_frames = len(trace)
    n_sec = n_frames / frame_rate
    xticks = np.arange(0, n_frames + 5, interval_frames)
    xticklabels = np.arange(0, n_sec + 0.1, interval_sec)
    xticklabels = xticklabels + window[0]
    if interval_sec >= 1:
        xticklabels = [int(x) for x in xticklabels]
    return xticks, xticklabels



def plot_mean_trace(traces, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-2,2],
                    ax=None):
    """
    Function that accepts an array of single trial traces and plots the mean and SEM of the trace, with xticklabels in seconds

    :param traces: array of individual trial traces to average and plot. traces must be of equal length
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
    :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
    :param color: color to plot the trace
    :param interval_sec: interval in seconds for x_axis labels
    :param xlims: range in seconds to plot. Must be <= the length of the traces
    :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis

    :return: axis handle
    """
    # xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces)
        times = np.arange(0, len(trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.plot(trace, label=legend_label, linewidth=3, color=color)
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)

        xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks(xticks)
        if interval_sec < 1:
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


# def plot_mean_trace(traces, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=0.5, xlims=[-2,2], ax=None):
#     xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
#     if ax is None:
#         fig, ax = plt.subplots()
#     if len(traces) > 0:
#         trace = np.mean(traces, axis=0)
#         times = np.arange(0, len(trace), 1)
#         sem = (np.std(traces)) / np.sqrt(float(len(traces)))
#
#         ax.plot(trace, label=legend_label, linewidth=3, color=color)
#         ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)
#         xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec)
#         ax.set_xticks([np.round(x,2) for x in xticks])
#         ax.set_xticklabels([np.round(x,1) for x in xticklabels])
#         ax.set_xlim(xlims[0] * int(frame_rate), xlims[1] * int(frame_rate))
#         ax.set_xlabel('time (sec)')
#         ax.set_ylabel(ylabel)
#     sns.despine(ax=ax)
#     return ax


# def plot_mean_trace(traces, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=0.5, xlims=[-1, 2],
#                     ax=None):
#     """
#     Function that accepts an array of single trial traces and plots the mean and SEM of the trace, with xticklabels in seconds
#
#     :param traces: array of individual trial traces to average and plot. traces must be of equal length
#     :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
#     :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
#     :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
#     :param color: color to plot the trace
#     :param interval_sec: interval in seconds for x_axis labels
#     :param xlims: range in seconds to plot. Must be <= the length of the traces
#     :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis
#
#     :return: axis handle
#     """
#     xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
#     if ax is None:
#         fig, ax = plt.subplots()
#     if len(traces) > 0:
#         trace = np.mean(traces, axis=0)
#         times = np.arange(0, len(trace), 1)
#         sem = (np.std(traces)) / np.sqrt(float(len(traces)))
#         ax.plot(trace, label=legend_label, linewidth=3, color=color)
#         ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)
#
#         xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)
#         ax.set_xlim(xlims[0] * int(frame_rate), xlims[1] * int(frame_rate))
#         ax.set_xlabel('time (sec)')
#         ax.set_ylabel(ylabel)
#     sns.despine(ax=ax)
#     return ax

def plot_mean_trace_with_variability(traces, frame_rate, ylabel='dF/F', label=None, color='k', interval_sec=1,
                                     xlims=[-4, 4], ax=None):
#     xlim = [xlims[0] + np.abs(xlims[0]), xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        mean_trace = np.mean(traces, axis=0)
        times = np.arange(0, len(mean_trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        for trace in traces:
            ax.plot(trace, linewidth=1, color='gray')
        ax.plot(mean_trace, label=label, linewidth=3, color=color, zorder=100)
        xticks, xticklabels = get_xticks_xticklabels(mean_trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(0,(np.abs(xlims[0])+xlims[1]) * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    return ax

def plot_flashes_on_trace(ax, analysis, duration=0.25, alpha=0.15):
    """
    Function to create transparent gray bar spanning the duration of visual stimulus presentations to overlay on existing figure

    :param ax: axis on which to plot stimulus presentation times
    :param analysis: ResponseAnalysis class instance
    :param trial_type: 'go' or 'catch'. If 'go', different alpha levels are used for stimulus presentations before and after change time
    :param omitted: boolean, use True if plotting response to omitted flashes
    :param alpha: value between 0-1 to set transparency level of gray bars demarcating stimulus times

    :return: axis handle
    """
    frame_rate = analysis.ophys_frame_rate
    start_frame = np.abs(analysis.sweep_window[0]) * frame_rate
    end_frame = start_frame + (duration * frame_rate)
    ax.axvspan(start_frame, end_frame, facecolor='gray', edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_single_trial_trace(trace, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1,
                            xlims=[-4, 4], ax=None):
    """
    Function to plot a single timeseries trace with xticklabels in secconds

    :return: axis handle
    :param trace: single trial timeseries trace to plot
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
    :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
    :param color: color to plot the trace
    :param interval_sec: interval in seconds for x_axis labels
    :param xlims: range in seconds to plot. Must be <= the length of the traces
    :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis

    :return: axis handle
    """
    xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(trace, label=legend_label, linewidth=3, color=color)

    xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec)
    ax.set_xticks([int(x) for x in xticks])
    ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_xlim(xlims[0] * int(frame_rate), xlims[1] * int(frame_rate))
    ax.set_xlabel('time (sec)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_image_response_for_trial_types(analysis, cell, save_dir=None):
    """
    Function to plot trial avereraged response of a cell for all images separately for 'go' and 'catch' trials. Creates figure and axes to plot.

    :param analysis: ResponseAnalysis class instance
    :param cell: cell index for cell to plot
    :param save: boolean, if True, saves figure to a folder called 'image_responses' in the analysis_dir attribute of the analysis object

    :return: None
    """
    df = analysis.trial_response_df.copy()
    trials = analysis.dataset.trials
    images = trials.change_image_name.unique()
    colors = sns.color_palette('hls', len(images))
    figsize = (20, 5)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for i, trial_type in enumerate(['go', 'catch']):
        for c, change_image_name in enumerate(images):
            selected_trials = trials[
                (trials.change_image_name == change_image_name) & (trials.trial_type == trial_type)].trial.values
            traces = df[(df.cell == cell) & (df.trial.isin(selected_trials))].dff_trace.values
            ax[i] = plot_mean_trace(traces, analysis.ophys_frame_rate, legend_label=None, color=colors[c],
                                    interval_sec=1,
                                    xlims=[-4, 4], ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], analysis, trial_type=trial_type, omitted=False, alpha=0.3)
        ax[i].set_title(trial_type)
    ax[i].set_ylabel('')
    ax[i].legend(images, loc=9, bbox_to_anchor=(1.1, 1))
    title = str(cell) + '_' + str(
        df[df.cell == cell].cell_specimen_id.values[0]) + '_' + analysis.dataset.analysis_folder
    plt.suptitle(title, x=0.47, y=1., horizontalalignment='center')
    fig.tight_layout()
    if save_dir:
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(right=0.85)
        save_figure(fig, figsize, save_dir, 'image_responses', analysis.dataset.analysis_folder+'_'+str(cell), formats=['.png'])
        plt.close()


def plot_trial_trace_heatmap(trial_response_df, cell, cmap='viridis', vmax=0.5, colorbar=False, ax=None, save_dir=None):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = trial_response_df.copy()
    rows = 1
    cols = len(df.change_image_name.unique())
    colors = get_colors_for_response_types(response_types)
    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        ax = ax.ravel()
    resp_types = []
    for i, change_image_name in enumerate(np.sort(df.change_image_name.unique())):
        im_df = df[(df.cell == cell) & (df.change_image_name == change_image_name)]
        n_frames = im_df.dff_trace.values[0].shape[0]
        n_trials = im_df.dff_trace.shape[0]
        response_matrix = np.empty((n_trials, n_frames))
        response_type_list = []
        segments = []
        idx = 0
        segments.append(idx)
        for y, response_type in enumerate(response_types):
            sub_df = im_df[(im_df.behavioral_response_type == response_type)]
            traces = sub_df.dff_trace.values
            for pos, trial in enumerate(range(traces.shape[0])[::-1]):
                response_matrix[idx, :] = traces[int(trial)]
                response_type_list.append(response_type)
                idx += 1
            segments.append(idx)
            if vmax:
                cax = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=vmax, vmin=0)
            else:
                cax = ax[i].pcolormesh(response_matrix, cmap=cmap)
            ax[i].set_ylim(0, response_matrix.shape[0])
            ax[i].set_xlim(0, response_matrix.shape[1])
            ax[i].set_yticks(segments)
            ax[i].set_yticklabels('')
            ax[i].set_xlabel('time (s)')
            xticks, xticklabels = get_xticks_xticklabels(np.arange(0, response_matrix.shape[1], 1), 31., interval_sec=2)
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels([int(x) for x in xticklabels])
            #             ax[i].vlines(x=np.mean(xticks), ymin=0, ymax=response_matrix.shape[0], color='w', linewidth=1)
            ax[i].set_title(change_image_name)
        for s in range(len(segments) - 1):
            ax[i].vlines(x=-10, ymin=segments[s], ymax=segments[s + 1], color=colors[s], linewidth=30)
        ax[0].set_ylabel('trials')
        resp_types.append(response_type_list)
    plt.tight_layout()
    if colorbar:
        plt.colorbar(cax, ax=ax[i])
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'trial_trace_heatmap', 'roi_' + str(cell))
    return ax


def plot_mean_response_by_repeat(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.flash_response_df.copy()
    n_repeats = 15
    palette = sns.color_palette("RdBu", n_colors=n_repeats)
    norm = plt.Normalize(0, n_repeats)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    df = flash_response_df[flash_response_df.cell == cell]
    df = df[df['repeat'] < n_repeats]
    figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.stripplot(data=df, x='image_name', y='mean_response', jitter=.2, size=3, ax=ax, hue='repeat',
                       palette=palette)
    ax.set_xticklabels(df.image_name.unique(), rotation=90);
    ax.legend_.remove()
    cbar = ax.figure.colorbar(mappable=sm, ax=ax)
    cbar.set_label('repeat')
    ax.set_title(str(cell)+'_'+analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_repeat', analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_mean_response_by_image_block(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.flash_response_df.copy()
    n_blocks = len(flash_response_df.image_block.unique())
    palette = sns.color_palette("RdBu", n_colors=n_blocks)
    norm = plt.Normalize(0, n_blocks)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    df = flash_response_df[flash_response_df.cell == cell]
    figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.stripplot(data=df, x='image_name', y='mean_response', jitter=.2, size=3, ax=ax, hue='image_block',
                       palette=palette)
    ax.set_xticklabels(df.image_name.unique(), rotation=90);
    ax.legend_.remove()
    cbar = ax.figure.colorbar(mappable=sm, ax=ax)
    cbar.set_label('image_block')
    ax.set_title(str(cell)+'_'+analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_image_block', analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_sequence_violation(analysis, cell_index, ax=None, save=False):
    dataset = analysis.dataset
    oddball_block = analysis.get_stimulus_block('oddball')
    # oddball_block = analysis.response_df_dict['oddball']
    ophys_frame_rate = dataset.metadata.ophys_frame_rate.values[0]
    if ax is None:
        figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)

    start_times = oddball_block[
        (oddball_block.violation_sequence == False) & (oddball_block.sequence_start == True)].start_time.values
    cell_trace = analysis.dataset.dff_traces_array[cell_index]
    traces = []
    for start_time in start_times:
        trace, timestamps = ut.get_trace_around_timepoint(start_time, cell_trace,
                                                          dataset.timestamps_ophys,
                                                          [0, 2], ophys_frame_rate)
        traces.append(trace)

    ax = plot_mean_trace(traces, ophys_frame_rate, color='b', interval_sec=0.5, ax=ax,
                     legend_label='habituated sequence', xlims=[-1,1])

    start_times = oddball_block[
        (oddball_block.violation_sequence == True) & (oddball_block.sequence_start == True)].start_time.values
    traces = []
    for start_time in start_times:
        trace, timestamps = ut.get_trace_around_timepoint(start_time, cell_trace,
                                                          dataset.timestamps_ophys,
                                                          [0, 2], ophys_frame_rate)
        traces.append(trace)
    ax = plot_mean_trace(traces, ophys_frame_rate, color='r', interval_sec=0.5, ax=ax,
                         legend_label='violation sequence', xlims=[-1,1])
    ax.axvspan(0.75 * ophys_frame_rate, 1 * ophys_frame_rate, facecolor='gray', edgecolor='none', alpha=0.3,
               linewidth=0, zorder=1)
    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell_index)
    ax.set_title('cell index: ' + str(cell_index) + ', cell_specimen_id: ' + str(cell_specimen_id))
    # ax.legend(bbox_to_anchor=(1,1))
    ax.legend(loc=9, bbox_to_anchor=(1, 1))

    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.75)
        save_figure(fig, figsize, dataset.analysis_dir, 'sequence_responses', 'cell_' + str(cell_index))
        plt.close()
    return ax


def plot_randomized_control_responses(analysis, cell_specimen_id, ax=None, save=False):
    from scipy.stats import sem as compute_sem
    if ax is None:
        figsize = (8,5)
        fig, ax = plt.subplots(figsize=figsize)

    rc_pre = analysis.response_df_dict['randomized_control_pre']
    mean = []
    sem = []
    sequence_images = analysis.get_sequence_images()
    oddballs = analysis.get_oddball_images()
    images = sequence_images + oddballs
    for image_id in images:
        means = rc_pre[(rc_pre.cell_specimen_id==cell_specimen_id)&(rc_pre.image_id==image_id)].mean_response.values
        mean.append(np.mean(means))
        sem.append(compute_sem(means))
    ax.errorbar(np.arange(0,len(images)),mean,yerr=sem,fmt='o',color='g',label='pre')

    rc_post = analysis.response_df_dict['randomized_control_post']
    mean = []
    sem = []
    for image_id in images:
        means = rc_post[(rc_post.cell_specimen_id==cell_specimen_id)&(rc_pre.image_id==image_id)].mean_response.values
        mean.append(np.mean(means))
        sem.append(compute_sem(means))
    ax.errorbar(np.arange(0,len(images)),mean,yerr=sem,fmt='o',color='m',label='post')
    ax.set_ylabel('mean dF/F')
    ax.set_xlabel('image_id')
    ax.set_xticks(np.arange(0,len(images)))
    ax.set_xticklabels([int(image) for image in images]);
    ax.set_title('cell '+str(cell_specimen_id)+' - randomized control block')
    ax.legend(bbox_to_anchor=(1.3,1))

    if save:
        save_figure(fig,figsize,analysis.dataset.analysis_dir,'randomized_control','cell_'+str(cell_specimen_id))
        plt.close()
    return ax


def plot_event_detection(dff_traces_array, events, analysis_dir):
    figsize = (20, 15)
    xlims_list = [[0, dff_traces_array[0].shape[0]], [10000, 12000], [60000, 62000]]
    for cell in range(len(dff_traces_array)):
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax = ax.ravel()
        for i, xlims in enumerate(xlims_list):
            ax[i].plot(dff_traces_array[cell], label='dF/F from L0')
            ax[i].plot(events[cell], color='r', label='events')
            ax[i].set_title('roi ' + str(cell))
            ax[i].set_xlabel('2P frames')
            ax[i].set_ylabel('dF/F')
            ax[i].set_xlim(xlims)
        plt.legend()
        fig.tight_layout()
        save_figure(fig, figsize, analysis_dir, 'event_detection', str(cell))
        plt.close()


def plot_trace_with_stimulus_blocks(analysis, cell_index, ax=None, save=False):
    dataset = analysis.dataset
    block_df = analysis.block_df
    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell_index)

    colors = sns.color_palette('deep')
    if ax is None:
        figsize=(20,5)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dataset.timestamps_ophys, dataset.dff_traces_array[cell_index])
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('dF/F')
    ax.set_title(str(cell_specimen_id)+'_'+str(cell_index)+'_'+dataset.analysis_folder)

    for i, block_name in enumerate(block_df.block_name.values):
        start_time = block_df[block_df.block_name==block_name].start_time.values[0]
        end_time = block_df[block_df.block_name==block_name].end_time.values[0]
        ax.axvspan(start_time, end_time, facecolor=colors[i], edgecolor='none', alpha=0.3, linewidth=0, zorder=1, label=block_name)
    ax.legend(bbox_to_anchor=(1,1))

    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(right=0.8)
        save_figure(fig, figsize, dataset.analysis_dir, 'cell_traces', 'cell_'+str(cell_specimen_id))
    return ax

def plot_image_tuning_curve(rdf, image_ids, cell_specimen_id, label='pre', color='g',ax=None, save=False):
    from scipy.stats import sem as compute_sem
    if ax is None:
        figsize = (8,5)
        fig, ax = plt.subplots(figsize=figsize)

    mean = []
    sem = []
    for image_id in image_ids:
        means = rdf[(rdf.cell_specimen_id==cell_specimen_id)&(rdf.image_id==image_id)].mean_response.values
        mean.append(np.mean(means))
        sem.append(compute_sem(means))
    ax.plot(np.arange(0,len(image_ids)),mean,color=color)
    ax.errorbar(np.arange(0,len(image_ids)),mean,yerr=sem,fmt='o',color=color,label=label)
    ax.set_xticks(np.arange(0,len(image_ids),1))
    ax.set_xticklabels([int(image_id) for image_id in image_ids])
    ax.set_ylabel('dF/F')
    ax.set_xlabel('image ID')
    ax.set_title('cell '+str(cell_specimen_id))
    return ax

def plot_image_tc_across_stimulus_types(analysis, cell_specimen_id, ax=None, save_dir=None):
    colors = sns.color_palette('deep')
    # sequence_images = analysis.get_sequence_images()
    # oddball_images = analysis.get_oddball_images()
    image_ids = analysis.get_image_ids()
    # image_ids = list(sequence_images) + list(oddball_images)
    response_df_dict = analysis.response_df_dict.copy()
    odf = response_df_dict['oddball']
    rc_pre = response_df_dict['randomized_control_pre']
    rc_post = response_df_dict['randomized_control_post']
    tdf = response_df_dict['transition_control']

    if ax is None:
        figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)

    label = 'sequence'
    ax = plot_image_tuning_curve(odf, image_ids, cell_specimen_id, label=label, color=colors[3], ax=ax, save=False)

    label = 'randomized pre'
    ax = plot_image_tuning_curve(rc_pre, image_ids, cell_specimen_id, label=label, color=colors[0], ax=ax, save=False)

    label = 'randomized post'
    ax = plot_image_tuning_curve(rc_post, image_ids, cell_specimen_id, label=label, color=colors[2], ax=ax, save=False)

    label = 'transition control'
    ax = plot_image_tuning_curve(tdf, image_ids, cell_specimen_id, label=label, color=colors[4], ax=ax, save=False)

    ax.legend(loc='upper right')
    ax.legend(bbox_to_anchor=(1.1, 1))
    if save_dir:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'image_tuning_curves', 'cell_' + str(cell_specimen_id))
        plt.close()
    return ax

def plot_image(dataset, session_block_name, image_id, ax=None):
    st = dataset.stimulus_table
    st = st[st.session_block_name==session_block_name]
    file_name = st[st.image_id==image_id].iloc[0].data_file_name
    file_index = st[st.image_id==image_id].iloc[0].data_file_index
    image_data = np.load(file_name)
    image = image_data[file_index, :, :]
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title(str(int(image_id)))
    return ax


def plot_cell_summary_figure(analysis, cell_specimen_id, save=False, show=True):
    dataset = analysis.dataset
    cell_index = dataset.get_cell_index_for_cell_specimen_id(cell_specimen_id)
    sns.set_context('talk', rc={'lines.markeredgewidth': 2})
    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .14), yspan=(0, .22))
    ax = plot_cell_zoom(dataset.roi_mask_array, dataset.max_projection, cell_index, spacex=20, spacey=20,
                           show_mask=True, ax=ax)
    ax.set_title('')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.12, .26), yspan=(0, .22))
    ax = plot_cell_zoom(dataset.roi_mask_array, dataset.red_channel_image, cell_index, spacex=20, spacey=20,
                           show_mask=False, ax=ax)
    ax.set_title('')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.25, .85), yspan=(0, .22))
    ax = plot_trace_with_stimulus_blocks(analysis, cell_index, ax=ax, save=False)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .25), yspan=(.22, .42))
    ax = plot_sequence_violation(analysis, cell_index, save=False, ax=ax)
    ax.set_title('')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.32, .58), yspan=(.22, .42))
    ax = plot_randomized_control_responses(analysis, cell_specimen_id, ax=ax, save=False)
    ax.set_title('randomized control blocks')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.64, .9), yspan=(.22, .42))
    ax = plot_image_tc_across_stimulus_types(analysis, cell_specimen_id, ax=ax, save_dir=None)

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.83, 1), yspan=(.78, 1))
    # table_data = format_table_data(dataset)
    # xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    # xtable.scale(1, 3)
    # ax.axis('off');

    fig.tight_layout()
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'cell_summary_plots', 'cell_' + str(cell_specimen_id))
        save_figure(fig, figsize, dataset.cache_dir, 'cell_summary', dataset.analysis_folder + '_' + str(cell_specimen_id))
        if not show:
            plt.close()

if __name__ == '__main__':
    experiment_id = 719996589

    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset

    dataset = VisualBehaviorOphysDataset(experiment_id)

    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

    analysis = ResponseAnalysis(dataset)

    print('plotting cell responses')
    for cell in dataset.get_cell_indices():
        plot_image_response_for_trial_types(analysis, cell)
    print('done')
