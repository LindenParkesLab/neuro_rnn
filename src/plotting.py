import sys, os, platform
import urllib.request
from src.utils import get_p_val_string, compute_pc_var, get_my_colors

import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy import stats

import nibabel as nib
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from nilearn import datasets, plotting as nilearn_plotting
from sklearn.decomposition import PCA

from src.utils import get_my_colors

CBIG_BASE_URL = (
    'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/'
    'stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/'
    'Parcellations/FreeSurfer5.3/fsaverage5/label'
)


def _get_schaefer_annot(n_parcels=200, yeo_networks=7, data_dir=None):
    """Download Schaefer FreeSurfer .annot files if not already cached."""
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'schaefer_2018')
    annot_dir = os.path.join(data_dir, 'fsaverage5')
    os.makedirs(annot_dir, exist_ok=True)

    paths = {}
    for hemi in ('lh', 'rh'):
        fname = f'{hemi}.Schaefer2018_{n_parcels}Parcels_{yeo_networks}Networks_order.annot'
        local_path = os.path.join(annot_dir, fname)
        if not os.path.isfile(local_path):
            url = f'{CBIG_BASE_URL}/{fname}'
            print(f'Downloading {fname} ...')
            urllib.request.urlretrieve(url, local_path)
        paths[hemi] = local_path
    return paths['lh'], paths['rh']


def _roi_to_vtx(roi_data, annot_file):
    """Map parcel-level data to vertex-level using a FreeSurfer .annot file."""
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    vtx_data = np.zeros(labels.shape)
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
    for i in unique_labels:
        vtx_data[labels == i] = roi_data[i - 1]
    return vtx_data


def plot_surface(data, hemi='lh', n_parcels=200, yeo_networks=7,
                 cmap='coolwarm', cblim=None, title=None,
                 figsize=(3.5, 2), data_dir=None):
    """Plot parcel-level data on an fsaverage5 inflated surface.

    Parameters
    ----------
    data : array-like, shape (n_parcels_per_hemi,) or (n_parcels,)
        Parcel-level values. For hemi='lh' with n_parcels=200, pass 100
        values corresponding to the left-hemisphere parcels.
        For hemi='both', pass the full 200-parcel vector (LH then RH).
    hemi : str
        'lh' for left hemisphere only, 'rh' for right only, 'both' for both.
    n_parcels : int
        Total number of Schaefer parcels (both hemispheres), default 200.
    yeo_networks : int
        Yeo network resolution (7 or 17), default 7.
    cmap : str
        Matplotlib colormap name. Default 'coolwarm' (diverging).
    cblim : tuple or None
        (vmin, vmax) for the colorbar. If None, auto-determined; for
        diverging colormaps, symmetrized around zero.
    title : str or None
        Optional title for the figure.
    figsize : tuple
        Figure size (width, height).
    data_dir : str or None
        Directory for caching atlas files. Defaults to ~/nilearn_data/schaefer_2018.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    data = np.asarray(data, dtype=float)
    lh_annot, rh_annot = _get_schaefer_annot(n_parcels, yeo_networks, data_dir)
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    half = n_parcels // 2

    # determine which hemispheres to plot
    if hemi == 'both':
        hemis = ['lh', 'rh']
        vtx = {
            'lh': _roi_to_vtx(data[:half], lh_annot),
            'rh': _roi_to_vtx(data[half:], rh_annot),
        }
    elif hemi == 'lh':
        hemis = ['lh']
        vtx = {'lh': _roi_to_vtx(data[:half], lh_annot)}
    else:
        hemis = ['rh']
        vtx = {'rh': _roi_to_vtx(data[:half], rh_annot)}

    # color limits
    if cblim is not None:
        vmin, vmax = cblim
    else:
        vmax = np.nanmax(np.abs(data))
        if cmap in ('coolwarm', 'RdBu_r', 'RdYlBu_r', 'bwr', 'seismic'):
            vmin = -vmax
        else:
            vmin = np.nanmin(data)

    # layout: 2 rows (lateral, medial) x n_hemis columns
    n_cols = len(hemis)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize,
                             subplot_kw={'projection': '3d'},
                             gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    hemi_map = {'lh': ('left', 'infl_left', 'sulc_left'),
                'rh': ('right', 'infl_right', 'sulc_right')}

    for col, h in enumerate(hemis):
        hemi_label, surf_key, sulc_key = hemi_map[h]
        for row, view in enumerate(['lateral', 'medial']):
            ax = axes[row, col]
            nilearn_plotting.plot_surf_roi(
                fsaverage[surf_key], roi_map=vtx[h],
                hemi=hemi_label, view=view,
                vmin=vmin, vmax=vmax,
                bg_map=fsaverage[sulc_key], bg_on_data=True,
                axes=ax, darkness=0.5,
                cmap=cmap, colorbar=False,
            )
            # bring the medial camera closer so both views appear the same size
            if view == 'medial':
                ax._dist = ax._dist * 0.85

    # colorbar
    im = plt.imshow(np.array([[vmin, vmax]]), cmap=cmap, vmin=vmin, vmax=vmax)
    im.set_visible(False)
    cb_ax = fig.add_axes([0.82, 0.25, 0.03, 0.5])
    fig.colorbar(im, cax=cb_ax)

    if title is not None:
        fig.suptitle(title, fontsize=10, y=0.95)

    fig.subplots_adjust(left=-0.15, right=0.8, bottom=-0.05, top=1.05,
                        wspace=0.0, hspace=0.0)
    plt.show()

    return fig


def my_reg_plot(x, y, xlabel, ylabel, ax, c='gray', annotate='pearson', regr_line=True, kde=True, fontsize=8):
    if len(x.shape) > 1 and len(y.shape) > 1:
        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            mask_x = ~np.eye(x.shape[0], dtype=bool) * ~np.isnan(x)
            mask_y = ~np.eye(y.shape[0], dtype=bool) * ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
        else:
            mask_x = ~np.isnan(x)
            mask_y = ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
    elif len(x.shape) == 1 and len(y.shape) == 1:
        mask_x = ~np.isnan(x)
        mask_y = ~np.isnan(y)
        mask = mask_x * mask_y
        indices = np.where(mask)
    else:
        print('error: input array dimension mismatch.')

    try:
        x = x[indices]
        y = y[indices]
    except:
        pass

    try:
        c = c[indices]
    except:
        pass

    # kde plot
    if kde == True:
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.25)
        except:
            pass

    # regression line
    if regr_line == True:
        # color_blue = sns.color_palette("Set1")[1]
        my_colors = get_my_colors()
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=my_colors['north_sea_green'])

    # scatter plot
    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=2.5, alpha=0.5)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=2.5, alpha=0.5)

    # axis options
    ax.set_xlabel(xlabel, labelpad=0)
    ax.set_ylabel(ylabel, labelpad=0)
    # ax.tick_params(pad=-2.5)
    # ax.grid(False)
    # sns.despine(right=True, top=True, ax=ax)
    sns.despine(offset=0, trim=False, left=False, right=True, top=True, bottom=False, ax=ax)
    ax.tick_params(left=True, bottom=True)

    # annotation
    r, r_p = sp.stats.pearsonr(x, y)
    rho, rho_p = sp.stats.spearmanr(x, y)
    if type(annotate) == str:
        if annotate == 'pearson':
            textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'spearman':
            textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'both':
            textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                                 rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
    elif type(annotate) == tuple:
        coef = annotate[0]
        p = annotate[1]
        textstr = 'coef = {:.2f}, {:}'.format(coef, get_p_val_string(p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    else:
        pass


def get_conditions(info):
    """Get a list of task conditions to plot."""
    conditions = info.columns
    # This condition's unique value should be less than 5
    new_conditions = list()
    for c in conditions:
        try:
            n_cond = len(pd.unique(info[c]))
            if 1 < n_cond < 5:
                new_conditions.append(c)
        except TypeError:
            pass

    return new_conditions


def plot_activity(hidden_activity, info, config, condition='choice', n_components=0, mask=None):
    cmap = sns.color_palette("Set2")
    if mask is not None:
        hidden_activity = hidden_activity[:, :, mask]
    [n_trials, n_timepoints, n_nodes] = hidden_activity.shape

    if n_components == 0:
        hidden_activity = hidden_activity.mean(axis=-1)
        f_width = 5
        f_height = 1.5
    elif n_components:
        pca = PCA(n_components=n_components)
        activity_reshape = np.reshape(hidden_activity, (n_trials*n_timepoints, n_nodes))
        pca.fit(activity_reshape)
        print(pca.explained_variance_ratio_)
        f_width = 4
        f_height = 4

    if n_components >= 3:
        f, ax = plt.subplots(1, 1, figsize=(f_width, f_height), subplot_kw={'projection': '3d'})
    else:
        f, ax = plt.subplots(1, 1, figsize=(f_width, f_height))

    if n_components == 0:
        for value in pd.unique(info[condition]):
            activity_to_plot = hidden_activity[info[condition] == value].mean(axis=0)
            t_plot = np.arange(hidden_activity.shape[1]) * config['time_step']
            ax.plot(t_plot, activity_to_plot, 'o-', label=str(value), color=cmap[value])
    else:
        for trial in np.arange(n_trials):
            try:
                trial_size = info['coh'].iloc[trial]/info['coh'].max()
            except:
                trial_size = 0.5

            activity_to_plot = hidden_activity[trial]
            activity_to_plot = pca.transform(activity_to_plot)
            if n_components == 2:
                plt.plot(activity_to_plot[:, 0], activity_to_plot[:, 1], 'o-', color=cmap[info[condition].iloc[trial]], linewidth=trial_size, markersize=trial_size)
            elif n_components == 3:
                plt.plot(activity_to_plot[:, 0], activity_to_plot[:, 1], activity_to_plot[:, 2], 'o-', color=cmap[info[condition].iloc[trial]], linewidth=trial_size, markersize=trial_size)

            # add mean
            for value in pd.unique(info[condition]):
                activity_to_plot = hidden_activity[info[condition] == value].mean(axis=0)
                activity_to_plot = pca.transform(activity_to_plot)
                if n_components == 2:
                    plt.plot(activity_to_plot[:, 0], activity_to_plot[:, 1], 'o-', color=cmap[value], label=str(value), path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
                elif n_components == 3:
                    plt.plot(activity_to_plot[:, 0], activity_to_plot[:, 1], activity_to_plot[:, 2], 'o-', color=cmap[value], label=str(value), path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

    if n_components == 0:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Activity')
        ax.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
    else:
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        if n_components >= 3:
            ax.set_zlabel('PC 3')
        # ax.legend(title=condition)

    if mask is not None and type(mask) == int:
        ax.set_title('Unit {:d}'.format(mask + 1))

    f.tight_layout()
    plt.show()


def plot_activity_variance(hidden_activity, config, n_components=3, normalize=True, mean=False, mask=None):
    if mask is not None:
        hidden_activity = hidden_activity[:, :, mask]
    hidden_activity_pc_var = compute_pc_var(hidden_activity, n_components=n_components, normalize=normalize)

    cmap = sns.color_palette("Set1")
    f, ax = plt.subplots(1, 1, figsize=(5, 1.5))
    t_plot = np.arange(hidden_activity.shape[1]) * config['time_step']
    if mean:
        ax.plot(t_plot, hidden_activity_pc_var.mean(axis=1), 'o-', label='mean', color=cmap[0])
    else:
        if n_components > 0:
            for pc in np.arange(n_components):
                ax.plot(t_plot, hidden_activity_pc_var[:, pc], 'o-', label=str(pc+1), color=cmap[pc])
        else:
            ax.plot(t_plot, hidden_activity_pc_var[:, 0], 'o-', label='Mean', color=cmap[0])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('PC variance')
    if n_components > 0:
        ax.legend(title='PCs', loc='center left', bbox_to_anchor=(1.0, 0.5))
    else:
        ax.legend(title='', loc='center left', bbox_to_anchor=(1.0, 0.5))
    f.tight_layout()
    plt.show()


def plot_accuracy_vs_feature(accuracy, feature, rnn_label='', feature_label='',
                             fig_width=4, fig_height=2, accuracy_color='r', feature_color='k', x_step=100):
    f, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax_twin = ax.twinx()

    # add accuracy to plot
    n_runs = accuracy.shape[0]
    accuracy_mean = np.nanmean(accuracy, axis=0)*100
    accuracy_std = np.nanstd(accuracy, axis=0)*100
    ci = 1.96 * (accuracy_std / np.sqrt(n_runs))
    accuracy_ci_lower = accuracy_mean - ci
    accuracy_ci_upper = accuracy_mean + ci

    n_logged_epochs = accuracy.shape[1]
    n_epochs = n_logged_epochs * x_step
    x = np.arange(x_step, n_epochs + x_step, x_step)
    # print(x)

    ax.plot(x, accuracy_mean, color=accuracy_color, label=rnn_label)
    ax.fill_between(x, accuracy_ci_lower, accuracy_ci_upper, color=accuracy_color, alpha=0.25)
    ax.set_ylim([-2.5, 100])
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epochs')

    # overlay feature
    y_mean = np.nanmean(feature, axis=0)
    y_std = np.nanstd(feature, axis=0)
    ci = 1.96 * (y_std / np.sqrt(n_runs))
    ci_lower = y_mean - ci
    ci_upper = y_mean + ci

    x_step2 = int(n_logged_epochs / (feature.shape[1]-1))
    x2 = x[::x_step2]
    if len(x2) < len(y_mean):
        x2 = np.append(x2, n_epochs)
    # print(x_step2, x2)

    if y_mean.ndim == 1:
        ax_twin.plot(x2, y_mean, color=feature_color, label=feature_label)
        ax_twin.fill_between(x2, ci_lower, ci_upper, color=feature_color, alpha=0.25)
    elif y_mean.ndim == 2:
        n_loops = y_mean.shape[1]
        for i in np.arange(n_loops):
            try:
                ax_twin.plot(x2, y_mean[:, i], color=feature_color[i], alpha=0.75)
                ax_twin.fill_between(x2, ci_lower[:, i], ci_upper[:, i], color=feature_color[i], alpha=0.25)
            except:
                ax_twin.plot(x2, y_mean[:, i], color=feature_color, alpha=0.75)
                ax_twin.fill_between(x2, ci_lower[:, i], ci_upper[:, i], color=feature_color, alpha=0.25)

    ax_twin.set_ylabel(feature_label)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines + lines2, labels + labels2)

    # set axis limits and despine
    for this_ax in [ax, ax_twin]:
        sns.despine(offset=3, trim=False, left=False, right=False, top=True, bottom=False, ax=this_ax)
    
    ax_twin.hlines(y=1, xmin=x2[0], xmax=x2[-1], linestyle='dashed', color='k')

    f.tight_layout()
    plt.show()

    return f

def plot_iodata(
    x, y, n_trials=7, palette=None,
    rc_params={}, fig_params={}, ax_params={}, lg_params={},
    title=None, show=True, savefig=False, fname='io_data',
    **kwargs
):
    """
    Plot input (x) and output (y) data.

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    n_trials : _type_, optional
        _description_, by default 7
    palette : _type_, optional
        _description_, by default None
    rc_params : dict
        dictionary of matplotlib rc parameters, by default {}
    fig_params : dict
        dictionary of figure properties, by default {}
    ax_params : dict
        dictionary of axes properties, by default {}
    lg_params : dict
        dictionary of legend settings, by default {}
    title : _type_, optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default None
    """
    # get end points for trials to plot trial separators
    if isinstance(x, list):
        n_trials = np.min([len(x), 10])
        x = x[:n_trials]
        y = y[:n_trials]

        tf, end_points = 0, []
        for _, trial in enumerate(x):
            tf += len(trial)
            end_points.append(tf)
    else:
        end_points = None

    # set plotting theme
    rc_defaults = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
                   'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'legend.fontsize': 8, 'legend.loc': 'best',
                   'lines.linewidth': 1, 'savefig.format': 'png'}
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)

    # open figure and axes
    fig_defaults = {'figsize': (12, 2)}  # 12, 4.5
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    ax = fig.subplots(1, 1)

    # plot inputs (x) and outputs (y)
    sns.lineplot(
        data=x, palette=palette, dashes=False, legend=False, ax=ax,
        **kwargs
    )
    sns.lineplot(
        data=y, palette=palette, dashes=False, legend=False, ax=ax,
        linewidth=1.5, **kwargs
    )

    # set legend
    x_labels = ['x'] if x.ndim == 1 else [f'x{n+1}' for n in range(x.shape[1])]
    y_labels = ['y'] if y.ndim == 1 else [f'y{n+1}' for n in range(y.shape[1])]
    lg_defaults = {'labels': x_labels + y_labels}
    lg_defaults.update(**lg_params)
    ax.legend(handles=ax.lines, **lg_defaults)

    # set axes properties
    ax_defaults = {'xlabel': 'time steps', 'ylabel': 'signal amplitude',
                   'xlim': [0, 200]}
    ax_defaults.update(**ax_params)
    ax.set(**ax_defaults)

    # plot trial line separators
    if end_points is not None:
        min_y = np.min(y).astype(int)
        max_y = np.max(y).astype(int)
        for tf in end_points:
            ax.plot(
                tf * np.ones((2)), np.array([min_y, max_y]), c='black',
                linestyle='--'
            )

    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()

    # reset rc defaults (keep Arial as the default font)
    mpl.rcdefaults()
    from src import use_arial
    use_arial()