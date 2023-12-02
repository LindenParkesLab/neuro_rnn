import sys, os, platform
from src.utils import get_p_val_string

import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy import stats

import seaborn as sns
import pkg_resources
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        color_blue = sns.color_palette("Set1")[1]
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color_blue)

    # scatter plot
    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=5, alpha=0.5)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=5, alpha=0.5)

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


def plot_activity(activity, info, config, condition='choice', run_pca=False, mask=None):
    if run_pca:
        f_width = 4
        f_height = 4
    else:
        f_width = 5
        f_height = 1.5

    if mask is not None:
        activity = activity[:, :, mask]
        if activity.ndim == 3 and run_pca:
            activity_reshape = np.reshape(activity, (-1, activity.shape[-1]))
            pca = PCA(n_components=2)
            pca.fit(activity_reshape)
        elif activity.ndim == 3 and not run_pca:
            activity = activity.mean(axis=-1)
    elif mask is None and run_pca:
        activity_reshape = np.reshape(activity, (-1, activity.shape[-1]))
        pca = PCA(n_components=2)
        pca.fit(activity_reshape)
    elif mask is None and not run_pca:
        activity = activity.mean(axis=-1)

    f, ax = plt.subplots(1, 1, figsize=(f_width, f_height))
    t_plot = np.arange(activity.shape[1]) * config['dt']
    values = pd.unique(info[condition])
    for value in values:
        if run_pca:
            a = activity[info[condition] == value].mean(axis=0)
            a = pca.transform(a)  # (N_time, N_PC)
            plt.plot(a[:, 0], a[:, 1], 'o-', label=str(value))
        else:
            a = activity[info[condition] == value]
            ax.plot(t_plot, a.mean(axis=0), 'o-', label=str(value))
    ax.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
    if run_pca:
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
    else:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Activity')
    if mask is not None and type(mask) == int:
        ax.set_title('Unit {:d}'.format(mask + 1))


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
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
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

    # reset rc defaults
    mpl.rcdefaults()