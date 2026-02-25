# ----------------------------------------------------------------------------------------- #
# Code set-up
# ----------------------------------------------------------------------------------------- #

import os, random, time, sys, warnings
from datetime import datetime

import numpy as np
import scipy as sp
from scipy.spatial import distance
from scipy.ndimage import uniform_filter1d
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
import pandas as pd
from IPython.display import display
import torch
import src.utils as utils
import math
from tqdm import tqdm
from src.neural_network import ModelStateManager, ModelDataManager, create_rnn_and_env_for_model, run_testing, run_testing_rest

# import plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams.update({"font.size": 8})
plt.rcParams["svg.fonttype"] = "none"
plt.rc('font', family='Arial')
plt.ioff()
import seaborn as sns
sns.set_style("white")
from src.plotting import my_reg_plot

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ----------------------------------------------------------------------------------------- #
# Define autocorrelation function
# ----------------------------------------------------------------------------------------- #

def compute_ac_acw_aca(ts, nlags=None, fit=False):
    """ Calculate the signal autocorrelation (lagged correlation)

    Parameters
    ----------
    ts : np.array (n_timepoints,)
        Time series.
    nlags : int (default=None)
        Number of lags to compute acf over

    Returns
    -------
    ac : np.array (n_timepoints,) or (n_lags,)
        Time lagged (auto)correlation
    acw_50 : int
        Lag at which (auto)correlation drops to its minimum value that is >= 0.5
    acw_0 : int
        Lag at which (auto)correlation drops to its minimum value that is >= 0.0
    aca_50 : float
        Area under the AC curve up until the acw_50 point
    aca_0 : float
        Area under the AC curve up until the acw_0 point

    """

    # deal with flat inputs
    if False: #np.all(ts == np.mean(ts)):
        
        ac = np.zeros(nlags) 
        acw_50 = 0
        acw_0 = 0
        aca_50 = 0
        aca_0 = 0

    else:
        
        # compute auto correlation function using statsmodels
        ac = acf(ts, nlags=nlags)[:-1]
        
        # get data for the 50% point
        if np.any(ac < 0.5):
            ac_trim_50 = ac[:np.where(ac < 0.5)[0][0]]
            acw_50 = ac_trim_50.shape[0]
            aca_50 = ac_trim_50.sum(axis=0)
        else:
            acw_50 = 0
            aca_50 = 0
        
        # get data for the 0 point
        if np.any(ac < 0):
            ac_trim_0 = ac[:np.where(ac < 0)[0][0]]
            acw_0 = ac_trim_0.shape[0]
            aca_0 = ac_trim_0.sum(axis=0)
        else:
            acw_0 = 0
            aca_0 = 0
    
    return ac, acw_50, acw_0, aca_50, aca_0

# ----------------------------------------------------------------------------------------- #
# Set up files paths and options
# ----------------------------------------------------------------------------------------- #

# directories and files
username = os.getenv('USER')
if sys.platform == 'darwin':
    if username == 'ahmad':
        datadir = '/Users/ahmad/software/snaplab_github/neuro_rnn/data'
        modeldir = '/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model'
        modeldir = '/Volumes/Sabrent_2TB/rutgers/neuro_rnn/data/20250116' 
        outdir = modeldir
elif sys.platform == 'linux':
    if username == 'ab2792':
        datadir = '/home/ab2792/software/snaplab_github/neuro_rnn/data'
        modeldir = '/home/ab2792/data/neuro_rnn/results/pytorch/model'
        outdir = modeldir
    elif username == 'lindenmp':
        datadir = '/home/lindenmp/research_projects/neuro_rnn/data'
        modeldir = '/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
        outdir = '/home/lindenmp/research_projects/neuro_rnn/results/figs'

sa_kernel_file = os.path.join(datadir, 'schaefer200_sa-axis.npy')
ut_kernel_file = os.path.join(datadir, 'schaefer200_ut-axis.npy')
eucl_kernel_file = os.path.join(datadir, 'schaefer200_centroids.csv')
myelin_data_file = os.path.join(modeldir, 'myelin', 'HCP_YA_Schaefer2018_200Parcels_7Networks_order_myelin.npy')
myelin_data_df_file = os.path.join(modeldir, 'myelin', 'HCP_YA_Schaefer2018_200Parcels_7Networks_order_myelin_df.csv')
fmri_data_file = '/Volumes/Sabrent_2TB/rutgers/HCP/fmri/HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest.npy'
fmri_data_df_file = '/Volumes/Sabrent_2TB/rutgers/HCP/fmri/HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest_df.csv'

check_only = False
verbose = False
save_figs = True

do_accuracy = True
do_eigen = True
do_acf = True 

# ----------------------------------------------------------------------------------------- #
# Load axes and kernels
# ----------------------------------------------------------------------------------------- #

hidden_size = 100

# euclidean
centroids = pd.read_csv(eucl_kernel_file)
centroids = centroids[:hidden_size]  # pull out left hemisphere
roi_names = centroids['ROI Name'].str.replace('7Networks_LH_','')
centroids.set_index("ROI Name", inplace=True)
distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
distance_matrix_eucl = utils.normalize_x(distance_matrix)

# sa axis
sa_axis = np.load(sa_kernel_file)
sa_axis = sa_axis[:hidden_size]  # pull out left hemisphere
distance_matrix = utils.get_brainmap_distance(brain_map=sa_axis)
distance_matrix_sa = utils.normalize_x(distance_matrix)

# ut axis
ut_axis = np.load(ut_kernel_file)
ut_axis = ut_axis[:hidden_size]  # pull out left hemisphere
distance_matrix = utils.get_brainmap_distance(brain_map=ut_axis)
distance_matrix_ut = utils.normalize_x(distance_matrix)

# n i/o
n_io = utils.get_n_io(mask_weights=True, hidden_size=hidden_size)
n_nodes_in = int(n_io.split('-')[0])
n_nodes_out = int(n_io.split('-')[1])
n_nodes_inter = hidden_size - n_nodes_in - n_nodes_out
n_nodes = n_nodes_in + n_nodes_inter + n_nodes_out

# node groups
node_groups = np.concatenate((np.ones((n_nodes_in,))*0, 
                              np.ones((n_nodes_inter,))*1, 
                              np.ones((n_nodes_out,))*2), 
                             axis=0)
node_groups_labels = ['in' if v == 0 else 'inter' if v==1 else 'out' for v in node_groups]

# node colors
col_input = np.array([228, 149, 145]) / 255 
col_bys = np.array([129, 223, 180]) / 255 
col_output = np.array([122, 139, 165]) / 255
colors = np.concatenate((np.tile(col_input, [n_nodes_in,1]), 
                         np.tile(col_bys, [n_nodes_inter,1]), 
                         np.tile(col_output, [n_nodes_out,1])), 
                        axis=0 ) 
colors_categ = np.unique(colors,axis=0)

# ----------------------------------------------------------------------------------------- #
# BEGIN MAIN LOOP
# ----------------------------------------------------------------------------------------- #

model_params_file = os.path.join(datadir, 'model_params_alpha_pdm.csv')
tmp = utils.load_params_csv(model_params_file)
n_df_rows = len(tmp)

for index_start in np.arange(0, n_df_rows, 3):
    
    print(' ')

    # choose rows and load params
    rows_to_plot = list(np.arange(index_start, index_start+3))
    print(f'Plotting data from models {rows_to_plot}')

    model_params, task_names, n_tasks, kernel_labels, n_kernels = utils.get_params_dataframe(model_params_file, rows=rows_to_plot, verbose=False)
    n_models = len(model_params)
    
    if not model_params.alpha.nunique() == 1:
        raise ValueError('Inconsistent alpha parameter.') 
    if not model_params.rnn_model.nunique() == 1:
        raise ValueError('Inconsistent alpha parameter.') 
    
    # set up output file names
    rnn_alpha = model_params.alpha[0]
    rnn_model = model_params.rnn_model[0]
    
    models_file = os.path.splitext(os.path.basename(model_params_file))[0]
    suff = '{:}_{:}_alpha-{:}'.format(models_file, rnn_model, rnn_alpha)
    file_path_plots = os.path.join(outdir, 'accuracy_' + suff + '_plots.svg')
    file_path_violin = os.path.join(outdir, 'accuracy_' + suff + '_violins.svg')
    file_path_diff = os.path.join(outdir, 'accuracy_' + suff + '_diff.svg')
    file_path_ac = [[ os.path.join(outdir, 'acf_' + suff + '_task_' + model_params.kernel_type[i] + '.svg') for i in range(len(model_params)) ],
                    [ os.path.join(outdir, 'acf_' + suff + '_rest_' + model_params.kernel_type[i] + '.svg') for i in range(len(model_params)) ]]
    

    # ----------------------------------------------------------------------------------------- #
    # Load trained models
    # ----------------------------------------------------------------------------------------- #

    device = torch.device('cpu')

    if check_only:
        
        found_count_models = 0
        found_count_outputs = 0
        for model_idx in range(n_models):
            this = model_params.iloc[model_idx]
            file_path_models = os.path.join(modeldir, this.file_str_models)
            file_path_outputs = os.path.join(modeldir, this.file_str_outputs)
            file_exists_models = os.path.isfile(file_path_models)
            file_exists_outputs = os.path.isfile(file_path_outputs)
            if file_exists_models:
                found_count_models += 1
            else:
                print(f'Not found ... {this.file_str_models} ({model_idx})')
            if file_exists_outputs:
                found_count_outputs += 1
            else:
                print(f'Not found ... {this.file_str_outputs} ({model_idx})')
        print(f'Found {found_count_models}/{n_models} model files.')
        print(f'Found {found_count_outputs}/{n_models} output files.')
        
    else:
        
        trained_models_all = [] # list where each item holds the model states from one model
        model_accuracy_all = [] # list where each items holds performance data from one model
        
        for model_idx in tqdm(range(n_models), desc='Loading data'):
            
            # get model details
            this = model_params.iloc[model_idx]
            utils.check_if_supported(task=this.task_no_modifier, modifier=this.task_modifier)

            # get data file name
            file_path_models = os.path.join(modeldir, this.file_str_models)
            file_path_outputs = os.path.join(modeldir, this.file_str_outputs)
            
            # remove delay variability for testing
            if 'delay' in this['env_kwargs']['timing']:
                d, _ = utils.parse_task_modifier(this.task_modifier)
                this['env_kwargs']['timing']['delay'] = d[0]

            # get info
            state_manager = ModelStateManager(file_path_models)
            n_runs, logged_epochs, keys_per_epoch = state_manager.get_info()
            n_runs = np.min((n_runs,100))
            
            # create rnn and dataset for each trained run
            trained_models = [] # list where each item holds the model states from one run 
            for run in range(n_runs):
                dataset, rnn = create_rnn_and_env_for_model(this, 
                                                            run = run, 
                                                            epoch = logged_epochs[-1], 
                                                            data_dir = modeldir, 
                                                            device = device)
                trained_models.append({'dataset': dataset, 'rnn': rnn})
            trained_models_all.append(trained_models)
            
            # load outputs
            output_manager = ModelDataManager(file_path_outputs)
            test_accuracy_all_runs = output_manager.load_key_across_runs('test_accuracy')
            model_accuracy_all.append(test_accuracy_all_runs)

    # ----------------------------------------------------------------------------------------- #
    # Test models
    # ----------------------------------------------------------------------------------------- #

    # define number of test trials
    n_trials = 50

    # define window size for smoothing of noise input during resting state testing
    smooth_noise = 0

    # initialize output lists where each item is data from one model
    test_data_task = []
    test_data_rest = []

    # models loop
    for model_idx in tqdm(range(n_models), desc='Testing models'):
        
        # get all trained runs of this model
        trained_models = trained_models_all[model_idx]
        
        # initialize model output lists where each item is data from one run
        test_model_task = []
        test_model_rest = []
        
        # runs loop
        for run in range(len(trained_models)):
            
            # get this run's ngym dataset
            dataset = trained_models[run]['dataset']
            
            # get this run's trained model
            rnn = trained_models[run]['rnn']
            
            # test this run on unseen task trials
            accuracy, inputs, labels, hidden_activity, output_activity, info \
                = run_testing(dataset=dataset,
                            model=rnn,
                            n_trials=n_trials,
                            verbose=False)
            this_test_task = {
                'accuracy': accuracy,
                'inputs': inputs,
                'labels': labels,
                'hidden_activity': hidden_activity,
                'output_activity': output_activity,
                'info': info
            }
            test_model_task.append(this_test_task)
            
            # test this run on noise inputs
            inputs_rest, hidden_activity_rest, output_activity_rest = run_testing_rest(rnn, 
                                                                                    smooth_noise=smooth_noise, 
                                                                                    noise_mean=0.5, 
                                                                                    noise_sd=0.15,
                                                                                    n_steps=500,
                                                                                    fix_input_channels=[])
            this_test_rest = {
                'inputs': inputs_rest,
                'hidden_activity': hidden_activity_rest,
                'output_activity': output_activity_rest
            }
            test_model_rest.append(this_test_rest)
        
        # append this model's data (all runs)
        test_data_task.append(test_model_task)
        test_data_rest.append(test_model_rest)

    # ----------------------------------------------------------------------------------------- #
    # Compute ACF
    # ----------------------------------------------------------------------------------------- #

    # initialize lists that will store model-wise data
    ac_data_task = []
    ac_data_rest = []
    dynamic_keys = ['acw_50','acw_0','aca_50','aca_0']

    # models loop
    for model_idx in tqdm(range(n_models), desc='Calculating AC'):
        
        # initialize lists that will store run-wise data
        ac_model_task = []
        ac_model_rest = []
        
        # calculate data dimensions
        n_runs = len(test_data_task[model_idx])
        n_nodes = test_data_task[model_idx][0]['hidden_activity'][0].shape[1]
        n_time_task = test_data_task[model_idx][0]['hidden_activity'][0].shape[0]
        n_lags_task = n_time_task-1
        n_lags_rest = 50
        
        # runs loop
        for run in range(n_runs):
            
            # initialize arrays of AC, ACW, and ACA values for this run
            ac_run_task = dict()
            ac_run_task['ac'] = np.zeros((n_nodes,n_lags_task,n_trials))
            for k in dynamic_keys:
                ac_run_task[k] = np.zeros((n_nodes,n_trials))
            
            ac_run_rest = dict()
            ac_run_rest['ac'] = np.zeros((n_nodes,n_lags_rest,))
            for k in dynamic_keys:
                ac_run_rest[k] = np.zeros((n_nodes,))
            
            # nodes loop
            for node in range(n_nodes):
                
                # get values for task trials
                for trial in range(n_trials):
                    ts = test_data_task[model_idx][run]['hidden_activity'][trial][:,node]
                    ac, acw_50, acw_0, aca_50, aca_0 = compute_ac_acw_aca(ts, nlags=n_lags_task)
                    ac_run_task['ac'][node,:,trial] = ac
                    for k in dynamic_keys:
                        ac_run_task[k][node,trial] = globals()[k]
                
                # get values for rest
                ts = test_data_rest[model_idx][run]['hidden_activity'][:,node]
                ac, acw_50, acw_0, aca_50, aca_0 = compute_ac_acw_aca(ts, nlags=n_lags_rest)
                ac_run_rest['ac'][node,:] = ac
                for k in dynamic_keys:
                    ac_run_rest[k][node] = globals()[k]
            
            # append this run's data to this model's list
            ac_model_task.append(ac_run_task)
            ac_model_rest.append(ac_run_rest)
        
        # append this model's list to the model-wise list
        ac_data_task.append(ac_model_task)
        ac_data_rest.append(ac_model_rest)

    # ----------------------------------------------------------------------------------------- #
    # Plot ACF
    # ----------------------------------------------------------------------------------------- #

    # figure params
    size_scale = 2
    font_size = int(5 * size_scale)
    n_fig_columns = min((3, n_tasks))
    n_fig_rows = math.ceil( n_tasks / n_fig_columns )
    fig_size_w = 2.5*n_fig_columns*size_scale
    fig_size_h = 2*n_fig_rows*size_scale

    plotting_data = [
        [ac_data_task, 'task'],
        [ac_data_rest, 'rest']
    ]

    # begin plotting loop
    for i in range(len(plotting_data)):

        # for the full AC plots, we will create one subplot per task/kernel combination
        for kernel_idx in range(n_kernels):
            
            # create subplots
            f_ac, _ = plt.subplots(n_fig_rows, n_fig_columns, figsize=(fig_size_w,fig_size_h),
                                squeeze=True, sharex=True, sharey=True)
            plt.rcParams['font.size'] = font_size
            
            # models loop
            for model_idx in range(n_models):
                if model_params.loc[model_idx, 'kernel_index'] == kernel_idx:
                    task_index = model_params.loc[model_idx, 'task_index']
                    
                    # add a y=0 line for reference
                    f_ac.axes[task_index].axhline(y=0, color='k', linewidth=1.)
                    
                    # initialize the y-data matrix
                    this_data = plotting_data[i][0]
                    n_lags = this_data[model_idx][0]['ac'].shape[1]
                    y_plot = np.zeros((n_nodes,n_lags,n_runs)) 
                    
                    # populate matrix with mean across trials for each node and each run
                    for run in range(n_runs):
                        run_data = this_data[model_idx][run]['ac']
                        if run_data.ndim == 3:
                            y_plot[:,:,run] = np.nanmean(run_data, axis=2)
                        else:
                            y_plot[:,:,run] = run_data
                    
                    # calculate mean across runs
                    y_plot = np.squeeze(np.nanmean(y_plot, axis=2))
                    
                    # plot
                    for node in range(n_nodes):
                        f_ac.axes[task_index].plot(y_plot[node,:], color=colors[node,:], linewidth=0.5, alpha=1)
                    f_ac.axes[task_index].set_title(model_params.loc[model_idx,'task_label'])

            # polish plots
            f_ac.axes[0].set_ylim((-1,1))
            f_ac.suptitle(kernel_labels[kernel_idx] + ' (' + plotting_data[i][1] + ')')
            sns.despine(fig=f_ac, offset=0, trim=False, left=False, right=True, top=True, bottom=False)
            f_ac.tight_layout()
            # plt.show()
            
            # save svg
            if save_figs:
                f_ac.savefig(file_path_ac[i][kernel_idx], dpi=300, bbox_inches='tight', pad_inches=0.01)

    # ----------------------------------------------------------------------------------------- #
    # Plot performance
    # ----------------------------------------------------------------------------------------- #

    if not check_only:
        
        # import colors
        color_palette = utils.get_my_colors(cat_trio=False, as_list=True)

        # initialise figures
        size_scale = 1.5
        font_size = int(10 * size_scale)
        n_fig_columns = min((3, n_tasks))
        n_fig_rows = math.ceil( n_tasks / n_fig_columns )
        fig_size_w = 3.5*n_fig_columns*size_scale
        fig_size_h = 3.5*n_fig_rows*size_scale
        f_plots, ax_plots = plt.subplots(n_fig_rows, n_fig_columns, figsize=(fig_size_w,fig_size_h),
                            squeeze=True, sharex=False, sharey=True)
        f_violin, ax_violin = plt.subplots(n_fig_rows, n_fig_columns, figsize=(fig_size_w,fig_size_h),
                            squeeze=True, sharex=False, sharey=True)
        f_diff = plt.figure(figsize=(3.5*size_scale,3.5*size_scale))
        plt.rcParams['font.size'] = font_size
        xlims_plots = np.zeros(n_tasks)
        violin_bodies = []
        all_means = [{} for _ in range(n_tasks)]

        for model_idx in np.arange(n_models):
            
            # get model details
            this = model_params.iloc[model_idx]
            
            try:
                
                # get data
                test_accuracy_all_runs = model_accuracy_all[model_idx]
                
                # extract accuracy from each run
                test_accuracy = np.zeros((this.n_runs,test_accuracy_all_runs[0].shape[0]-1))
                for run in np.arange(this.n_runs):
                    test_accuracy[run,:] = test_accuracy_all_runs[run][1:] * 100

                n_epochs_actual = test_accuracy.shape[1] * 100
                n_logged_epochs = int(n_epochs_actual / 100)
                if n_logged_epochs > 500:
                    n_logged_epochs = 125
                    n_epochs_actual = n_logged_epochs*100
                    test_accuracy = test_accuracy[:,:n_logged_epochs]
                x_step = int(n_epochs_actual / (n_logged_epochs))
                x = np.arange(x_step, n_epochs_actual + x_step, x_step)

                # compute mean and ci across runs
                accuracy_mean_runs = test_accuracy.mean(axis=0)
                accuracy_std_runs = test_accuracy.std(axis=0)
                ci = 1.96 * (accuracy_std_runs / np.sqrt(test_accuracy.shape[0]))
                ci_lower = accuracy_mean_runs - ci
                ci_upper = accuracy_mean_runs + ci
                
                # compute mean of each run across time
                accuracy_mean_epochs = test_accuracy.mean(axis=1)
                
                # add values to sa_vs_euc list
                all_means[this.task_index][this.kernel_index] = accuracy_mean_runs

                # plot mean accuracy vs. time
                f_plots.axes[this.task_index].plot(x, accuracy_mean_runs, color=color_palette[this.kernel_index], label=str(this.kernel_label))
                f_plots.axes[this.task_index].fill_between(x, ci_lower, ci_upper, color=color_palette[this.kernel_index], alpha=0.1)
                xlims_plots[this.task_index] = np.max((xlims_plots[this.task_index],n_epochs_actual))
                
                # plot time-averaged accuracy as violin plot
                violin_parts = f_violin.axes[this.task_index].violinplot(dataset=accuracy_mean_epochs, positions=[this.kernel_index], showmedians=True)
                for partname in ('cbars','cmins','cmaxes','cmedians'):
                    vp = violin_parts[partname]
                    vp.set_edgecolor(color_palette[this.kernel_index])
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(color_palette[this.kernel_index])
                    pc.set_edgecolor(color_palette[this.kernel_index])
                if this.task_index == 0:
                    violin_bodies.append(violin_parts['bodies'][0])
                
            except:
                
                print("An error occurred:", sys.exc_info()[0])
                print("Exception message:", sys.exc_info()[1])

        # set title and labels for line plots
        xlim_method = 'auto' # 'auto' or x
        for task_index in np.arange(n_tasks):
            # f.axes[task_index].set_xlabel('Epochs', fontsize=font_size)
            # if task_index == 0: f.axes[task_index].set_ylabel('Test Accuracy (%)', fontsize=font_size)
            if xlim_method == 'auto':
                xmax = xlims_plots[task_index]
            else:
                xmax = xlim_method
            f_plots.axes[task_index].set_xlim([0, xmax+10])
            f_plots.axes[task_index].set_ylim([0, 105])
            f_plots.axes[task_index].set_xticks(np.arange(0,xmax+10,int(xmax/4)))
            f_plots.axes[task_index].set_yticks(np.arange(0,100+1,20))
            f_plots.axes[task_index].tick_params(bottom=True,left=True)
            if task_index == 0:
                f_plots.axes[task_index].legend(loc='lower right')
            f_plots.axes[task_index].set_title(utils.get_task_label(task_names[task_index]), fontsize=font_size)
            f_plots.axes[task_index].grid()
        f_plots.text(0.51, 0.0, 'Epoch', ha='center', va='center')
        f_plots.text(0.0, 0.5, 'Test Accuracy (%)', ha='center', va='center', rotation='vertical')
        
        # set title and labels for violin plots
        for task_index in np.arange(n_tasks):
            f_violin.axes[task_index].set_ylim([0, 105])
            x_ticks = np.arange(0,n_kernels+1)
            f_violin.axes[task_index].set_xticks(x_ticks)
            f_violin.axes[task_index].set_xticks((-1,))
            f_violin.axes[task_index].set_xticklabels([' '])
            f_violin.axes[task_index].set_yticks(np.arange(0,100+1,20))
            f_violin.axes[task_index].tick_params(bottom=True,left=True)
            if task_index == 0:
                f_violin.axes[task_index].legend(violin_bodies, kernel_labels, loc='lower right')
            f_violin.axes[task_index].set_title(utils.get_task_label(task_names[task_index]), fontsize=font_size)
            f_violin.axes[task_index].grid()
        f_violin.text(0.0, 0.5, ' ', ha='center', va='center', rotation='vertical')
        f_violin.text(0.0, 0.5, 'Learning Speed (a.u.)', ha='center', va='center', rotation='vertical')
        
        # # plot sa vs. eucl means
        # diff_vals = np.zeros((n_tasks,all_means[0][0].shape[0]))
        # sa_idx = kernel_labels.index('Masked S-A')
        # eucl_idx = kernel_labels.index('Masked Eucl.')
        # y_min = 0
        # y_max = 0
        # for task_index in range(n_tasks):
        #     y = all_means[task_index][sa_idx] - all_means[task_index][eucl_idx]
        #     y_min = min(y_min, min(y))
        #     y_max = max(y_max, max(y))
        #     plt.plot(x, y, figure=f_diff, linestyle='-', linewidth=1., color=[0,0,0], alpha=0.2)
        #     diff_vals[task_index,:] = y
        # plt.plot(x, np.mean(diff_vals,axis=0), figure=f_diff, linestyle='-', linewidth=2., color=[0,0,0], alpha=1.)
        # y_lim = 1.1*(max(np.abs(y_min),np.abs(y_max)))
        # ax = f_diff.axes[0]
        # ax.set_ylim([-y_lim,y_lim])
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Accuracy difference (%)')

        
        sns.despine(fig=f_plots, offset=0, trim=False, left=False, right=True, top=True, bottom=False)
        sns.despine(fig=f_violin, offset=0, trim=False, left=False, right=True, top=True, bottom=False)
        # sns.despine(fig=f_diff, offset=0, trim=False, left=False, right=True, top=True, bottom=False)
        
        f_plots.tight_layout()
        f_violin.tight_layout()
        # plt.show()
        
        # save svg
        if save_figs:
            f_plots.savefig(file_path_plots, dpi=300, bbox_inches='tight', pad_inches=0.01)
            f_violin.savefig(file_path_violin, dpi=300, bbox_inches='tight', pad_inches=0.01)
            # f_diff.savefig(file_path_diff, dpi=300, bbox_inches='tight', pad_inches=0.01)