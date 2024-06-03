import numpy as np
import scipy as sp
from scipy import signal
from scipy.spatial import distance
import torch
from sklearn.decomposition import PCA

def normalize_x(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_kernel(hidden_size=200, location=0, kernel_std_frac=0.2):
    kernel_size = hidden_size - 1  # kernel size
    kernel_size_half = kernel_size//2  # half size of kernel
    kernel_std = int((kernel_size-1) * kernel_std_frac)
    kernel_1d = signal.gaussian(kernel_size, std=kernel_std).reshape(kernel_size, 1)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel = np.zeros((hidden_size, hidden_size))
    kernel[location, location] = 1

    if (location - kernel_size_half) < 0:
        cut = np.abs((location - kernel_size_half))
        kernel_2d = kernel_2d[cut:, :][:, cut:]
        kernel[:(kernel_size-cut), :(kernel_size-cut)] = kernel_2d
    elif (location + kernel_size_half) >= hidden_size:
        cut = kernel_size - ((location + kernel_size_half) - hidden_size)
        kernel_2d = kernel_2d[:cut, :][:, :cut]
        kernel[hidden_size-cut:, hidden_size-cut:] = kernel_2d
    else:
        kernel[location-kernel_size_half:location+kernel_size_half+1, location-kernel_size_half:location+kernel_size_half+1] = kernel_2d

    return kernel


def map_kernel_to_epochs(n_epochs=1000, hidden_size=200):
    kernel_location = np.linspace(0, hidden_size, n_epochs, endpoint=False)
    kernel_location = np.floor(kernel_location).astype(int)

    return kernel_location


def build_reg_ken(n_epochs=1000, hidden_size=200, kernel_std_frac=0.2, type='spotlight', comet_buffer_frac=0.1, comet_tail_frac=0.25):
    kernel_location = map_kernel_to_epochs(n_epochs=n_epochs, hidden_size=hidden_size)
    # kernel_location = np.flip(kernel_location)

    kernel = np.zeros((hidden_size, hidden_size, n_epochs))
    if type == 'additive' or type == 'comet':
        kernel_lagged = np.zeros((hidden_size, hidden_size, n_epochs))
        buffer = int(n_epochs * comet_buffer_frac)

    for epoch in range(n_epochs):
        if type == 'spotlight':
            kernel[:, :, epoch] = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch], kernel_std_frac=kernel_std_frac)
        elif type == 'additive' or type == 'comet':
            if epoch == 0:
                kernel[:, :, epoch] = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch], kernel_std_frac=kernel_std_frac)
                previous_kernel = kernel[:, :, epoch].copy()
            else:
                new_kernel = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch], kernel_std_frac=kernel_std_frac)
                new_kernel -= previous_kernel
                new_kernel[new_kernel < 0] = 0
                kernel[:, :, epoch] = previous_kernel + new_kernel
                previous_kernel = kernel[:, :, epoch].copy()

            if (epoch - buffer) > 0:
                kernel_lagged[:, :, epoch] = kernel[:, :, epoch-buffer].copy()

    if type == 'spotlight' or type == 'additive':
        return kernel
    elif type == 'comet':
        return kernel - (kernel_lagged * comet_tail_frac)


def get_p_val_string(p_val):
    if p_val == 0.0:
        p_str = "-log10($\mathit{:}$)>25".format('{p}')
    elif p_val < 0.05:
        p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
    # elif p_val < 0.001:
    #     p_str = '$\mathit{:}$ < 0.001'.format('{p}')
    # elif p_val >= 0.001 and p_val < 0.05:
    #     p_str = '$\mathit{:}$ < 0.05'.format('{p}')
    else:
        p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

    return p_str


def fix_labels(labels, decision=4, trim=2):
    if trim == 0:
        return labels
    else:
        labels_out = labels.copy()
        batch_size = labels.shape[1]
        cut = decision - trim

        x = labels_out != 0
        x_pad = np.zeros((cut, batch_size)).astype(bool)
        y = np.append(x[cut:, :], x_pad, axis=0)
        xy = x*y
        labels_out[xy] = 0

        return labels_out


def bandpower(ts, fs, fmin, fmax):
    """
    Helper function for compute_rlfp.

    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series
    fs : np.float
        sampling frequency
    fs : np.fmin
        minimum frequency of interest
    fs : np.fmax
        maximum frequency of interest
    Returns
    -------
    rlfp : np.float
        relative low frequency power
    """

    f, Pxx = sp.signal.periodogram(ts, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1

    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def compute_rlfp(ts, tr, low=None, high=None, num_bands=5, band_of_interest=1):
    """
    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series
    tr : np.float
        tr in seconds

    Returns
    -------
    rlfp : np.float
        relative low frequency power
    """

    sample_freq = 1 / tr
    nyq_freq = sample_freq / 2

    y = sp.stats.zscore(ts)

    if low is None and high is None:
        band_intervals = np.linspace(0, nyq_freq, num_bands + 1)
    else:
        band_intervals = np.linspace(low, high, num_bands + 1)

    band_freq_range = band_intervals[band_of_interest - 1:band_of_interest + 1]

    return bandpower(y, sample_freq, band_freq_range[0], band_freq_range[1])


def make_matrix_masks(input_weight_mask, output_weight_mask, bystanders):
    hidden_size = len(input_weight_mask)
    # diagonal
    # input-input
    input_input = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    input_input[input_weight_mask, :] = True
    input_input[:, bystanders] = False
    input_input[:, output_weight_mask] = False

    # bystanders-bystanders
    bystanders_bystanders = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    bystanders_bystanders[bystanders, :] = True
    bystanders_bystanders[:, input_weight_mask] = False
    bystanders_bystanders[:, output_weight_mask] = False

    # output-output
    output_output = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    output_output[output_weight_mask, :] = True
    output_output[:, input_weight_mask] = False
    output_output[:, bystanders] = False

    # direct bottom-up connections (input to output)
    output_input = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    output_input[input_weight_mask, :] = True
    output_input[:, ~output_weight_mask] = False

    # direct top-down connections (output to input)
    input_output = output_input.T

    # input to bystanders
    bystanders_input = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    bystanders_input[:, bystanders] = True
    bystanders_input[bystanders, :] = False
    bystanders_input[output_weight_mask, :] = False

    # bystanders to inputs
    input_bystanders = bystanders_input.T

    # output to bystanders
    bystanders_output = torch.zeros((hidden_size, hidden_size), dtype=torch.bool)
    bystanders_output[:, bystanders] = True
    bystanders_output[bystanders, :] = False
    bystanders_output[input_weight_mask, :] = False

    # bystanders to outputs
    output_bystanders = bystanders_output.T

    # bidirectional masks
    input_output_symmetric = output_input + input_output
    input_bystanders_symmetric = bystanders_input + input_bystanders
    output_bystanders_symmetric = bystanders_output + output_bystanders

    masks = dict()
    # vector masks
    masks['input_weight_mask'] = input_weight_mask
    masks['output_weight_mask'] = output_weight_mask
    masks['bystanders'] = bystanders
    # matrix masks
    masks['input_input'] = input_input
    masks['bystanders_bystanders'] = bystanders_bystanders
    masks['output_output'] = output_output

    masks['input_output'] = input_output
    masks['output_input'] = output_input
    masks['input_bystanders'] = input_bystanders
    masks['bystanders_input'] = bystanders_input
    masks['output_bystanders'] = output_bystanders
    masks['bystanders_output'] = bystanders_output

    masks['input_output_symmetric'] = input_output_symmetric
    masks['input_bystanders_symmetric'] = input_bystanders_symmetric
    masks['output_bystanders_symmetric'] = output_bystanders_symmetric

    return masks


def get_weight_masks(hidden_size=200, n_io=30):
    input_idx = n_io
    output_idx = int(hidden_size-n_io)

    input_weight_mask = np.zeros((hidden_size,)).astype(bool)
    input_weight_mask[:input_idx] = True
    output_weight_mask = np.zeros((hidden_size, )).astype(bool)
    output_weight_mask[output_idx:] = True
    bystanders = ~(input_weight_mask + output_weight_mask)

    masks = make_matrix_masks(input_weight_mask, output_weight_mask, bystanders)

    return masks


def get_weight_masks_schaefer(roi_names, input_system='Vis', output_system='Default'):
    hidden_size = len(roi_names)
    input_weight_mask = np.zeros((hidden_size,)).astype(bool)
    output_weight_mask = np.zeros((hidden_size,)).astype(bool)
    for roi in np.arange(hidden_size):
        if input_system in roi_names[roi]:
            input_weight_mask[roi] = True
        if output_system in roi_names[roi]:
            output_weight_mask[roi] = True
    bystanders = ~(input_weight_mask + output_weight_mask)

    masks = make_matrix_masks(input_weight_mask, output_weight_mask, bystanders)

    return masks


def get_file_str(config):
    # task parameters
    task = config['task_with_modifier']
    seq_len = config['seq_len']

    # RNN model and training parameters
    rnn_model = config['rnn_model']
    hidden_size = config['hidden_size']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']
    mask_weights = config['mask_weights']
    n_io = config['n_io']

    # regularization parameters
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']
    kernel_type = config['kernel_type']
    
    # create name string
    file_str = 'task-{:}-{:}_' \
                'model-{:}-{:}-{:}-{:}-{:}-{:}_' \
                'wmask-{:}-{:}_' \
                'reg-{:}-{:}-{:}' \
        .format(task, seq_len,
                rnn_model, hidden_size, batch_size, lr, n_runs, n_epochs,
                mask_weights, n_io,
                reg_type, reg_weight, kernel_type)

    return file_str


def autocorr(data, max_lag=1000, lag_step=1):
    """ Calculate the signal autocorrelation (lagged correlation)

    Parameters
    ----------
    data : array 1D
        Time series to compute autocorrelation over.
    max_lag : int (default=1000)
        Maximum delay to compute AC, in samples of signal.
    lag_step : int (default=1)
        Step size (lag advance) to move by when computing correlation.

    Returns
    -------
    AC_timepoints : array, 1D
        Time points (in samples) at which correlation was computed.
    AC : array, 1D
        Time lagged (auto)correlation.

    """
    ###
    # have FT implementation
    ###
    AC_timepoints = np.arange(0, max_lag, lag_step)
    AC = np.zeros(len(AC_timepoints))
    AC[0] = np.sum((data - np.mean(data))**2)
    for ind, lag in enumerate(AC_timepoints[1:]):
        AC[ind + 1] = np.sum((data[:-lag] - np.mean(data[:-lag]))
                             * (data[lag:] - np.mean(data[lag:])))

    return AC_timepoints, AC / AC[0]

# The exponential decay function
def exp_decay(x, tau, init):
    return init*np.e**(-x/tau)


def compute_fc(ts):
    """
    Parameters
    ----------
    ts : np.array (n_timepoints,n_parcels)
        time series

    Returns
    -------
    fc : np.array (n_parcels,n_parcels)
        functional connectivity matrix
    """

    fc = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(fc, np.nan)
    fc = np.arctanh(fc)
    np.fill_diagonal(fc, 1)

    return fc

def compute_pc_var(hidden_activity, n_components=3, normalize=True):

    [n_trials, n_timepoints, n_nodes] = hidden_activity.shape
    if n_components > 0:
        pca = PCA(n_components=n_components)
        [n_trials, n_timepoints, n_nodes] = hidden_activity.shape
        activity_reshape = np.reshape(hidden_activity, (n_trials*n_timepoints, n_nodes))
        pca.fit(activity_reshape)

        hidden_activity_pc = np.zeros((n_trials, n_timepoints, n_components))
        for trial in np.arange(n_trials):
            hidden_activity_pc[trial] = pca.transform(hidden_activity[trial])
    else:
        hidden_activity_pc = np.mean(hidden_activity, axis=-1)
        hidden_activity_pc = hidden_activity_pc[:, :, np.newaxis]

    hidden_activity_pc_var = np.zeros((n_timepoints, hidden_activity_pc.shape[-1]))
    for timepoint in np.arange(n_timepoints):
        pc_var = np.var(hidden_activity_pc[:, timepoint], axis=0)
        if timepoint > 0 and normalize:
            hidden_activity_pc_var[timepoint] = pc_var / pc_var_prev
        else:
            hidden_activity_pc_var[timepoint] = pc_var
        pc_var_prev = pc_var.copy()
    hidden_activity_pc_var[np.isnan(hidden_activity_pc_var)] = 0
    hidden_activity_pc_var[np.isinf(hidden_activity_pc_var)] = 0

    return hidden_activity_pc_var

def get_my_colors(normalize=True, as_list=False, cat_trio=False):
    # color palette (RGB / HEX):
    # raspberry blush: rgba(234,86,81,255) / #ea5651
    # conch shell: rgba(238,186,169,255) / #eebaa9
    # cinnamon: rgba(165,74,54,255) / #a54a36
    # wenge: rgba(63,44,41,255) / #3f2c29
    # savannah green: rgba(194,158,62,255) / #c29e3e
    # new age: rgba(217,206,209,255) / #d9ced1
    # starry night blue: rgba(48,65,121,255) / #304179
    # north sea green: rgba(0,111,116,255) / #006f74
    my_colors = dict()
    my_colors['raspberry_blush'] = [234, 86, 81]
    my_colors['starry_night_blue'] = [48, 65, 121]
    my_colors['north_sea_green'] = [0, 111, 116]
    if not cat_trio:
        my_colors['conch_shell'] = [238, 186, 169]
        my_colors['cinnamon'] = [165, 74, 54]
        my_colors['wenge'] = [63, 44, 41]
        my_colors['savannah_green'] = [194, 158, 62]
        my_colors['new_age'] = [217, 206, 209]

    if normalize:
        for key in my_colors.keys():
            my_colors[key] = [color / 255 for color in my_colors[key]]

    if as_list:
        my_colors = list(my_colors.values())

    return my_colors


def get_slopes(feature, segment_size=20):
    n_runs, n_epochs = feature.shape
    n_epochs_trim = n_epochs - segment_size
    slopes = np.zeros((n_runs, n_epochs))
    slopes[:] = np.nan

    x = np.arange(segment_size)
    y = sp.stats.zscore(feature, axis=1)

    for i in np.arange(n_runs):
        for j in np.arange(n_epochs_trim):
            # results = sp.stats.linregress(x, y[i, j:(j+segment_size)])
            # slopes[i, j] = results.slope
            results = sp.stats.pearsonr(x, y[i, j:(j+segment_size)])
            slopes[i, j] = results[0]

    return slopes


def get_n_threads(threads_in=None, verbose=0):
    
    import os, multiprocessing
    
    if verbose: print(' ')

    cpu_count = multiprocessing.cpu_count()
    if verbose: print('CPUs available = ' + str(cpu_count))

    omp_limit = os.environ.get('OMP_NUM_THREADS')
    if omp_limit:
        if verbose: print('Openmp limit   = ' + omp_limit)
        omp_limit = int(omp_limit)
    else:
        if verbose: print('Openmp limit   = ' + str(omp_limit))
        omp_limit = cpu_count

    if verbose: print('User requested = ' + str(threads_in))
    if threads_in == None or threads_in < 1:
        n_threads = min(cpu_count, omp_limit)
    else:
        n_threads = min(cpu_count, threads_in)
    if verbose: print('Will use ' + str(n_threads) + ' thread(s).')
    
    if verbose: print(' ')
    
    return n_threads


def get_device(device_opt):
    cuda_avail = torch.cuda.is_available()
    if device_opt == 'None':
        device = torch.device('cuda' if cuda_avail else 'cpu')
    else:
        if device_opt == 'cuda':
            if cuda_avail:
                device = torch.device('cuda')
            else:
                print('CUDA not availble!')
                device = torch.device('cpu')
        elif device_opt == 'cpu':
            device = torch.device('cpu')
        else:
            print('Device choice not recognized!')
            device = torch.device('cpu')
    print('\nDevice: ' + device.type + '.\n')
    return device


def get_brainmap_distance(brain_map):
    if brain_map.ndim == 1 or brain_map.shape[1] == 1:
        brain_map = brain_map.reshape(-1,1)
    distance_matrix = distance.squareform(distance.pdist(brain_map, 'euclidean'))

    return distance_matrix


def get_n_io(mask_weights=True, hidden_size=100):
    if mask_weights and hidden_size == 50:
        n_io = '9-13' # Default
        # n_io = '9-4' # Cont
    elif mask_weights and hidden_size == 100:
        n_io = '14-27' # Default
        # n_io = '14-13' # Cont
    elif mask_weights and hidden_size == 200:
        n_io = '31-52' # Default
        # n_io = '31-22' # Cont
    else:
        n_io = 'na'
    return n_io


def get_task_modifier(task):
    delimiter = '-'
    supported_modifiers = [
        'ND', 'MD', 'LD',
        'Del00_Dec1', 'Del00_Dec3', 'Del00_Dec5',
        'Del05_Dec1', 'Del05_Dec3', 'Del05_Dec5', 
        'Del10_Dec1', 'Del10_Dec3', 'Del10_Dec5',
        'Dec1', 'Dec3'
        ]
    task_subs = task.split(delimiter)
    if str(task_subs[-1]) in supported_modifiers:
        modifier = str(task_subs[-1])
        task = delimiter.join(task_subs[:-1])
    else:
        task = task
        modifier = ''
    return task, modifier


def check_if_supported(task, modifier):
    supported_tasks = [
        'ContextDecisionMaking-v0-ND',
        'ContextDecisionMaking-v0-MD',
        'ContextDecisionMaking-v0-LD',
        
        'DelayComparison-v0',
        
        'DelayMatchCategory-v0',
        
        'DelayMatchSample-v0',
        
        'DelayMatchSampleDistractor1D-v0',
        
        'DelayPairedAssociation-v0',
        
        'DualDelayMatchSample-v0',
        
        'GoNogo-v0-Del00_Dec1',
        'GoNogo-v0-Del00_Dec3',
        'GoNogo-v0-Del00_Dec5',
        'GoNogo-v0-Del05_Dec1',
        'GoNogo-v0-Del05_Dec3',
        'GoNogo-v0-Del05_Dec5',
        'GoNogo-v0-Del10_Dec1',
        'GoNogo-v0-Del10_Dec3',
        'GoNogo-v0-Del10_Dec5',
        
        'GoNogoVD-v0-Dec1',
        'GoNogoVD-v0-Dec3',
        
        'MultiSensoryIntegration-v0',
        
        'PerceptualDecisionMaking-v0-ND',
        'PerceptualDecisionMaking-v0-MD',
        'PerceptualDecisionMaking-v0-LD',
        
        'PerceptualDecisionMakingDelayResponse-v0',
    ]
    if modifier != '' and task + '-' + modifier not in supported_tasks:
        raise ValueError("The combination of task '{:}' and modifier '{:}' is not supported.\n\nSupported tasks: \n{:}\n"\
            .format(task, modifier, '\n'.join(supported_tasks)))
    if modifier == '' and task not in supported_tasks:
        raise ValueError("The task '{:}' is not supported.\n\nSupported tasks: \n{:}\n"\
            .format(task, '\n'.join(supported_tasks)))


def get_seq_len_and_timing(task, modifier='', seq_len_multi=5):
    
    other = {} # other timing arguments used in seq_len calculation, not passed to ngym
    
    if task == 'ContextDecisionMaking-v0':
        if modifier == '':
            timing = {'fixation': 300, 'stimulus': 1000, 'decision': 300}
        else:
            if modifier == 'ND':
                delay = 0
            elif modifier == 'MD':
                delay = 1500
            elif modifier == 'LD':
                delay = 3000
            timing = {'fixation': 300, 'stimulus': 1000, 'delay': delay, 'decision': 300}
    
    elif task == 'DelayComparison-v0':
        timing = {'fixation': 300, 'stimulus1': 500, 'delay': 1000, 'stimulus2': 500, 'decision': 300}
    
    elif task == 'DelayMatchCategory-v0':
        timing = {'fixation': 300, 'sample': 700, 'first_delay': 1000,  'test': 700}
    
    elif task == 'DelayMatchSample-v0':
        timing = {'fixation': 300, 'sample': 500, 'delay': 1000, 'test': 500, 'decision': 300}
    
    elif task == 'DelayMatchSampleDistractor1D-v0':
        timing = {'fixation': 300, 'sample': 500, 'delay1': 1000, 'test1': 500, 'delay2': 1000, 'test2': 500, 'delay3': 1000, 'test3': 500 }
    
    elif task == 'DelayPairedAssociation-v0':
        timing = {'fixation': 300, 'stim1': 1000, 'delay_btw_stim': 1000, 'stim2': 1000, 'delay_aft_stim': 1000, 'decision': 300}
    
    elif task == 'DualDelayMatchSample-v0':
        timing = {'fixation': 300, 'sample': 500, 'delay1': 500, 'cue1': 500, 'test1': 500, 'delay2': 500, 'cue2': 500, 'test2': 500}
    
    elif task == 'GoNogo-v0':
        if modifier == 'Del00_Dec1':
            delay = 0
            decision = 100
        elif modifier == 'Del00_Dec3':
            delay = 0
            decision = 300
        elif modifier == 'Del00_Dec5':
            delay = 0
            decision = 500
        elif modifier == 'Del05_Dec1':
            delay = 500
            decision = 100
        elif modifier == 'Del05_Dec3':
            delay = 500
            decision = 300
        elif modifier == 'Del05_Dec5':
            delay = 500
            decision = 500
        elif modifier == 'Del10_Dec1':
            delay = 1000
            decision = 100
        elif modifier == 'Del10_Dec3':
            delay = 1000
            decision = 300
        elif modifier == 'Del10_Dec5':
            delay = 1000
            decision = 500
        timing = {'fixation': 300, 'stimulus': 500, 'delay': delay, 'decision': decision}
        
    elif task == 'GoNogoVD-v0':
        if modifier == 'Dec1':
            decision = 100
        elif modifier == 'Dec3':
            decision = 300
        timing = {'fixation': 300, 'stimulus': 500, 'decision': decision}
        other = {'delay': 600}
    
    elif task == 'MultiSensoryIntegration-v0':
        timing = {'fixation': 300, 'stimulus': 1000, 'decision': 300}
    
    elif task == 'PerceptualDecisionMaking-v0':
        if modifier == 'ND':
            delay = 0
        elif modifier == 'MD':
            delay = 500
        elif modifier == 'LD':
            delay = 1000
        timing = {'fixation': 300, 'stimulus': 2000, 'delay': delay, 'decision': 300}
    
    elif task == 'PerceptualDecisionMakingDelayResponse-v0':
        timing = {'fixation': 300, 'stimulus': 1200, 'decision': 300}
        other = {'delay': 1600}
    
    else:
        raise ValueError("Task '{:}' is not supported.".format(task))
    
    seq_len_base = ( sum(timing.values()) + sum(other.values()) ) / 100
    seq_len = int( seq_len_base * seq_len_multi )
    
    return seq_len, timing


def get_task_label(task):
    if task == 'ContextDecisionMaking-v0':
        task_label = 'Context Decision Making (Ctx DM) - Var. Del.'
    elif task == 'ContextDecisionMaking-v0-ND':
        task_label = 'Context Decision Making (Ctx DM) - ND'
    elif task == 'ContextDecisionMaking-v0-MD':
        task_label = 'Context Decision Making (Ctx DM) - MD'
    elif task == 'ContextDecisionMaking-v0-LD':
        task_label = 'Context Decision Making (Ctx DM) - LD'
    elif task == 'DelayComparison-v0':
        task_label = 'Delayed Comparison (DC)'
    elif task == 'DelayMatchCategory-v0':
        task_label = 'Delayed Match to Category (DMC)'
    elif task == 'DelayMatchSample-v0':
        task_label = 'Delayed Match to Sample (DMS)'
    elif task == 'DelayMatchSampleDistractor1D-v0':
        task_label = 'Delayed Match to Sample with Distractors (DMS-D)'
    elif task == 'DelayPairedAssociation-v0':
        task_label = 'Delayed Paired Association (DPA)'
    elif task == 'DualDelayMatchSample-v0':
        task_label = 'Dual Delayed Match to Sample (DDMS)'
    elif task == 'GoNogo-v0-ND':
        task_label = 'Go/No-Go (GNG) - ND'
    elif task == 'GoNogo-v0-MD':
        task_label = 'Go/No-Go (GNG) - MD'
    elif task == 'GoNogo-v0-LD':
        task_label = 'Go/No-Go (GNG) - LD'
    elif task == 'GoNogo-v0-Del00_Dec1':
        task_label = 'Go/No-Go (GNG) - Del. 00 - Dec. 1'
    elif task == 'GoNogo-v0-Del00_Dec3':
        task_label = 'Go/No-Go (GNG) - Del. 00 - Dec. 3'
    elif task == 'GoNogo-v0-Del00_Dec5':
        task_label = 'Go/No-Go (GNG) - Del. 00 - Dec. 5'
    elif task == 'GoNogo-v0-Del05_Dec1':
        task_label = 'Go/No-Go (GNG) - Del. 05 - Dec. 1'
    elif task == 'GoNogo-v0-Del05_Dec3':
        task_label = 'Go/No-Go (GNG) - Del. 05 - Dec. 3'
    elif task == 'GoNogo-v0-Del05_Dec5':
        task_label = 'Go/No-Go (GNG) - Del. 05 - Dec. 5'
    elif task == 'GoNogo-v0-Del10_Dec1':
        task_label = 'Go/No-Go (GNG) - Del. 10 - Dec. 1'
    elif task == 'GoNogo-v0-Del10_Dec3':
        task_label = 'Go/No-Go (GNG) - Del. 10 - Dec. 3'
    elif task == 'GoNogo-v0-Del10_Dec5':
        task_label = 'Go/No-Go (GNG) - Del. 10 - Dec. 5'
    elif task == 'GoNogoVD-v0-Dec1':
        task_label = 'Go/No-Go (GNG) - Var. Del. - Dec. 1'
    elif task == 'GoNogoVD-v0-Dec3':
        task_label = 'Go/No-Go (GNG) - Var. Del. - Dec. 3'
    elif task == 'MultiSensoryIntegration-v0-ND':
        task_label = 'Multi Sensory Integration (MultSen DM) - ND'
    elif task == 'MultiSensoryIntegration-v0-MD':
        task_label = 'Multi Sensory Integration (MultSen DM) - MD'
    elif task == 'MultiSensoryIntegration-v0-LD':
        task_label = 'Multi Sensory Integration (MultSen DM) - LD'
    elif task == 'PerceptualDecisionMaking-v0-ND':
        task_label = 'Perceptual Decision Making (DM) - ND'
    elif task == 'PerceptualDecisionMaking-v0-MD':
        task_label = 'Perceptual Decision Making (DM) - MD'
    elif task == 'PerceptualDecisionMaking-v0-LD':
        task_label = 'Perceptual Decision Making (DM) - LD'
    elif task == 'PerceptualDecisionMakingDelayResponse-v0':
        task_label = 'Perceptual Decision Making (DM) - Var. Del.'
        
    return task_label


def get_kernel_label(kernel_type='None', mask_weights=False, reg_weight=0.0):
    if mask_weights:
        m = 'Masked '
    else:
        m = 'Unmasked '
    if kernel_type == 'sa_axis':
        kernel_label = m + 'S-A RNN'
    elif kernel_type == 'ut_axis':
        kernel_label = m + 'U-T RNN'
    elif kernel_type == 'sf_axis':
        kernel_label = m + 'S-F RNN'
    elif kernel_type == 'euclidean':
        kernel_label = m + 'Eucl. RNN'
    elif kernel_type == 'struct_conn':
        kernel_label = m + 'SC RNN'
    elif kernel_type == 'None':
        if reg_weight == 0:
            r = ' non-reg.'
        else:
            r = ' reg.'
        kernel_label = m + 'RNN' + r
    else:
        kernel_label = 'unknown'
    return kernel_label
    

def load_params_csv(model_params_csv):
    import pandas as pd
    model_params = pd.read_csv(model_params_csv, keep_default_na=False, na_values=['NaN'])
    kernel_labels = []
    for row in np.arange(len(model_params)):
        kernel_labels.append(get_kernel_label(kernel_type=model_params.kernel_type.iloc[row], \
                                                mask_weights=model_params.mask_weights.iloc[row], \
                                                reg_weight=model_params.reg_weight.iloc[row]))
    model_params['kernel_label'] = kernel_labels
    return model_params
