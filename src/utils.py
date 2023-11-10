import numpy as np
import scipy as sp
from scipy import signal
import torch


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
    task = config['task']
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    decision = config['env_kwargs']['timing']['decision']

    # RNN model and training parameters
    rnn_model = config['rnn_model']
    hidden_size = config['hidden_size']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']
    lr = config['lr']
    mask_weights = config['mask_weights']
    n_io = config['n_io']

    # regularization parameters
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']
    kernel_type = config['kernel_type']
    if kernel_type == 'static':
        kernel_std_frac = config['kernel_std_frac']
    elif kernel_type == 'comet':
        kernel_std_frac = config['kernel_std_frac']
        comet_buffer_frac = config['comet_buffer_frac']
        comet_tail_frac = config['comet_tail_frac']

    if kernel_type is None or kernel_type == 'sa_axis' or kernel_type == 'euclidean':
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_io, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type)
    elif kernel_type == 'static':
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_io, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type, kernel_std_frac)
    elif kernel_type == 'comet':
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_io, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type, kernel_std_frac, comet_buffer_frac, comet_tail_frac)

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
