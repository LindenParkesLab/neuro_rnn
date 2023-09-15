import numpy as np
from scipy import signal


def normalize_x(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_kernel(hidden_size=200, location=0):
    kernel_size = hidden_size - 1  # kernel size
    kernel_size_half = kernel_size//2  # half size of kernel
    kernel_std = int((kernel_size-1)*.1)
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


def build_reg_ken(n_epochs=1000, hidden_size=200, type='spotlight', buffer_frac=0.1, kernel_frac=0.25):
    kernel_location = map_kernel_to_epochs(n_epochs=n_epochs, hidden_size=hidden_size)
    # kernel_location = np.flip(kernel_location)

    kernel = np.zeros((hidden_size, hidden_size, n_epochs))
    if type == 'additive' or type == 'comet':
        kernel_lagged = np.zeros((hidden_size, hidden_size, n_epochs))
        buffer = int(n_epochs * buffer_frac)

    for epoch in range(n_epochs):
        if type == 'spotlight':
            kernel[:, :, epoch] = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch])
        elif type == 'additive' or type == 'comet':
            if epoch == 0:
                kernel[:, :, epoch] = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch])
                previous_kernel = kernel[:, :, epoch].copy()
            else:
                new_kernel = get_kernel(hidden_size=hidden_size, location=kernel_location[epoch])
                new_kernel -= previous_kernel
                new_kernel[new_kernel < 0] = 0
                kernel[:, :, epoch] = previous_kernel + new_kernel
                previous_kernel = kernel[:, :, epoch].copy()

            if (epoch - buffer) > 0:
                kernel_lagged[:, :, epoch] = kernel[:, :, epoch-buffer].copy()

    if type == 'spotlight' or type == 'additive':
        return kernel
    elif type == 'comet':
        return kernel - (kernel_lagged * kernel_frac)

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
