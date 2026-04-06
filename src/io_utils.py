import os, argparse
import numpy as np
import pandas as pd

def parse_weight_init_value(value):
    """
    Parse weight initialization value from string or other formats.
    
    Parameters
    ----------
    value : str, float, list, or None
        Value to parse for weight initialization
        
    Returns
    -------
    None, float, tuple, or ndarray
        Parsed initialization value
    """
    if value is None or (isinstance(value, str) and value.lower() == 'none'):
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Handle space or comma separated values
        parts = value.replace(',', ' ').split()
        if len(parts) == 1:
            try:
                return float(parts[0])
            except ValueError:
                return None
        elif len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError:
                return None
    
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (ValueError, TypeError):
            return None
    
    # If it's already a numpy array, return as-is
    if isinstance(value, np.ndarray):
        return value
    
    return None


def parse_alpha_value(value):
    """
    Parse the alpha (continuous-time) parameter from various input formats.

    Parameters
    ----------
    value : float, int, str, or np.ndarray
        - Numeric (float/int): returned as a Python float.
        - String that converts to a float: returned as float.
        - String path to a .txt file: loaded via np.loadtxt and returned as a
          1-D np.ndarray (one value per line).
        - np.ndarray: returned as-is (must be 1-D).

    Returns
    -------
    float or np.ndarray
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"Alpha array must be 1-D, got shape {value.shape}")
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        path = os.path.expanduser(value)
        if os.path.isfile(path):
            arr = np.loadtxt(path)
            arr = np.atleast_1d(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"Alpha file '{path}' must contain a 1-D array, got shape {arr.shape}"
                )
            return arr
        raise ValueError(
            f"Cannot parse alpha value: '{value}' is not a float or a valid file path"
        )

    raise ValueError(f"Cannot parse alpha value of type {type(value)}: {value}")


def parse_boolean_value(value):
    """
    Parse boolean value from various string formats with explicit validation.
    
    Parameters
    ----------
    value : str, bool, int, or other
        Value to parse as boolean
        
    Returns
    -------
    bool
        Parsed boolean value
        
    Raises
    ------
    ValueError
        If value is not a recognized boolean format
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()
        
        # Explicit True values
        if value_lower in ['true', '1', 'yes', 'on']:
            return True
        
        # Explicit False values  
        elif value_lower in ['false', '0', 'no', 'off']:
            return False
        
        # Unrecognized string
        else:
            raise ValueError(f"Unrecognized boolean value: '{value}'. "
                           f"Valid options are: true/false, 1/0, yes/no, on/off (case insensitive)")
    
    # Handle integer cases
    if isinstance(value, int):
        if value == 1:
            return True
        elif value == 0:
            return False
        else:
            raise ValueError(f"Integer boolean values must be 0 or 1, got: {value}")
    
    # Handle None
    if value is None:
        return None
    
    # Anything else is an error
    raise ValueError(f"Cannot parse boolean from type {type(value)}: {value}")


def load_config_from_csv(params_csv, model_index):
    """
    Load model configuration from CSV file at specified index.
    All CSV columns are optional - any combination is supported.
    
    Parameters
    ----------
    params_csv : str
        Path to CSV file containing model parameters
    model_index : int
        Row index (0-based) of the model to load
        
    Returns
    -------
    dict
        Configuration dictionary with available parameters from CSV
    """
    try:
        # Load CSV file
        df = pd.read_csv(params_csv, keep_default_na=False, na_values=['NaN', 'nan', ''])
        
        if model_index >= len(df):
            raise ValueError(f"Model index {model_index} not found in CSV file (only {len(df)} rows)")
        
        # Get the model row
        model_row = df.iloc[model_index]
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file {params_csv}: {e}")
    
    config = {}
    
    # Define all possible parameter mappings with flexible column naming
    column_mapping = {
        # file locations
        'datadir': ['datadir', 'data_dir'],
        'outdir': ['outdir', 'out_dir', 'output_dir'],
        
        # task parameters
        'task': ['task', 'task_name'],
        'time_step': ['time_step'],
        'seq_len_multi': ['seq_len_multi', 'seq_len_multiplier'],
        
        # RNN model and training parameters
        'rnn_model': ['rnn_model', 'model_type'],
        'hidden_size': ['hidden_size'],
        'batch_size': ['batch_size'],
        'learning_rate': ['learning_rate', 'lr'],
        'n_runs': ['n_runs', 'num_runs'],
        'n_epochs': ['n_epochs', 'num_epochs'],
        'print_freq': ['print_freq'],
        'log_freq': ['log_freq'],
        'write_freq': ['write_freq'],
        'mask_weights': ['mask_weights'],
        'reservoir_mode': ['reservoir_mode'],
        'ridge_alpha': ['ridge_alpha'],
        'spatial_only_epochs': ['spatial_only_epochs'],
        'spectral_radius': ['spectral_radius'],
        'rec_noise': ['noise','rec_noise','node_noise','recurrent_noise'],
        
        # weight initialization parameters
        'init_ih_w': ['init_ih_w', 'init_input_hidden_weights'],
        'train_ih_w': ['train_ih_w', 'train_input_hidden_weights'],
        'init_ih_b': ['init_ih_b', 'init_input_hidden_bias'],
        'train_ih_b': ['train_ih_b', 'train_input_hidden_bias'],
        'init_hh_w': ['init_hh_w', 'init_hidden_hidden_weights'],
        'train_hh_w': ['train_hh_w', 'train_hidden_hidden_weights'],
        'init_hh_b': ['init_hh_b', 'init_hidden_hidden_bias'],
        'train_hh_b': ['train_hh_b', 'train_hidden_hidden_bias'],
        'init_ho_w': ['init_ho_w', 'init_hidden_output_weights'],
        'train_ho_w': ['train_ho_w', 'train_hidden_output_weights'],
        'init_ho_b': ['init_ho_b', 'init_hidden_output_bias'],
        'train_ho_b': ['train_ho_b', 'train_hidden_output_bias'],
        'allow_self_connections': ['allow_self_connections'],
        
        # regularization parameters
        'reg_type': ['reg_type', 'regularization_type'],
        'reg_weight': ['reg_weight', 'regularization_weight'],
        'kernel_type': ['kernel_type'],
        'kernel_normalization': ['kernel_normalization'],
        
        # continuous time parameter
        'alpha': ['alpha'],
        
        # device settings
        'device': ['device'],
        'n_threads': ['n_threads', 'num_threads']
    }
    
    # Extract values from CSV row with flexible column naming
    for config_key, possible_columns in column_mapping.items():
        value = None
        for col_name in possible_columns:
            if col_name in model_row.index and pd.notna(model_row[col_name]) and model_row[col_name] != '':
                value = model_row[col_name]
                break
        
        if value is not None:
            # Apply appropriate parsing based on parameter type
            if config_key in ['mask_weights', 'train_ih_w', 'train_ih_b', 'train_hh_w', 'train_hh_b',
                             'train_ho_w', 'train_ho_b', 'allow_self_connections', 'reservoir_mode']:
                config[config_key] = parse_boolean_value(value)
            elif config_key in ['init_ih_w', 'init_ih_b', 'init_hh_w', 'init_hh_b', 'init_ho_w', 'init_ho_b']:
                config[config_key] = parse_weight_init_value(value)
            elif config_key == 'alpha':
                config[config_key] = parse_alpha_value(value)
            elif config_key == 'kernel_type' and isinstance(value, str) and value.lower() == 'none':
                config[config_key] = None
            else:
                # Convert numpy scalar types to native Python types for PyTorch/stdlib compatibility
                if isinstance(value, np.integer):
                    config[config_key] = int(value)
                elif isinstance(value, np.floating):
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
    
    return config


def merge_configs(csv_config, args_config):
    """
    Merge CSV configuration with command line arguments.
    Command line arguments override values in CSV.
    
    Parameters
    ----------
    csv_config : dict
        Configuration loaded from CSV
    args_config : dict
        Configuration from command line arguments
        
    Returns
    -------
    dict
        Merged configuration with command line args taking precedence
    """
    # Start with CSV config
    merged = csv_config.copy()
    
    # Override with any explicitly provided command line arguments
    for key, value in args_config.items():
        merged[key] = value 
    
    return merged


# Sentinel value to distinguish "not provided" from "explicitly None"
_NOT_PROVIDED = object()

def get_args():
    '''Function to get args from command line and return the args.

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser(
        description='Train RNN models with either individual parameters or CSV configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with individual parameters
  python train_rnn.py --task PerceptualDecisionMaking-v0 --hidden_size 100 --n_runs 10
  
  # Train using CSV configuration
  python train_rnn.py --params_csv models.csv --model_index 0
  
  # Train using CSV with some parameter overrides (including explicit None)
  python train_rnn.py --params_csv models.csv --model_index 0 --n_runs 5 --kernel_type None
  
  # Use partial CSV (any combination of parameters)
  python train_rnn.py --params_csv partial_config.csv --model_index 0 --task CustomTask-v0 --device cuda
        """
    )

    # CSV input options
    csv_group = parser.add_argument_group('CSV Configuration')
    csv_group.add_argument('--params_csv', type=str, default=_NOT_PROVIDED,
                          help='Path to CSV file containing model parameters (all columns optional)')
    csv_group.add_argument('--model_index', type=int, default=_NOT_PROVIDED,
                          help='Row index (0-based) of model to train from CSV file')

    # file locations
    file_group = parser.add_argument_group('File Locations')
    file_group.add_argument('--datadir', type=str, default=_NOT_PROVIDED,
                           help='Path to directory containing embedding kernel data')
    file_group.add_argument('--outdir', type=str, default=_NOT_PROVIDED,
                           help='Output directory path')
    
    # device settings
    device_group = parser.add_argument_group('Device Settings')
    device_group.add_argument('--device', type=str, default=_NOT_PROVIDED,
                             help='Device to use: cpu, cuda, mps, or None for auto-detection')
    device_group.add_argument('--n_threads', type=int, default=_NOT_PROVIDED,
                             help='Number of threads for parallel processing')

    # data parameters
    task_group = parser.add_argument_group('Task Parameters')
    task_group.add_argument('--task', type=str, default=_NOT_PROVIDED,
                           help='Task name (default: PerceptualDecisionMaking-v0)')
    task_group.add_argument('--time_step', type=int, default=_NOT_PROVIDED,
                           help='Time step in milliseconds (default: 100)')
    task_group.add_argument('--seq_len_multi', type=int, default=_NOT_PROVIDED,
                           help='Sequence length multiplier (default: 5)')

    # RNN model and training parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--rnn_model', type=str, default=_NOT_PROVIDED,
                            help='RNN model type (default: rnn-tanh)')
    model_group.add_argument('--hidden_size', type=int, default=_NOT_PROVIDED,
                            help='Hidden layer size (default: 100)')
    model_group.add_argument('--batch_size', type=int, default=_NOT_PROVIDED,
                            help='Batch size (default: 32)')
    model_group.add_argument('--learning_rate', type=float, default=_NOT_PROVIDED,
                            help='Learning rate (default: 0.001)')
    model_group.add_argument('--n_runs', type=int, default=_NOT_PROVIDED,
                            help='Number of training runs (default: 10)')
    model_group.add_argument('--n_epochs', type=int, default=_NOT_PROVIDED,
                            help='Number of epochs (default: 5000)')
    model_group.add_argument('--print_freq', type=int, default=_NOT_PROVIDED,
                            help='Terminal print frequency in epochs (default: 100)')
    model_group.add_argument('--log_freq', type=int, default=_NOT_PROVIDED,
                            help='Performance logging frequency in epochs (default: 100)')
    model_group.add_argument('--write_freq', type=int, default=_NOT_PROVIDED,
                            help='H5 checkpoint write frequency in epochs (default: 1000)')
    model_group.add_argument('--mask_weights', type=str, default=_NOT_PROVIDED,
                            help='Whether to mask weights: True or False (default: False)')
    model_group.add_argument('--reservoir_mode', type=str, default=_NOT_PROVIDED,
                            help='Use reservoir computing with Ridge regression for output weights: True/False (default: False)')
    model_group.add_argument('--ridge_alpha', type=float, default=_NOT_PROVIDED,
                            help='Ridge regression regularization strength (default: 1.0)')
    model_group.add_argument('--spatial_only_epochs', type=int, default=_NOT_PROVIDED,
                            help='Number of initial epochs to train with spatial loss only before including task loss (default: 0)')
    model_group.add_argument('--spectral_radius', type=float, default=_NOT_PROVIDED,
                            help='Target spectral radius for reservoir weight matrix (default: 0.9)')
    model_group.add_argument('--rec_noise', type=float, default=_NOT_PROVIDED,
                             help='Scaling factor for added node-level (recurrent) noise (default: 0.0)')
    
    # weight initialization parameters
    init_group = parser.add_argument_group('Flexible Weight Initialization')
    init_group.add_argument('--init_ih_w', type=str, default=_NOT_PROVIDED,
                           help='Input-to-hidden weight initialization: None, float, or "min max"')
    init_group.add_argument('--train_ih_w', type=str, default=_NOT_PROVIDED,
                           help='Train input-to-hidden weights: True/False (default: True)')
    init_group.add_argument('--init_ih_b', type=str, default=_NOT_PROVIDED,
                           help='Input-to-hidden bias initialization: None, float, or "min max"')
    init_group.add_argument('--train_ih_b', type=str, default=_NOT_PROVIDED,
                           help='Train input-to-hidden biases: True/False (default: True)')
    init_group.add_argument('--init_hh_w', type=str, default=_NOT_PROVIDED,
                           help='Hidden-to-hidden weight initialization: None, float, or "min max"')
    init_group.add_argument('--train_hh_w', type=str, default=_NOT_PROVIDED,
                           help='Train hidden-to-hidden weights: True/False (default: True)')
    init_group.add_argument('--init_hh_b', type=str, default=_NOT_PROVIDED,
                           help='Hidden-to-hidden bias initialization: None, float, or "min max"')
    init_group.add_argument('--train_hh_b', type=str, default=_NOT_PROVIDED,
                           help='Train hidden-to-hidden biases: True/False (default: True)')
    init_group.add_argument('--init_ho_w', type=str, default=_NOT_PROVIDED,
                           help='Hidden-to-output weight initialization: None, float, or "min max"')
    init_group.add_argument('--train_ho_w', type=str, default=_NOT_PROVIDED,
                           help='Train hidden-to-output weights: True/False (default: True)')
    init_group.add_argument('--init_ho_b', type=str, default=_NOT_PROVIDED,
                           help='Hidden-to-output bias initialization: None, float, or "min max"')
    init_group.add_argument('--train_ho_b', type=str, default=_NOT_PROVIDED,
                           help='Train hidden-to-output biases: True/False (default: True)')
    init_group.add_argument('--allow_self_connections', type=str, default=_NOT_PROVIDED,
                           help='Allow self-connections: True/False (default: True)')
 
    # regularization parameters
    reg_group = parser.add_argument_group('Regularization Parameters')
    reg_group.add_argument('--reg_type', type=str, default=_NOT_PROVIDED,
                          help='Regularization type: l1, l2, l2s, pearson, or pearson_l2s (default: l2)')
    reg_group.add_argument('--reg_weight', type=float, default=_NOT_PROVIDED,
                          help='Regularization weight (default: 0.001)')
    reg_group.add_argument('--kernel_type', type=str, default=_NOT_PROVIDED,
                          help='Kernel type for spatial regularization (default: None)')
    reg_group.add_argument('--kernel_normalization', type=str, default=_NOT_PROVIDED,
                          help='Kernel normalization method (default: mean)')
    
    # continuous time parameter
    time_group = parser.add_argument_group('Continuous Time Parameters')
    time_group.add_argument('--alpha', type=str, default=_NOT_PROVIDED,
                           help='Continuous time parameter: float scalar or path to a .txt file '
                                'containing one value per line (default: 1.0)')

    # parse inputs
    args = parser.parse_args()
    
    # Validation: CSV and model_index must be provided together
    csv_provided = args.params_csv is not _NOT_PROVIDED
    index_provided = args.model_index is not _NOT_PROVIDED
    
    if csv_provided != index_provided:
        parser.error("--params_csv and --model_index must be provided together (or both omitted)")
    
    return args

                
def create_config_from_args(args):
    """
    Create configuration dictionary from command line arguments.
    Uses sentinel value to distinguish "not provided" vs "explicitly None".
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns
    -------
    dict
        Configuration dictionary with only explicitly provided arguments
    """
    config = {}
    
    # Simple parameters that can be directly copied
    simple_params = [
        'datadir', 'outdir', 'device', 'n_threads', 'task', 'time_step', 'seq_len_multi',
        'rnn_model', 'hidden_size', 'batch_size', 'learning_rate', 'n_runs',
        'n_epochs', 'print_freq', 'log_freq', 'write_freq', 'reg_type', 'reg_weight', 'kernel_type',
        'kernel_normalization', 'rec_noise', 'ridge_alpha', 'spectral_radius', 'spatial_only_epochs'
    ]
    
    for param in simple_params:
        value = getattr(args, param, _NOT_PROVIDED)
        if value is not _NOT_PROVIDED:  # user actually provided this argument
            # handle string "None" to actual None conversion
            if isinstance(value, str) and value.lower() == 'none':
                config[param] = None
            else:
                config[param] = value
    
    # Handle special parsing for specific parameters
    if getattr(args, 'mask_weights', _NOT_PROVIDED) is not _NOT_PROVIDED:
        config['mask_weights'] = parse_boolean_value(args.mask_weights)
    
    # Handle flexible weight initialization parameters
    weight_init_params = [
        'init_ih_w', 'init_ih_b', 'init_hh_w', 'init_hh_b', 'init_ho_w', 'init_ho_b'
    ]
    for param in weight_init_params:
        value = getattr(args, param, _NOT_PROVIDED)
        if value is not _NOT_PROVIDED:
            config[param] = parse_weight_init_value(value)
    
    # Handle boolean training parameters
    train_params = [
        'train_ih_w', 'train_ih_b', 'train_hh_w', 'train_hh_b',
        'train_ho_w', 'train_ho_b', 'allow_self_connections', 'reservoir_mode'
    ]
    for param in train_params:
        value = getattr(args, param, _NOT_PROVIDED)
        if value is not _NOT_PROVIDED:
            config[param] = parse_boolean_value(value)

    # Handle alpha (scalar float or path to a .txt vector file)
    alpha_raw = getattr(args, 'alpha', _NOT_PROVIDED)
    if alpha_raw is not _NOT_PROVIDED:
        config['alpha'] = parse_alpha_value(alpha_raw)

    return config


def validate_freq_config(config):
    """
    Validate that print_freq, log_freq, and write_freq are consistent with n_epochs.

    Rules enforced:
      - n_epochs % print_freq == 0
      - n_epochs % log_freq  == 0
      - n_epochs % write_freq == 0
      - write_freq % log_freq == 0  (can only write what has been logged)
    """
    n_epochs   = config['n_epochs']
    print_freq = config['print_freq']
    log_freq   = config['log_freq']
    write_freq = config['write_freq']

    errors = []
    if n_epochs % print_freq != 0:
        errors.append(f"n_epochs ({n_epochs}) must be divisible by print_freq ({print_freq})")
    if n_epochs % log_freq != 0:
        errors.append(f"n_epochs ({n_epochs}) must be divisible by log_freq ({log_freq})")
    if n_epochs % write_freq != 0:
        errors.append(f"n_epochs ({n_epochs}) must be divisible by write_freq ({write_freq})")
    if write_freq % log_freq != 0:
        errors.append(f"write_freq ({write_freq}) must be divisible by log_freq ({log_freq})")

    if errors:
        raise ValueError("Invalid frequency configuration:\n" + "\n".join(f"  - {e}" for e in errors))


def build_config():
    """
    Build the full configuration dictionary from command line arguments and/or a CSV file.

    Parses arguments, optionally loads and merges a CSV configuration, then applies defaults.

    Returns
    -------
    dict
        Fully resolved configuration dictionary
    """
    args = get_args()
    args_config = create_config_from_args(args)

    if args.params_csv is not _NOT_PROVIDED:
        csv_config = load_config_from_csv(args.params_csv, args.model_index)
        config = merge_configs(csv_config, args_config)
    else:
        config = args_config

    config = apply_defaults(config)

    if not config.get('reservoir_mode', False):
        validate_freq_config(config)

    # Validate required fields
    for required in ['datadir', 'outdir']:
        if not config.get(required):
            raise ValueError(f"'{required}' is required but was not provided.")

    return config


def apply_defaults(config):
    """
    Apply default values to configuration for any missing required parameters.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Configuration with defaults applied
    """
    defaults = {
        'datadir': None,
        'outdir': None,
        'device': 'None',
        'n_threads': None,
        'task': 'PerceptualDecisionMaking-v0',
        'time_step': 100,
        'seq_len_multi': 5,
        'rnn_model': 'rnn-tanh',
        'hidden_size': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'n_runs': 10,
        'n_epochs': 5000,
        'print_freq': 100,
        'log_freq': 100,
        'write_freq': 1000,
        'mask_weights': False,
        'reservoir_mode': False,
        'ridge_alpha': 1.0,
        'spatial_only_epochs': 0,
        'spectral_radius': 0.9,
        'reg_type': 'l2',
        'reg_weight': 0.001,
        'kernel_type': None,
        'kernel_normalization': 'mean',
        'rec_noise': 0.0,
        'alpha': 1.0,
        'init_ih_w': None,
        'train_ih_w': True,
        'init_ih_b': None,
        'train_ih_b': True,
        'init_hh_w': None,
        'train_hh_w': True,
        'init_hh_b': None,
        'train_hh_b': True,
        'init_ho_w': None,
        'train_ho_w': True,
        'init_ho_b': None,
        'train_ho_b': True,
        'allow_self_connections': True
    }
    
    # Apply defaults only for missing keys
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value

    # Expand ~ in path arguments
    for path_key in ['datadir', 'outdir']:
        if config.get(path_key):
            config[path_key] = os.path.expanduser(config[path_key])

    return config


def print_config(config):
    """
    Print all configuration parameters.
    Arrays, tensors, and DataFrames are summarised by shape/dtype rather than
    printed in full; dicts show only their keys.
    """
    def _fmt(value):
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return f'Tensor shape={tuple(value.shape)} dtype={value.dtype}'
            if isinstance(value, torch.device):
                return str(value)
        except ImportError:
            pass

        if isinstance(value, np.ndarray):
            return f'ndarray shape={value.shape} dtype={value.dtype}'
        if isinstance(value, pd.DataFrame):
            return f'DataFrame shape={value.shape}'
        if isinstance(value, pd.Series):
            return f'Series len={len(value)}'
        if isinstance(value, dict):
            return str(value)
        if isinstance(value, (list, tuple)) and len(value) > 8:
            return f'{type(value).__name__} len={len(value)}'
        return str(value)

    width = 64
    print()
    print('- ' * (width // 2))
    print('Configuration')
    print('- ' * (width // 2))
    pad = max((len(k) for k in config), default=10)
    for key, value in config.items():
        print(f'{key:<{pad}}  {_fmt(value)}')
    print('- ' * (width // 2))
    print()