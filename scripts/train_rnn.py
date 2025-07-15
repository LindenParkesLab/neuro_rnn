import os, random, argparse, warnings, sys
import numpy as np
from scipy.spatial import distance
import pandas as pd
from functools import partial
import pickle

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from src.neural_network import train_helper, ModelStateManager, ModelDataManager
from src import utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%

def cleanup_and_exit(exit_code, message):
    """Clean up resources that prevent script exit, and force exit"""
    import gc
    import time
    
    print("Cleaning up for script exit...", flush=True)
    
    # 1. PyTorch cleanup
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except:
        pass
    
    # 2. HDF5 cleanup  
    try:
        import h5py
        for obj_id in h5py.h5f.get_obj_ids(h5py.h5f.OBJ_ALL):
            try:
                h5py.h5o.close(obj_id)
            except:
                pass
    except:
        pass
    
    # 3. Multiprocessing cleanup
    try:
        import multiprocessing as mp
        for p in mp.active_children():
            p.terminate()
            p.join(timeout=0.5)
    except:
        pass
    
    # 4. Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    # 5. Brief pause for cleanup
    time.sleep(0.1)
    
    print("Cleanup completed", flush=True)
    if exit_code == 0:
        print(f"Script completed successfully{': ' + message if message else ''}", flush=True)
    else:
        print(f"Script failed (exit code {exit_code}){': ' + message if message else ''}", file=sys.stderr, flush=True)

    os._exit(exit_code)


def train_helper_with_gpu_assignment(run_gpu_pair, config):
    """
    Wrapper function for multiprocessing with GPU assignment
    This needs to be at module level to be pickleable
    """
    run, gpu_id = run_gpu_pair
    from src.neural_network import train_helper_with_gpu
    return train_helper_with_gpu(run, config, gpu_id)


def train(config):
    # get config params
    datadir = config['datadir']
    outdir = config['outdir']
    device = config['device']
    n_threads = config['n_threads']

    # RNN model and training parameters
    hidden_size = config['hidden_size']
    n_runs = config['n_runs']
    mask_weights = config['mask_weights']

    # regularization parameters
    kernel_type = config['kernel_type']
    kernel_normalization = config['kernel_normalization']

    # setup output dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # weight masks
    if mask_weights:
        centroids = pd.read_csv(os.path.join(datadir, 'schaefer{0}_centroids.csv'.format(hidden_size * 2)))
        centroids = centroids[:hidden_size]  # pull out left hemisphere
        roi_names = list(centroids['ROI Name'])
        input_system = 'Vis'
        output_system = 'Default'
        masks = utils.get_weight_masks_schaefer(roi_names=roi_names, input_system=input_system, output_system=output_system)
        n_io = '{0}-{1}'.format(np.sum(masks['input_weight_mask']), np.sum(masks['output_weight_mask']))
        config['masks'] = masks
        config['centroids'] = centroids
    else:
        n_io = 'na'
    config['n_io'] = n_io

    # setup regularization kernel
    regularization_kernel, distance_matrix = utils.load_embedding(kernel_type=kernel_type,
                                                                  datadir=datadir,
                                                                  hidden_size=hidden_size,
                                                                  kernel_normalization=kernel_normalization)
    config['distance_matrix'] = distance_matrix
    config['regularization_kernel'] = regularization_kernel

    # get file name
    file_str = utils.get_file_str(config)
    print('\n')
    print(file_str)

    # set file paths
    outputs_path = os.path.join(outdir, file_str + '_outputs.h5')
    models_path = os.path.join(outdir, file_str + '_models.h5')
    config_path = os.path.join(outdir, file_str + '_config.npy')

    output_manager = ModelDataManager(outputs_path)
    model_manager = ModelStateManager(models_path)
    
    # check if outputs exist
    if os.path.isfile(models_path) and os.path.isfile(outputs_path):
        print('found existing output files! checking for missing runs... ')
        n_compl_models, _, _ = model_manager.get_info()
        n_compl_outputs, _ = output_manager.get_info()
        n_compl_runs = np.min((n_compl_models,n_compl_outputs))
        print(f'found outputs for {n_compl_runs} runs')
        if n_compl_runs == n_runs:
            all_done = True
            print('training already completed! skipping...')
        else:
            all_done = False
            rem_runs = np.arange(n_compl_runs,n_runs)
            print(f'will train {len(rem_runs)} more runs')
    else:
        all_done = False
        rem_runs = np.arange(n_runs)
    
    if not all_done:
        # Print GPU environment info for debugging
        if device.type == 'cuda':
            utils.print_gpu_environment_info()
        
        # prepare partial function for multiprocessing
        partial_train_helper = partial(train_helper, config=config)
        
        # Enhanced GPU processing with multi-GPU support
        if device.type == 'cuda':
            available_gpus = utils.get_safe_gpu_list()
            
            if len(available_gpus) > 1 and n_threads and n_threads > 1:
                print(f'Running in parallel on {len(available_gpus)} GPUs using {min(n_threads, len(available_gpus))} processes...')
                
                # Get optimal GPU assignment for all runs
                gpu_assignments = utils.get_optimal_gpu_assignment(len(rem_runs), n_threads)
                
                # Prepare processing chunks
                max_concurrent = min(n_threads, len(available_gpus))
                proc_chunks = np.array_split(rem_runs, np.ceil(len(rem_runs) / max_concurrent))
                
                # Create partial function for GPU assignment (FIXED - using module level function)
                partial_train_with_gpu = partial(train_helper_with_gpu_assignment, config=config)
                
                # Process chunks in parallel
                for chunk in proc_chunks:
                    # Get GPU assignments for this chunk
                    chunk_gpu_assignments = [gpu_assignments[np.where(rem_runs == run)[0][0]] for run in chunk]
                    
                    # Create run-GPU pairs for this chunk
                    run_gpu_pairs = list(zip(chunk, chunk_gpu_assignments))
                    
                    # Run this chunk in parallel
                    with torch.multiprocessing.get_context('spawn').Pool(processes=len(chunk), maxtasksperchild=1) as pool:
                        results = pool.map(partial_train_with_gpu, run_gpu_pairs)
                        outputs, models = zip(*results)
                    
                    # Save outputs and models
                    print(f'Saving outputs and models for runs {chunk+1}')
                    for idx, run in enumerate(chunk):
                        output_manager.save_model_data(outputs[idx], run)
                        model_manager.save_model_states(models[idx], run)
            
            else:
                # Single GPU or sequential processing
                if len(available_gpus) > 1:
                    print(f'Running sequentially across {len(available_gpus)} GPUs...')
                    gpu_assignments = utils.get_optimal_gpu_assignment(len(rem_runs))
                else:
                    print(f'Running in serial on single GPU...')
                    gpu_assignments = [None] * len(rem_runs)
                
                # Sequential processing with GPU rotation
                from src.neural_network import train_helper_with_gpu
                for i, run in enumerate(rem_runs):
                    gpu_id = gpu_assignments[i] if gpu_assignments[i] is not None else None
                    outputs, models = train_helper_with_gpu(run, config, gpu_id)
                    
                    # Save outputs and models
                    print(f'Saving outputs and models for run {run+1}')
                    output_manager.save_model_data(outputs, run)
                    model_manager.save_model_states(models, run)
        
        elif device.type == 'mps' or (device.type == 'cpu' and n_threads == 1):
            print(f'Running in serial on {device.type}...')
            # Original serial processing
            for run in rem_runs:
                outputs, models = partial_train_helper(run)
                # save outputs and models
                print(f'saving outputs and models for run {run+1}')
                output_manager.save_model_data(outputs, run)
                model_manager.save_model_states(models, run)
        
        else:
            print(f'Running in parallel on {device.type} using {n_threads} threads...')
            # Original CPU parallel processing
            proc_chunks = np.array_split(rem_runs, np.ceil(len(rem_runs)/n_threads))
            for chunk in proc_chunks:
                with torch.multiprocessing.get_context('spawn').Pool(processes=len(chunk), maxtasksperchild=1) as pool:
                    outputs, models = zip(*pool.map(partial_train_helper, chunk))
                # save outputs and models
                print(f'saving outputs and models for runs {chunk+1}')
                for idx, run in enumerate(chunk):
                    output_manager.save_model_data(outputs[idx], run)
                    model_manager.save_model_states(models[idx], run)
        
        # save config
        np.save(config_path, config)


def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    # file locations
    parser.add_argument('--datadir', type=str, default='/home/lindenmp/research_projects/neuro_rnn/data')
    parser.add_argument('--outdir', type=str, default='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model')
    
    # device settings
    parser.add_argument('--device', type=str, default='None')
    parser.add_argument('--n_threads', type=int, default=None)

    # data parameters
    parser.add_argument('--task', type=str, default='PerceptualDecisionMaking-v0')
    parser.add_argument('--dt', type=int, default=100)
    parser.add_argument('--seq_len_multi', type=int, default=5)

    # RNN model and training parameters
    parser.add_argument('--rnn_model', type=str, default='rnn-tanh')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--epoch_log', type=int, default=100)
    parser.add_argument('--mask_weights', type=str, default='True')
    # parser.add_argument('--init_rnn_weights', type=lambda s: None if s.lower() == 'none' else tuple(map(float, s.split(','))), nargs=1, default=None)
    parser.add_argument('--init_rnn_weights', type=str, nargs='+', default=None)
 
    # regularization parameters
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--kernel_type', type=str, default='None')
    parser.add_argument('--kernel_normalization', type=str, default='mean')
    
    # continuous time parameter
    parser.add_argument('--alpha', type=float, default=0.0)

    # parse inputs
    args = parser.parse_args()
    args.datadir = os.path.expanduser(args.datadir)
    args.outdir = os.path.expanduser(args.outdir)

    return args


if __name__ == '__main__':

    exit_code = 0
    error_message = ""
    
    args = get_args()
    
    # device configuration
    device = utils.get_device(args.device)
    if device.type == 'cpu':
        n_threads = utils.get_n_threads(args.n_threads, 1)
        n_gpu = 0
    else:
        # n_threads = None
        n_threads = args.n_threads
        device = utils.get_device(device_opt=args.device, n_devices=n_threads)
        n_gpu = utils.get_n_gpu()
        try:
            for ii in range(n_gpu):
                print(f'gpu {ii} -- {torch.cuda.get_device_name(ii)}')
        except:
            print(f'gpu -- {device}')

    # kernel and mask
    if args.kernel_type == 'None':
        kernel_type = None
    else:
        kernel_type = args.kernel_type

    if args.mask_weights == 'False':
        mask_weights = False
    elif args.mask_weights == 'True':
        mask_weights = True
    
    # rnn weights
    if args.init_rnn_weights is not None:
        args.init_rnn_weights = utils.parse_float_tuple(args.init_rnn_weights)
    
    # task details 
    task_with_modifier = args.task
    task_no_modifier, task_modifier = utils.get_task_modifier(task_with_modifier)
    utils.check_if_supported(task=task_no_modifier, modifier=task_modifier)
    seq_len, timing = utils.get_seq_len_and_timing(task=task_no_modifier, modifier=task_modifier, seq_len_multi=args.seq_len_multi, dt=args.dt)
    env_kwargs = {'dt': args.dt, 'timing': timing}
    extra_kwargs = utils.get_extra_task_options(task_no_modifier)
    env_kwargs.update(extra_kwargs)
    print(' ')
    print('Task:             ' + task_no_modifier)
    print('Task modifier:    ' + task_modifier)
    print('Sequence length:  ' + str(seq_len))
    print('Task options:     ' + str(env_kwargs))
    print('Timing:           ' + str(timing))
    print('Alpha:            ' + str(args.alpha))
    
    # package all info into config
    config = {
        
        # file locations
        'datadir': args.datadir,
        'outdir': args.outdir,

        # task parameters
        'task_no_modifier': task_no_modifier,
        'task_modifier': task_modifier,
        'task_with_modifier': task_with_modifier,
        'dt': args.dt,
        'seq_len_multi': args.seq_len_multi,
        'seq_len': seq_len,

        # RNN model and training parameters
        'rnn_model': args.rnn_model,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_runs': args.n_runs,
        'n_epochs': args.n_epochs,
        'epoch_log': args.epoch_log,
        'mask_weights': mask_weights,
        'init_rnn_weights': args.init_rnn_weights,

        # regularization parameters
        'reg_type': args.reg_type,
        'reg_weight': args.reg_weight,
        'kernel_type': kernel_type,
        'kernel_normalization': args.kernel_normalization,
        
        # continuous time parameter
        'alpha': args.alpha,

        # ngym env
        'env_kwargs': env_kwargs,
        
        # device settings
        'device': device,
        'n_threads': n_threads,
        'n_gpu': n_gpu
        
    }

    try:
        train(config=config)
        exit_code = 0
        error_message = ''
        print(error_message, flush=True)

    except KeyboardInterrupt:
        exit_code = 2
        error_message = "Script interrupted by user (Ctrl+C)"
        print("\nScript interrupted by user")
        
    except FileNotFoundError as e:
        exit_code = 5
        error_message = f"File not found: {str(e)}"
        print(f"File error: {e}", file=sys.stderr, flush=True)
        
    except PermissionError as e:
        exit_code = 5
        error_message = f"Permission error: {str(e)}"
        print(f"Permission error: {e}", file=sys.stderr, flush=True)
        
    except Exception as e:
        exit_code = 1
        error_message = f"Unexpected error: {str(e)}"
        print(f"Error occurred: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()

    finally:
        cleanup_and_exit(exit_code, error_message)
        # os._exit(0)
