import os, random, warnings, sys
import numpy as np
from scipy.spatial import distance
import pandas as pd
from functools import partial
import pickle

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from src.neural_network import train_helper, ModelStateManager, ModelDataManager
from src import utils
from src.io_utils import build_config, print_config

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


def train_single(config):
    """Train one model config (a normal model, or one spin-permuted null geometry)
    to its own output files. Body factored out of train() so the null path can
    call it once per permutation."""
    outdir = config['outdir']
    device = config['device']
    n_threads = config['n_threads']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']

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

    # expose file paths in config so train_helper can build checkpoint callbacks
    config['models_path']  = models_path
    config['outputs_path'] = outputs_path

    # run indices to train (explicit null_run_ids if set, else the contiguous
    # [null_run, null_run + n_runs); see _run_ids)
    runs = _run_ids(config)

    # check which runs are complete, partial, or not yet started
    if os.path.isfile(models_path) and os.path.isfile(outputs_path):
        print('found existing output files! checking for missing/incomplete runs...')
        complete_runs = set()
        for run in runs:
            model_epochs  = model_manager.get_run_epochs(run)
            output_epochs = output_manager.get_run_epochs(run)
            if model_epochs and output_epochs:
                last_epoch = min(max(model_epochs), max(output_epochs))
                if last_epoch == n_epochs - 1:
                    complete_runs.add(run)
        n_complete = len(complete_runs)
        n_partial  = sum(
            1 for r in runs
            if r not in complete_runs
            and model_manager.get_run_epochs(r)
            and output_manager.get_run_epochs(r)
        )
        print(f'found {n_complete} completed runs, {n_partial} partial runs')
        if n_complete == len(runs):
            all_done = True
            print('training already completed! skipping...')
        else:
            all_done = False
            rem_runs = np.array([r for r in runs if r not in complete_runs])
            print(f'will train {len(rem_runs)} more runs ({n_partial} resuming, {len(rem_runs) - n_partial} new)')
    else:
        all_done = False
        rem_runs = np.array(runs)
    
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
                        outputs_list = pool.map(partial_train_with_gpu, run_gpu_pairs)

                    # Save outputs
                    print(f'Saving outputs for runs {chunk+1}')
                    for idx, run in enumerate(chunk):
                        output_manager.save_model_data(outputs_list[idx], run=run, epoch=n_epochs - 1)
            
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
                    outputs = train_helper_with_gpu(run, config, gpu_id)

                    # Save outputs
                    print(f'Saving outputs for run {run}')
                    output_manager.save_model_data(outputs, run=run, epoch=n_epochs - 1)
        
        elif device.type == 'mps' or (device.type == 'cpu' and n_threads == 1):
            print(f'Running in serial on {device.type}...')
            # Original serial processing
            for run in rem_runs:
                outputs = partial_train_helper(run)
                # save outputs
                print(f'saving outputs for run {run}')
                output_manager.save_model_data(outputs, run=run, epoch=n_epochs - 1)
        
        else:
            print(f'Running in parallel on {device.type} using {n_threads} threads...')
            # Original CPU parallel processing
            proc_chunks = np.array_split(rem_runs, np.ceil(len(rem_runs)/n_threads))
            for chunk in proc_chunks:
                with torch.multiprocessing.get_context('spawn').Pool(processes=len(chunk), maxtasksperchild=1) as pool:
                    outputs_list = pool.map(partial_train_helper, chunk)
                # save outputs
                print(f'saving outputs for runs {chunk}')
                for idx, run in enumerate(chunk):
                    output_manager.save_model_data(outputs_list[idx], run=run, epoch=n_epochs - 1)
        
        # save config (exclude internal runtime keys)
        config_to_save = {k: v for k, v in config.items() if k not in ('models_path', 'outputs_path')}
        np.save(config_path, config_to_save)


def _run_ids(config):
    """Explicit list of run indices to train for one model file.

    If ``null_run_ids`` is set (a non-empty list), those exact run indices are
    trained -- e.g. 5 runs chosen to span the real VE distribution for the
    mixed-effects null, so each spin geometry is evaluated over a representative
    bracket of training draws rather than a contiguous block (the run index seeds
    init / curriculum, so run r reuses the real model's run-r training draw).
    Otherwise falls back to the contiguous range [null_run, null_run + n_runs),
    which leaves ordinary models and the fixed-init null unchanged.
    """
    ids = config.get('null_run_ids', None)
    if ids:
        ids = [int(r) for r in ids]
        # null_run_ids takes precedence over the null_run/n_runs range. Make that
        # explicit rather than silent: n_runs must match, and a non-zero null_run
        # (which would also mis-tag the output filenames) is rejected.
        n_runs = int(config.get('n_runs', len(ids)) or len(ids))
        if n_runs != len(ids):
            raise ValueError(
                f'null_run_ids has {len(ids)} entries but n_runs={n_runs}; set n_runs={len(ids)}')
        if int(config.get('null_run', 0) or 0) != 0:
            raise ValueError(
                f"null_run={config.get('null_run')} is set alongside null_run_ids; leave null_run=0 "
                f"when using explicit run ids (else the -nr filename tag would not match runs {ids})")
        return ids
    start = int(config.get('null_run', 0) or 0)
    return list(range(start, start + int(config['n_runs'])))


def _incomplete_runs(models_path, outputs_path, model_manager, output_manager, runs, n_epochs):
    """Runs still needing training for one model file (resume-aware).

    ``runs`` is the explicit list of run indices to check.
    """
    if os.path.isfile(models_path) and os.path.isfile(outputs_path):
        complete = set()
        for run in runs:
            me = model_manager.get_run_epochs(run)
            oe = output_manager.get_run_epochs(run)
            if me and oe and min(max(me), max(oe)) == n_epochs - 1:
                complete.add(run)
        return [r for r in runs if r not in complete]
    return list(runs)


def _null_train_job(job):
    """Module-level worker: train one (permutation, run) job and return
    (perm_index, run, outputs). Module-level so it is picklable under 'spawn'."""
    perm_index, run, cfg = job
    return perm_index, run, train_helper(run, cfg)


def train_null_parallel(config, perms):
    """Train the spin-null ensemble by flattening every (permutation, run) pair
    into a single parallel pool, so a many-core CPU stays saturated even when the
    runs-per-permutation count is smaller than the available threads.

    Each permutation still writes to its own '-null{p}' files (identical layout to
    the sequential path); only the scheduling differs. Saving stays in the parent
    process, so final-output writes remain serialized.
    """
    outdir = config['outdir']
    n_threads = config['n_threads']
    n_epochs = config['n_epochs']
    run_ids = _run_ids(config)          # explicit null_run_ids or contiguous range
    base_K, base_D = config['regularization_kernel'], config['distance_matrix']

    # per-permutation setup: permuted kernel, file paths, managers, runs left to do
    perm_ctx, flat_jobs = [], []
    for p, perm in enumerate(perms):
        cfg = dict(config, null_index=p,
                   regularization_kernel=base_K[np.ix_(perm, perm)],
                   distance_matrix=(None if base_D is None else base_D[np.ix_(perm, perm)]))
        file_str = utils.get_file_str(cfg)
        outputs_path = os.path.join(outdir, file_str + '_outputs.h5')
        models_path = os.path.join(outdir, file_str + '_models.h5')
        config_path = os.path.join(outdir, file_str + '_config.npy')
        cfg['models_path'] = models_path
        cfg['outputs_path'] = outputs_path
        om = ModelDataManager(outputs_path)
        mm = ModelStateManager(models_path)
        rem = _incomplete_runs(models_path, outputs_path, mm, om, run_ids, n_epochs)
        perm_ctx.append({'cfg': cfg, 'om': om, 'config_path': config_path})
        flat_jobs.extend((p, run, cfg) for run in rem)

    n_done = len(perms) * len(run_ids) - len(flat_jobs)
    print(f'\nnull ensemble: {len(perms)} permutations x {len(run_ids)} runs '
          f'({n_done} complete, {len(flat_jobs)} to train) over {n_threads} threads')

    # train in thread-width chunks; route each result to its permutation's file
    n_chunks = int(np.ceil(len(flat_jobs) / n_threads)) if flat_jobs else 0
    for ci in range(n_chunks):
        chunk = flat_jobs[ci * n_threads:(ci + 1) * n_threads]
        print(f'  chunk {ci + 1}/{n_chunks}: {len(chunk)} (perm, run) jobs...')
        with torch.multiprocessing.get_context('spawn').Pool(
                processes=len(chunk), maxtasksperchild=1) as pool:
            results = pool.map(_null_train_job, chunk)
        for perm_index, run, outputs in results:
            perm_ctx[perm_index]['om'].save_model_data(outputs, run=run, epoch=n_epochs - 1)

    # save each permutation's config (exclude runtime-only keys)
    for ctx in perm_ctx:
        cfg_to_save = {k: v for k, v in ctx['cfg'].items()
                       if k not in ('models_path', 'outputs_path')}
        np.save(ctx['config_path'], cfg_to_save)


def train(config):
    # get config params
    datadir = config['datadir']
    outdir = config['outdir']
    device = config['device']
    n_threads = config['n_threads']

    # RNN model and training parameters
    hidden_size = config['hidden_size']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']
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

    # fan out into spin-permuted (SA-matched) null geometries, or train the single model
    null_perms = int(config.get('null_perms', 0) or 0)
    if null_perms and regularization_kernel is not None:
        from src.spatial_null import spin_permutations
        sphere = np.loadtxt(os.path.join(datadir, f'schaefer{hidden_size * 2}_LH_sphere_coords.txt'))
        perms = spin_permutations(sphere, null_perms, seed=int(config.get('null_seed', 0) or 0))
        if device.type == 'cpu' and n_threads and n_threads > 1:
            # flatten (permutation x run) into one pool so all threads stay busy
            train_null_parallel(config, perms)
        else:
            # serial / single-GPU path: train each permutation in turn
            base_K, base_D = regularization_kernel, distance_matrix
            for p, perm in enumerate(perms):
                print(f'\n=== null permutation {p + 1}/{null_perms} ===')
                cfg = dict(config, null_index=p,
                           regularization_kernel=base_K[np.ix_(perm, perm)],
                           distance_matrix=(None if base_D is None else base_D[np.ix_(perm, perm)]))
                train_single(cfg)
    else:
        train_single(config)


if __name__ == '__main__':

    exit_code = 0
    error_message = ""

    config = build_config()

    # device configuration
    device = utils.get_device(config['device'])
    if device.type == 'cpu':
        n_threads = utils.get_n_threads(config['n_threads'], 1)
        n_gpu = 0
    else:
        n_threads = config['n_threads']
        device = utils.get_device(device_opt=config['device'], n_devices=n_threads)
        n_gpu = utils.get_n_gpu()
        try:
            for ii in range(n_gpu):
                print(f'gpu {ii} -- {torch.cuda.get_device_name(ii)}')
        except:
            print(f'gpu -- {device}')

    config['device'] = device
    config['n_threads'] = n_threads
    config['n_gpu'] = n_gpu

    # task details
    task_with_modifier = config['task']
    task_no_modifier, task_modifier = utils.get_task_modifier(task_with_modifier)
    utils.check_if_supported(task=task_no_modifier, modifier=task_modifier)
    seq_len, timing = utils.get_seq_len_and_timing(task=task_no_modifier, modifier=task_modifier, seq_len_multi=config['seq_len_multi'], dt=config['time_step'])
    env_kwargs = {'dt': config['time_step'], 'timing': timing}
    extra_kwargs = utils.get_extra_task_options(task_no_modifier)
    env_kwargs.update(extra_kwargs)

    config['task_no_modifier'] = task_no_modifier
    config['task_modifier'] = task_modifier
    config['task_with_modifier'] = task_with_modifier
    config['seq_len'] = seq_len
    config['env_kwargs'] = env_kwargs

    print_config(config)

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
