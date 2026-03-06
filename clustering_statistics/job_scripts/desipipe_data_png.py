"""
Script to create and spawn desipipe tasks to compute clustering measurements on data for PNG key project.

To run interactively see __main__ section below.

To create and spawn the tasks on NERSC, use the following commands:
```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# create the list of tasks:
python desipipe_data_png.py  --blinded

# check the list of tasks (you can also check with desipipe tasks -q '*/*' to see all the queues and tasks):
desipipe tasks -q data_png  

# spawn i.e. launch the jobs:
desipipe spawn -q data_png --spawn  

# check the queue:
desipipe queues -q data_png  

# Restart some jobs that failed (for instance due to time limit):**
desipipe retry -q data_png --state RUNNING # FAILED # KILLED

# delete the queue and all the tasks:
desipipe delete -q data_png --force
```
"""

import logging

import os
from pathlib import Path
import functools

import numpy as np


logger = logging.getLogger('Data PNG')


# disable jax warning:
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.distributed").setLevel(logging.ERROR)
# Remove warning from jax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_queue():
    """Set up the desipipe queue and task manager."""
    from desipipe import Queue, Environment, TaskManager, spawn

    queue = Queue('data_png')
    queue.clear(kill=False)

    output, error = 'slurm_outputs/data_png/slurm-%j.out', 'slurm_outputs/data_png/slurm-%j.err'
    kwargs = {}
    environ = Environment('nersc-cosmodesi')
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
    tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
    return tm, tm80


def run_stats(cat_dir=None, stats_dir=None, tracer='LRG', zranges=[0.4, 1.1], weights=['default-fkp'], regions=['NGC','SGC'], stats=['mesh2_spectrum'], **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: pass  # print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options

    # May need to add this for the desipipe now interactive is fine...
    #from mpi4py import MPI
    #setup_logging(level=(logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.ERROR))

    cache = {}
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    for region in regions:
        for weight in weights:
            # options will be filled by default options in compute_stats_from_options following analysis='local_png'
            options = dict(catalog=dict(cat_dir=cat_dir, tracer=tracer, zrange=zranges, weight=weight, region=region, ext='fits'), 
                           mesh2_spectrum={'cut': False}, window_mesh2_spectrum={'cut': False})
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, analysis='local_png', **options)


def postprocess_stats(cat_dir=None, stats_dir=None, tracer='LRG', zranges=[0.4, 1.1], weights=['default-fkp'], stats=['mesh2_spectrum'], postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    for weight in weights:
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
        options = dict(catalog=dict(cat_dir=cat_dir, tracer=tracer, zrange=zranges, weight=weight, ext='fits'), 
                       combine_regions={'stats': stats}, mesh2_spectrum={'cut': False}, window_mesh2_spectrum={'cut': False})
        postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, analysis='local_png', **options)


def collect_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Whether to run in interactive mode (without spawning jobs with desipipe).')
    parser.add_argument('--blinded', action='store_true', help='Run with blinded data or not.')
    args = parser.parse_args()
    logger.info(args)
    return args


if __name__ == '__main__':
    """
    # To run interactively on NERSC, use the following commands:

    salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g

    salloc -N 1 -C "gpu" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
    
    # source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main  # don't forget this to load environment variables
    # module unload desi-clustering
    
    source /global/homes/e/edmondc/.bash_profile
    
    srun -n 4 python desipipe_data_png.py --interactive --blinded
    """
    from clustering_statistics import setup_logging
    from mpi4py import MPI
    setup_logging(level=(logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.ERROR))

    args = collect_argparser()

    mode = 'interactive' if args.interactive else 'slurm'
    if mode == 'interactive':
        logger.info('Running in interactive mode')
    else:         
        logger.info("Create queue with jobs inside using desipipe. Don\'t forget to run `desipipe spawn -q data_png --spawn` to launch the jobs!")
        tm, tm80 = setup_queue()

    def get_run_stats():
        if mode == 'interactive':
            return run_stats
        else: 
            _tm = tm80
            #if tracer in ['LRG']: _tm = tm
            return _tm.python_app(run_stats)

    todo = ['mesh2_spectrum'] #, 'window_mesh2_spectrum', 'covariance_mesh2_spectrum']
    postprocess = ['combine_regions']
    logger.info(f'Running stats {todo} and postprocess {postprocess}')
    
    cat_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL/')
    stats_dir = Path(os.getenv('SCRATCH', '.')) / 'DR2_local_png' / 'measurements' / 'loa-v1/v2/fNL'

    if args.blinded:
        cat_dir = cat_dir / 'blinded'
        stats_dir = stats_dir / 'blinded'
    else:
        logger.error('NOT READY FOR UNBLINDED DATA YET')
        import sys
        sys.exit(3)
    
    logger.info(f'cat_dir: {cat_dir}')
    logger.info(f'stats_dir: {stats_dir}')

    regions = ['NGC', 'SGC', 'N', 'NGCnoN', 'SGCnoDES', 'DES'][:2]  # + ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]]
    logger.info(f'Running on regions: {regions}')

    for tracer in ['LRG', 'LRG_zcmb', 'ELGnotqso', 'QSO', 'QSO_zcmb', ('LRG', 'QSO'), ('LRG', 'ELGnotqso'), ('ELGnotqso', 'QSO')][:1]:
        from clustering_statistics import tools
        logger.info(tracer)
        zranges = tools.propose_fiducial(kind='zranges', tracer=tracer, analysis='local_png')
        #zranges += tools.propose_fiducial(kind='zranges', tracer=tracer)
        logger.info(f'zranges: {zranges}')

        weights = ['default-fkp-oqe', 'default-fkp']
        # get_run_stats()(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, regions=regions, stats=todo)
        
        # if postprocess:
        #    postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, postprocess=postprocess, stats=todo)

        # Choice of imaging systematics avaialble in the catalogs: https://desi.lbl.gov/trac/wiki/keyprojects/Y3-DR/LSScat/imaging_systematics
        if tracer in ['LRG', 'LRG_zcmb']:
            weights = ['default-fkp-oqe-wsys-imlin_finezbin_allebvcmb']
            if 'zcmb' in tracer:
                weights += ['default-fkp-oqe-wsys-imlin_finezbin_allebv']
        elif tracer in ['ELGnotqso']:
            weights = ['default-fkp-oqe-wsys-imlin_finezbin_nodebv']
        elif tracer == ('LRG', 'QSO'):
            weights = [('default-fkp-oqe-wsys-imlin_finezbin_allebvcmb', 'default-fkp-oqe')]
        else:
            weights = []
        
        get_run_stats()(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, regions=regions, stats=['mesh2_spectrum'])

        if postprocess:
           postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, postprocess=postprocess, stats=['mesh2_spectrum'])