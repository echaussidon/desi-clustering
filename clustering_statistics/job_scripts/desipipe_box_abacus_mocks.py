"""
Script to create and spawn desipipe tasks to compute clustering measurements on abacus box mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_abacus_mocks.py  # create the list of tasks
desipipe tasks -q abacus_mocks  # check the list of tasks
desipipe spawn -q abacus_mocks --spawn  # spawn the jobs
desipipe queues -q abacus_mocks  # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('abacus_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/abacus_mocks/slurm-%j.out', 'slurm_outputs/abacus_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))


def run_stats(tracer='LRG', version='abacus-2ndgen', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum']):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import box_tools, setup_logging, compute_box_stats_from_options, fill_box_fiducial_options
    setup_logging()

    cache = {}
    zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)
    get_box_stats_fn = functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir)
    for imock in imocks:
        for zsnap in zsnaps:
            for los in 'xyz':
                options = dict(catalog=dict(version=version, tracer=tracer, zsnap=zsnap, los=los, imock=imock))
                options = fill_box_fiducial_options(options)
                compute_box_stats_from_options(stats, get_box_stats_fn=get_box_stats_fn, cache=cache, **options)


if __name__ == '__main__':

    mode = 'interactive'
    #mode = 'slurm'
    stats, postprocess = [], []
    stats = ['mesh2_spectrum'] # 'mesh3_spectrum']
    #stats = ['window_mesh2_spectrum']
    #stats = ['window_mesh3_spectrum']
    #postprocess = ['combine_regions']
    imocks = np.arange(25)

    stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/box/')
    version = 'abacus-2ndgen'

    for tracer in ['BGS_BRIGHT-21.35', 'LRG', 'ELG', 'QSO'][1:2]:

        _run_stats = run_stats if mode == 'interactive' else tm.python_app(run_stats)

        if any('window' in stat or 'covariance' in stat for stat in stats):
            _imocks = [0]
            _run_stats(tracer, version=version, imocks=[0], stats_dir=stats_dir, stats=stats)
        elif stats:
            batch_imocks = np.array_split(imocks, max(len(imocks) // 10, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                _run_stats(tracer, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats)
