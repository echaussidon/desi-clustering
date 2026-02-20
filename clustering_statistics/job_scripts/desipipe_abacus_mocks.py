"""
Script to create and spawn desipipe tasks to compute clustering measurements on abacus mocks.
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
environ = Environment(command='source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new') #, command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))


def run_stats(tracer='LRG', version='abacus-2ndgen-complete', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], ibatch=None, **kwargs):
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
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, combine_stats_from_options, fill_fiducial_options
    setup_logging()

    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    for imock in imocks:
        regions = ['NGC', 'SGC'][:1]
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock), mesh2_spectrum={}, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
            options = fill_fiducial_options(options)
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            combine_stats_from_options(stats, region_comb, regions, get_stats_fn=get_stats_fn, **options)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    mode = 'interactive'
    mode = 'slurm'
    #stats = ['mesh2_spectrum'] # 'mesh3_spectrum']
    #stats = ['window_mesh2_spectrum']
    stats = ['window_mesh3_spectrum']
    imocks = np.arange(25)

    stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    version = 'abacus-2ndgen-complete'
    
    for tracer in ['BGS_BRIGHT-21.35', 'LRG', 'ELG_LOP', 'QSO'][1:2]:
        if False:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=list(range(1001)))[:2]
            imocks = exists[1]['imock']
            rerun = []
            for zrange in tools.propose_fiducial('zranges', tracer):
                for kind in ['mesh2_spectrum', 'mesh3_spectrum']:
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, kind=kind, stats_dir=stats_dir, tracer=tracer, region='GCcomb', weight='default-FKP', zrange=zrange, version=version), test_if_readable=True, imock=list(range(1001)))
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        if any('window' in stat for stat in stats):
            _imocks = [0]
            nbatches = 1
            tasks = []
            for ibatch in range(nbatches):
                task = get_run_stats()(tracer, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats, ibatch=(ibatch, nbatches))
                tasks.append(task)
            if nbatches > 1:
                # Add dependence on other tasks
                get_run_stats()(tracer, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats, ibatch=nbatches, tasks=tasks)
        else:
            batch_imocks = np.array_split(imocks, max(len(imocks) // 10, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                get_run_stats()(tracer, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats)