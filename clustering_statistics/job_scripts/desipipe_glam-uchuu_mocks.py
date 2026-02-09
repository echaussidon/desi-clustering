"""
Script to create and spawn desipipe tasks to compute clustering measurements on glam-uchuu mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_glam-uchuu_mocks.py         # create the list of tasks
desipipe tasks  -q glam-uchuu_mocks         # check the list of tasks
desipipe spawn  -q glam-uchuu_mocks --spawn # spawn the jobs
desipipe queues -q glam-uchuu_mocks         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('glam-uchuu_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/glam-uchuu_mocks/slurm-%j.out', 'slurm_outputs/glam-uchuu_mocks/slurm-%j.err'
kwargs = {}
tmp_dir = Path(os.getenv('SCRATCH'), 'tmp')
tmp_dir.mkdir(exist_ok=True)
environ = Environment('nersc-cosmodesi', {'TMPDIR': tmp_dir, 'XLA_FLAGS': f'"--xla_gpu_cuda_data_dir={tmp_dir} --xla_dump_to={tmp_dir}"'})  # to avoid jax.errors.JaxRuntimeError: NOT_FOUND: /tmp/tempfile-nid008481-f65eab9791d225ca-724696-648e288b2979c
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=25), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))


def run_stats(tracer='LRG', version='glam-uchuu-v1-altmtl', imocks=[100], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], analysis='png_local',weight='default-oqe'):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, combine_stats_from_options, fill_fiducial_options
        
    setup_logging()
    cache = {}
    regions = ['NGC', 'SGC']
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    for imock in imocks:
        for region in regions:
            if analysis == 'full_shape':
                mesh2_spectrum = {'cut': True, 'auw': True}
            elif analysis == 'png_local':
                mesh2_spectrum = dict(optimal_weights=functools.partial(tools.compute_fiducial_png_weights, tracer=tracer) if 'oqe' in weight else None)
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock, weight=weight), 
                           mesh2_spectrum=mesh2_spectrum)
            options = fill_fiducial_options(options,analysis=analysis)
            for tracer in options['catalog']:
                options['catalog'][tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog'][tracer]['nran'])}
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            combine_stats_from_options(stats, region_comb, regions, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **options)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    mode = 'slurm'
    # mode = 'interactive'
    
    imocks2run = np.arange(100,150+1)
    # imocks2run = np.arange(100,100+1)

    # analysis = 'png_local'
    analysis = 'full_shape'
    
    tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']

    if analysis == 'full_shape':
        stats   = ['mesh2_spectrum', 'mesh3_spectrum']
        weights = ['default-noimsys-FKP','default-FKP']
    elif analysis == 'png_local':
        stats   = ['mesh2_spectrum']
        weights = ['default-noimsys-oqe','default-oqe']
    else:
        raise ValueError(f'{analysis} not supported.')

    stats_dir = Path(f'/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/') / analysis
    # stats_dir = Path(os.getenv('SCRATCH')) / 'glam-uchuu_mocks_validation' / analysis
    version = 'glam-uchuu-v1-altmtl'

    for tracer in tracers:
        for weight in weights:
            if True:
                exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
                imocks = exists[1]['imock']
                rerun = []
                for zrange in tools.propose_fiducial('zranges', tracer, analysis=analysis):
                    for kind in stats:
                        rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, kind=kind, stats_dir=stats_dir, tracer=tracer, region='GCcomb', weight=weight, zrange=zrange, version=version), test_if_readable=True, imock=imocks2run)
                        rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
                imocks = sorted(set(rerun))
            batch_imocks = np.array_split(imocks, max(len(imocks) // 10, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                run_stats_kws = dict(tracer=tracer, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats, analysis=analysis, weight=weight)
                if mode == 'interactive':
                    run_stats(**run_stats_kws)
                else:
                    _tm = tm if tracer in ['LRG'] else tm80
                    _tm.python_app(run_stats)(**run_stats_kws)
