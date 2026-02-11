"""
Script to create and spawn desipipe tasks to compute clustering measurements on HOLI mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 validation_holi_mocks.py
```
"""
import os
from pathlib import Path

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()


def run_stats(tracer='LRG', version='holi-v1-altmtl', weight='default-FKP', imocks=[451], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum']):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', False)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, combine_stats_from_options, fill_fiducial_options

    setup_logging()
    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)[-1:]
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), mesh2_spectrum={'cut': True, 'auw': True if 'altmtl' in version else None})
            options = fill_fiducial_options(options)
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            combine_stats_from_options(stats, region_comb, regions, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **options)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    imocks = 451 + np.arange(25)

    stats_dir = Path(os.getenv('SCRATCH')) / 'holi_mocks_validation'

    for tracer in ['LRG']:
        for weight in ['default_compntile', 'default']:
            run_stats(tracer, version='holi-v1-complete', weight=weight, imocks=imocks, stats_dir=stats_dir)
        run_stats(tracer, version='holi-v1-altmtl', weight=weight, imocks=imocks, stats_dir=stats_dir)
