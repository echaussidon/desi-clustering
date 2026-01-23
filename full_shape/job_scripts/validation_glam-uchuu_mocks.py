"""
Script to create and spawn desipipe tasks to compute clustering measurements on glam mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_glam_mocks.py  # create the list of tasks
desipipe tasks -q glam_mocks  # check the list of tasks
desipipe spawn -q glam_mocks --spawn  # spawn the jobs
desipipe queues -q glam_mocks  # check the queue
```
"""
import os
from pathlib import Path
import functools
import sys

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

sys.path.insert(0, '../')
import tools

setup_logging()


def run_stats(tracer='LRG', version='glam-v1-altmtl', weight='default_FKP', imocks=[451], meas_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum']):
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
    sys.path.insert(0, '../')
    import tools
    from tools import setup_logging
    from compute_fiducial_stats import compute_fiducial_stats_from_options, combine_fiducial_stats_from_options, fill_fiducial_options

    setup_logging()
    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)[-1:]
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), mesh2_spectrum={'cut': True, 'auw': True if 'altmtl' in version else None})
            options = fill_fiducial_options(options)
            options['catalog']['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog']['nran'])}
            compute_fiducial_stats_from_options(stats, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            _options_imock = dict(options)
            _options_imock['catalog'] = _options_imock['catalog'] | dict(imock=imock)
            combine_fiducial_stats_from_options(stats, region_comb, regions, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), **_options_imock)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    imocks = 100 + np.arange(5)

    meas_dir = Path(os.getenv('SCRATCH')) / 'glam-uchuu_mocks_validation'

    for tracer in ['LRG']:
        #for weight in ['default_compntile', 'default'][1:]:
        #    run_stats(tracer, version='glam-uchuu-v1-complete', weight=weight, imocks=imocks, meas_dir=meas_dir)
        weight = 'default'
        run_stats(tracer, version='glam-uchuu-v1-altmtl', weight=weight, imocks=imocks, meas_dir=meas_dir)
