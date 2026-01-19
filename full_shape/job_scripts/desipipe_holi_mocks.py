"""
Script to create and spawn desipipe tasks to compute clustering measurements on HOLI mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_holi_mocks.py  # create the list of tasks
desipipe tasks -q holi_mocks  # check the list of tasks
desipipe spawn -q holi_mocks --spawn  # spawn the jobs
desipipe queues -q holi_mocks  # check the queue
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

queue = Queue('holi_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/holi_mocks/slurm-%j.out', 'slurm_outputs/holi_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi') #, command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=25), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))


def run_stats(tracer='LRG', version='holi-v1-altmtl', imocks=[451], meas_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum']):
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
    zranges = tools.propose_fiducial('zranges', tracer)
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock), mesh2_spectrum={'cut': True, 'auw': True})
            options = fill_fiducial_options(options)
            compute_fiducial_stats_from_options(stats, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            _options_imock = dict(options)
            _options_imock['catalog'] = _options_imock['catalog'] | dict(imock=imock)
            combine_fiducial_stats_from_options(stats, region_comb, regions, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), **_options_imock)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    todo = ['exists', 'slurm']
    #todo = ['interactive']
    #todo = ['slurm']
    imocks = np.arange(1001)

    meas_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks_desipipe'
    version = 'holi-v1-altmtl'

    for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO']:
        if 'exists' in todo:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=list(range(1001)))[:2]
            imocks = exists[1]['imock']
            batch_imocks = np.array_split(imocks, len(imocks) // 10)
        for _imocks in batch_imocks:
            if 'interactive' in todo:
                run_stats(tracer, version=version, imocks=_imocks, meas_dir=meas_dir, stats=['mesh2_spectrum'])
            elif 'slurm' in todo:
                _tm = tm if tracer in ['LRG'] else tm80
                _tm.python_app(run_stats)(tracer, version=version, imocks=_imocks, meas_dir=meas_dir, stats=['mesh2_spectrum', 'mesh3_spectrum'])
