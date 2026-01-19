"""
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_holi_mocks.py  # create the list of tasks
desipipe tasks -q holi_mocks  # check the list of tasks
desipipe spawn -q holi_mocks --spawn  # spawn the jobs
desipipe queues -q holi_mocks  # check the queue
"""

import numpy as np

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

setup_logging()

queue = Queue('holi_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/holi_mocks/slurm-%j.out', 'slurm_outputs/holi_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi') #, command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=25), provider=dict(provider='nersc', time='01:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='01:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))


def run_stats(tracer='LRG', imocks=[451], stats=['mesh2_spectrum']):
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
    #meas_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks2'
    meas_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks_desipipe2'
    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)
    for imock in imocks:
        regions = ['NGC', 'SGC'][:1]
        for region in regions:
            options = dict(catalog=dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=imock), mesh2_spectrum={'cut': True, 'auw': True})
            options = fill_fiducial_options(**options)
            compute_fiducial_stats_from_options(stats, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), cache=cache, **options)
        jax.experimental.multihost_utils.sync_global_devices('measurements')
        for region_comb, regions in tools.possible_combine_regions(regions).items():
            _options_imock = dict(options)
            _options_imock['catalog'] = _options_imock['catalog'] | dict(imock=imock)
            combine_fiducial_stats_from_options(stats, region_comb, regions, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), **_options_imock)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    #mode = 'interactive'
    mode = 'slurm'
    imocks = 451 + np.arange(250)
    batch_imocks = np.array_split(imocks, len(imocks) // 10)

    for imocks in batch_imocks[1:]:
        for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO']:
            if 'interactive' in mode:
                run_stats(tracer, imocks=imocks, stats=['mesh2_spectrum'])
            else:
                _tm = tm if tracer in ['LRG'] else tm80
                _tm.python_app(run_stats)(tracer, imocks=imocks, stats=['mesh2_spectrum', 'mesh3_spectrum'])
