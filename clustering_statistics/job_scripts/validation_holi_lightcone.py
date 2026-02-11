"""
Script to create and spawn desipipe tasks to validate HOLI lightcones.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new
srun -n 4 validation_holi_lightcone.py
```
"""
import os
from pathlib import Path
import functools
import sys

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()


def run_stats(tracer='LRG', zranges=None, version='glam-uchuu-v1-altmtl', weight='default-FKP', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], get_catalog_fn=tools.get_catalog_fn):
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
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    for stat in stats:
        kw = dict(mesh3_spectrum=dict())
        if 'mesh3' in stat:
            if 'scoccimarro' in stat:
                kw['mesh3_spectrum'].update({'basis': 'scoccimarro', 'ells': [0, 2]})
                stat = 'mesh3_spectrum'
            else:
                stat = 'mesh3_spectrum'
        for imock in imocks:
            regions = ['NGC', 'SGC'][:1]
            for region in regions:
                options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), mesh2_spectrum={'cut': True, 'auw': True if 'altmtl' in version or 'data' in version else None}, **kw)
                options = fill_fiducial_options(options, analysis='full_shape_protected' if 'data' in version else 'full_shape')
                if 'uchuu-hf-complete' in version:
                    for tracer in options['catalog']:
                        options['catalog'][tracer]['nran'] = min(options['catalog'][tracer]['nran'], 4) 
                compute_stats_from_options(stat, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), get_catalog_fn=get_catalog_fn, cache=cache, **options)
            jax.experimental.multihost_utils.sync_global_devices('measurements')
            for region_comb, regions in tools.possible_combine_regions(regions).items():
                combine_stats_from_options(stat, region_comb, regions, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **options)
    #jax.distributed.shutdown()


if __name__ == '__main__':

    stats_dir = Path(os.getenv('CFS')) / 'cai' / 'holi_lightcone_validation'

    todo = ['ref']
    weight = 'default'
    stats = ['mesh2_spectrum', 'mesh3_spectrum_sugiyama', 'mesh3_spectrum_scoccimarro'][-1:]

    if 'ref' in todo:
        imocks = list(range(5))
        for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO'][:0]:
            run_stats(tracer, version='data-dr2-v2', weight=weight, stats=stats, stats_dir=stats_dir)
            ##run_stats(tracer, version='uchuu-hf-altmtl', weight=weight, stats=stats, stats_dir=stats_dir)
            ##run_stats(tracer, version='abacus-2ndgen-altmtl', weight=weight, stats=stats, imocks=imocks, stats_dir=stats_dir)
        for tracer in ['LRG', 'ELG_LOP', 'QSO'][-1:]:
            run_stats(tracer, version='uchuu-hf-complete', weight=weight, stats=stats, stats_dir=stats_dir)
            run_stats(tracer, version='abacus-2ndgen-complete', weight=weight, stats=stats, imocks=imocks, stats_dir=stats_dir)

        kw = dict(stats=['mesh2_spectrum'], zranges=[(0.8, 1.1)], version='data-dr2-v2', weight=weight)
        run_stats(('LRG', 'ELG_LOPnotqso'), **kw, stats_dir=stats_dir)
        run_stats(('LRG', 'QSO'), **kw, stats_dir=stats_dir)
        run_stats(('ELG_LOPnotqso', 'QSO'), **kw, stats_dir=stats_dir)

    if 'test' in todo:
        def get_catalog_fn(kind='data', tracer='LRG', **kwargs):
            cat_dir = Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/holi/webjax_v4.30/seed0003')
            if kind == 'data':
                return cat_dir / f'holi_{tracer}_v4.30_GCcomb_clustering.dat.h5'
            if kind == 'randoms':
                return [cat_dir / f'holi_{tracer}_v4.30_GCcomb_0_clustering.ran.h5']

        for tracer in ['LRG']:
            run_stats(tracer, version='holi-v4.30', weight=weight, stats=stats, stats_dir=stats_dir, get_catalog_fn=get_catalog_fn)