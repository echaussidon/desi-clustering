"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test.py
"""
import os
import sys
import functools
from pathlib import Path

import jax
import numpy as np
import lsstypes as types

from clustering_statistics import box_tools, setup_logging, compute_box_stats_from_options


def test_stats_fn(stats=['mesh2_spectrum']):
    catalog_options = dict(version='abacus-hf-v2', tracer='LRG', cosmo='000', hod='base', zsnap=0.5, imock=1)
    for stat in stats:
        for kw in [{'los': 'x'}, {'los': 'z'}]:
            fn1 = box_tools.get_box_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = box_tools.get_box_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'

    catalog_options = dict(version='abacus-hf-v2', tracer=('LRG', 'ELG'), cosmo='000', hod='base', zsnap=0.950, imock=1)
    for stat in stats:
        for kw in [{'los': 'x'}, {'los': 'z'}]:
            fn1 = box_tools.get_box_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = box_tools.get_box_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'
            _catalog_options = dict(catalog_options)
            _catalog_options.pop('tracer')
            fn2 = box_tools.get_box_stats_fn(kind=stat, catalog={'LRG': _catalog_options, 'ELG': _catalog_options}, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'


def test_spectrum(stats=['mesh2_spectrum', 'mesh3_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        version = 'abacus-hf-v2'
        zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)
        for zsnap in zsnaps[:1]:
            catalog_options = dict(version=version, tracer=tracer, zsnap=zsnap, imock=1)
            compute_box_stats_from_options(stats, catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir), mesh3_spectrum={'basis': 'sugiyama-diagonal'})
            compute_box_stats_from_options(stats, catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir), mesh3_spectrum={'basis': 'scoccimarro', 'ells': [0, 2], 'edges': {'step': 0.2}})


def test_recon(stat='recon_particle2_correlation'):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        version = 'abacus-hf-v2'
        zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)
        for zsnap in zsnaps[:1]:
            catalog_options = dict(version=version, tracer=tracer, zsnap=zsnap, imock=1)
            compute_box_stats_from_options(stat, catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir))


def test_cross(stats=['mesh2_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in [('LRG', 'ELG')]:
        version = 'abacus-hf-v2'
        zsnap = 0.950
        catalog_options = dict(version=version, tracer=tracer, zsnap=zsnap, imock=1)
        compute_box_stats_from_options(stats, catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir))


def test_window(stats=['mesh2_spectrum']):
    from jaxpower import get_mesh_attrs
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for stat in stats:
        for tracer in ['LRG', ('LRG', 'ELG')]:
            version = 'abacus-hf-v2'
            zsnap = 0.950
            catalog_options = dict(version=version, tracer=tracer, zsnap=zsnap, imock=1)
            compute_box_stats_from_options([stat, f'window_{stat}'], catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir))


def test_covariance():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum']
    for tracer in ['LRG', 'ELG', ('LRG', 'ELG')]:
        version = 'abacus-hf-v2'
        zsnap = 0.950
        catalog_options = dict(version=version, tracer=tracer, zsnap=zsnap, imock=1)
        compute_box_stats_from_options(stats, catalog=catalog_options, get_box_stats_fn=functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir))


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    from jax import config
    config.update('jax_enable_x64', True)
    jax.distributed.initialize()
    setup_logging()

    #test_stats_fn()
    #test_spectrum()
    #test_recon()
    #test_cross()
    #test_window()
    test_covariance()
    #jax.distributed.shutdown()