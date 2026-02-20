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

from clustering_statistics import tools, setup_logging, compute_stats_from_options


def test_stats_fn(stats=['mesh2_spectrum']):
    catalog_options = dict(version='holi-v1-altmtl', tracer='LRG', zrange=(0.4, 0.5), region='NGC', weight='default-FKP', imock=451)
    for stat in stats:
        for kw in [{'auw': True}, {'cut': True}]:
            fn1 = tools.get_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = tools.get_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'

    catalog_options = dict(version='holi-v1-altmtl', tracer=('LRG', 'ELG'), zrange=((0.4, 0.5), (0.4, 0.5)), region='NGC', weight='default-FKP', imock=451)
    for stat in stats:
        for kw in [{'auw': True}, {'cut': True}]:
            fn1 = tools.get_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = tools.get_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'
            _catalog_options = dict(catalog_options)
            _catalog_options.pop('tracer')
            zrange = _catalog_options.pop('zrange')
            fn2 = tools.get_stats_fn(kind=stat, catalog={'LRG': _catalog_options | dict(zrange=zrange[0]), 'ELG': _catalog_options | dict(zrange=zrange[1])}, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'


def test_auw(stats=['mesh2_spectrum']):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'cut': True, 'auw': True}, particle2_correlation={'auw': True})


def test_bitwise(stats=['mesh2_spectrum']):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-bitwise-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'cut': True, 'auw': True}, particle2_correlation={'auw': True})


def test_spectrum3(stats=['mesh3_spectrum']):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh3_spectrum={'basis': 'scoccimarro', 'ells': [0, 2]})


def test_recon(stat='recon_particle2_correlation'):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stat = 'mesh2_spectrum'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})


def test_expand_randoms(stat='mesh2_spectrum'):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stat = 'mesh2_spectrum'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            #catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='randoms', version='holi-v1-altmtl', tracer=tracer, region=region, nran=catalog_options['nran'], imock=catalog_options['imock'])})
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})
            fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                assert np.allclose(np.mean(spectrum.value()), 4749.380686357093)


def test_optimal_weights(stats=['mesh2_spectrum']):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)[:1]
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, particle2_correlation={}, analysis='png_local')


def test_cross(stats=['mesh2_spectrum']):
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC']:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, particle2_correlation={}, analysis='png_local')


def test_window(stats=['mesh2_spectrum']):
    from jaxpower import get_mesh_attrs
    from clustering_statistics.tools import propose_fiducial
    from clustering_statistics.spectrum3_tools import _get_window_edges
    mattrs = get_mesh_attrs(boxcenter=0., **propose_fiducial(kind='mesh2_spectrum', tracer='QSO')['mattrs'])
    #for edges in _get_window_edges(mattrs, scales=(1, 4)):
    #    print(edges, len(edges))
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for stat in stats:
        for tracer in ['LRG']:
            zranges = [(0.8, 1.1)]
            for region in ['NGC', 'SGC'][:1]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451, nran=1)
                #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
                compute_stats_from_options([stat, f'window_{stat}'][1:], catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, window_mesh2_spectrum={'cut': True})
        if 'mesh3' in stat: continue

        for tracer in [('LRG', 'ELG_LOPnotqso')]:
            zranges = [(0.8, 1.1)]
            for region in ['NGC', 'SGC'][:1]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451, nran=1)
                #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
                compute_stats_from_options([stat, f'window_{stat}'], catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, window_mesh2_spectrum={'cut': True}, analysis='png_local')


def test_covariance():
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][2:]
    for tracer in ['LRG', 'ELG_LOPnotqso'][:0]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True})
    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True})


def test_norm():
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stat = 'mesh3_spectrum'
    for tracer in ['BGS_BRIGHT-21.5']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC']:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', nran=2)
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir))
            fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                print(spectrum.get((0, 0, 0)).values('norm').mean())
                assert np.allclose(spectrum.get((0, 0, 0)).values('norm').mean(), 1.28543918)


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from jax import config
    config.update('jax_enable_x64', False)
    jax.distributed.initialize()
    setup_logging()

    test_covariance()
    #test_window(stats=['mesh3_spectrum'])

    test_stats_fn()
    test_auw(stats=['mesh2_spectrum'])
    test_bitwise(stats=['mesh2_spectrum'])
    test_expand_randoms()
    test_optimal_weights()
    test_cross()
    #test_window()
    #test_spectrum3()
    test_norm()
    test_recon()
    #jax.distributed.shutdown()