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

stats_dir = Path(os.getenv('CFS')) / 'cai' / 'holi_lightcone_validation'
plots_dir = Path('./_plots')


def run_stats(tracer='LRG', zranges=None, version='holi-v4.80', weight='default-FKP', imocks=[0], stats_dir=stats_dir, stats=['mesh2_spectrum'], get_catalog_fn=tools.get_catalog_fn):
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


def plot_density(imock=[0], tracer='LRG', zranges=None, version='holi-v4.80', weight='default', plots_dir=plots_dir, nside=64, get_catalog_fn=tools.get_catalog_fn):
    from clustering_statistics.density_tools import plot_density_projections
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    region = 'ALL'
    
    for zrange in [None]:
        edges = {'RA': np.linspace(0., 360., 361),
                 'DEC': np.linspace(-90., 90., 181)}
        def read_catalog(kind='data', **kwargs):
            fn = get_catalog_fn(kind=kind, **kwargs)
            return tools._read_catalog(fn)
        catalog = dict(version=version, tracer=tracer, zrange=zrange, region=region, weight=weight)
        plot_density_projections(get_catalog_fn=get_catalog_fn, read_catalog=read_catalog, divide_randoms='same', catalog=catalog,
                                 imock=imock, edges={name: edges[name] for name in ['RA', 'DEC']}, fn=plots_dir / f'angular_density_fluctuations_{version}_weight-{weight}_{tracer}_{region}.png', nside=nside, map_q=(0.1, 0.9))

    for zrange in zranges:
        zstep = 0.01
        edges = {'Z': np.arange(zrange[0], zrange[1] + zstep, zstep),
                 'RA': np.linspace(0., 360., 361),
                 'DEC': np.linspace(-90., 90., 181)}
        plot_density_projections(get_catalog_fn=get_catalog_fn, divide_randoms=True, catalog=catalog,
                                 imock=imock, edges=edges, fn=plots_dir / f'density_fluctuations_{version}_weight-{weight}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png', nside=nside)
        plot_density_projections(get_catalog_fn=get_catalog_fn, divide_randoms=False, catalog=catalog,
                                 imock=imock, edges=edges, fn=plots_dir / f'density_{version}_weight-{weight}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png', nside=nside)


def fit_large_scales(imock=0, tracer='LRG', zranges=None, version='holi-v4.80', weight='default', stats_dir=stats_dir, plots_dir=plots_dir, get_catalog_fn=tools.get_catalog_fn):
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    import lsstypes as types
    from jaxpower import MeshAttrs, create_sharding_mesh, BinMesh2SpectrumPoles, compute_mesh2_spectrum_mean, split_particles
    from clustering_statistics.spectrum2_tools import run_preliminary_fit_mesh2_spectrum, prepare_jaxpower_particles

    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)

    for zrange in zranges[-1:]:
        for region in ['NGC', 'SGC'][:1]:
            kw_catalog = dict(imock=imock, tracer=tracer, zrange=zrange, region=region, version=version, weight=weight)
            # Preliminary fit to the data
            fns = tools.get_stats_fn(kind='mesh2_spectrum_poles', stats_dir=stats_dir, catalog=kw_catalog | dict(imock='*'))
            spectrum = types.read(fns[0])
            mean = types.mean([types.read(fn) for fn in fns])
            spectrum = spectrum.clone(value=mean.value())
            fn = tools.get_stats_fn(kind='window_mesh2_spectrum_poles', stats_dir=stats_dir, catalog=kw_catalog)
            window = types.read(fn)
            mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
            k_edges = np.linspace(1e-6, 0.3, 1001)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles
            out = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=np.zeros_like(k), ell=ell) for ell in spectrum.ells])
            theory = run_preliminary_fit_mesh2_spectrum(spectrum, window, select={'k': (0.02, 0.08)}, theory='kaiser', fixed=['sn0'], out=out)
            fn = tools.get_stats_fn(kind='theory_mesh2_spectrum_poles', stats_dir=stats_dir, catalog=kw_catalog)
            tools.write_stats(fn, theory)

            mattrs = mattrs.clone(meshsize=512)
            with create_sharding_mesh(meshsize=mattrs.meshsize):
                # Then feed this to mean power spectrum computation
                bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=spectrum.ells)
                data = tools.read_clustering_catalog(kind='data', **kw_catalog, get_catalog_fn=get_catalog_fn)
                randoms = tools.read_clustering_catalog(kind='randoms', **kw_catalog, get_catalog_fn=get_catalog_fn)
                _, randoms, _ = prepare_jaxpower_particles(lambda: (data, randoms), mattrs=mattrs)[0]
                pole = next(iter(spectrum))
                kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
                meshes = []
                for iran, randoms in enumerate(split_particles([randoms, None], seed=42, fields=[1] * 2)):
                    randoms = randoms.exchange(backend='mpi')
                    alpha = pole.attrs['wsum_data'][0][min(iran, 0)] / randoms.weights.sum()
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))
                mean = compute_mesh2_spectrum_mean(meshes, theory=(theory, 'local'), los='firstpoint', bin=bin)
                with_gic = True
                if with_gic:
                    spectrumw0 = mean.value()[0]
                    k_edges = np.linspace(0., 0.5 * mattrs.kfun.min(), 2)
                    k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
                    k = np.mean(k_edges, axis=-1)
                    # inject spike at ell == 0
                    theory0 = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=np.zeros_like(k) + (ell == 0), ell=ell) for ell in spectrum.ells])
                    window0 = compute_mesh2_spectrum_mean(meshes, theory=(theory0, 'local'), los='firstpoint', bin=bin)
                    window0 = window0.clone(value=window0.value() / window0.value()[0])
                    mean = mean.clone(value=mean.value() - spectrumw0 * window0.value())
                norm = pole.values('norm').mean()
                mean = mean.clone(norm=norm * np.ones_like(mean.value()))
                fn = tools.get_stats_fn(kind='mean_mesh2_spectrum_poles', stats_dir=stats_dir, catalog=kw_catalog)
                tools.write_stats(fn, mean)


if __name__ == '__main__':

    #todo = ['test']
    #todo = ['density']
    todo = ['large_scales']
    weight = 'default'
    stats = ['mesh2_spectrum', 'mesh3_spectrum_sugiyama', 'mesh3_spectrum_scoccimarro', 'window_mesh2_spectrum'][:1]

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

    def get_holi_catalog_fn(kind='data', tracer='LRG', imock=0, version='v4.00', **kwargs):
        if version == 'v4.00':
            cat_dir = Path(f'/dvs_ro/cfs/cdirs/desi/mocks/cai/holi/{version}/') / f'seed{imock:04d}'
        else:
            cat_dir = Path(f'/dvs_ro/cfs/cdirs/desi/mocks/cai/holi/webjax_{version}/') / f'seed{imock:04d}'
        if kind == 'data':
            return cat_dir / f'holi_{tracer}_{version}_GCcomb_clustering.dat.h5'
        if kind == 'randoms':
            return [cat_dir / f'holi_{tracer}_{version}_GCcomb_0_clustering.ran.h5']
    
    if 'test' in todo:
        #version = 'v4.00'
        version = 'v4.80'
        tracers = ['LRG', 'ELG', 'QSO'][:1]

        imocks = []
        for imock in range(1000):
            if all(get_holi_catalog_fn(kind='data', tracer=tracer, version=version, imock=imock).exists() for tracer in tracers for kind in ['data']):
                imocks.append(imock)
            if len(imocks) > 9: break
        print(f'Running {imocks}')

        for tracer in tracers:
            if any('window' in stat for stat in stats):
                imocks = [1]
            run_stats(tracer, version=version, weight=weight, stats=stats, stats_dir=stats_dir, get_catalog_fn=get_holi_catalog_fn, imocks=imocks)

    if 'density' in todo:
        version = 'v4.80'
        #version = 'v4.00'
        tracers = ['LRG', 'ELG', 'QSO'][:1]

        def get_holi_catalog_fn(kind='data', tracer='LRG', imock=0, version='v4.00', nran=1, **kwargs):
            if version == 'v4.00':
                cat_dir = Path(f'/dvs_ro/cfs/cdirs/desi/mocks/cai/holi/{version}/') / f'seed{imock:04d}'
            else:
                cat_dir = Path(f'/dvs_ro/cfs/cdirs/desi/mocks/cai/holi/webjax_{version}/') / f'seed{imock:04d}'
            if kind == 'data':
                return cat_dir / f'holi_{tracer}_{version}_GCcomb_clustering.dat.h5'
            if kind == 'randoms':
                return [f'/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/rands_intiles_DARK_nomask_{iran:d}.fits' for iran in range(nran)]
        
        imocks = []
        for imock in range(1000):
            if all(get_holi_catalog_fn(kind=kind, tracer=tracer, version=version, imock=imock).exists() for tracer in tracers for kind in ['data']):
                imocks.append(imock)
            if len(imocks) > 40: break
        print(f'Running {imocks}')
        for tracer in tracers:
            plot_density(imock=imocks, tracer=tracer, version=version, weight='default', get_catalog_fn=get_holi_catalog_fn)

    if 'large_scales' in todo:
        version = 'v4.80'
        #version = 'v4.00'
        tracers = ['LRG', 'ELG', 'QSO'][:1]
        for tracer in tracers:
            fit_large_scales(imock=1, tracer=tracer, version=version, weight='default', stats_dir=stats_dir, plots_dir=plots_dir, get_catalog_fn=get_holi_catalog_fn)