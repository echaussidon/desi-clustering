import os
import logging
import functools
import copy
import warnings
from pathlib import Path
import itertools

import numpy as np
import jax
import lsstypes as types

from . import box_tools
from .box_tools import fill_box_fiducial_options, _merge_options, Catalog, setup_logging
from .correlation2_tools import compute_box_particle2_correlation
from .spectrum2_tools import compute_box_mesh2_spectrum, compute_window_box_mesh2_spectrum, compute_covariance_box_mesh2_spectrum, run_preliminary_fit_mesh2_spectrum
from .spectrum3_tools import compute_box_mesh3_spectrum
from .recon_tools import compute_box_reconstruction


logger = logging.getLogger('summary-statistics')


def compute_box_stats_from_options(stats, cache=None,
                                    get_box_stats_fn=box_tools.get_box_stats_fn,
                                    get_box_catalog_fn=None,
                                    read_clustering_box_catalog=box_tools.read_clustering_box_catalog,
                                    **kwargs):
    """
    Compute summary statistics based on the provided options.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to compute.
        Choices: ['mesh2_spectrum', 'mesh3_spectrum', 'recon_mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum']
    cache : dict, optional
        Cache to store intermediate results (binning class and parent/reference random catalog).
        See :func:`spectrum2_tools.compute_mesh2_spectrum`, :func:`spectrum3_tools.compute_mesh3_spectrum`,
        and func:`tools.read_clustering_box_catalog` for details.
    get_box_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    get_box_catalog_fn : callable, optional
        Function to get the filename for reading the catalog.
        If provided, it is given to ``read_clustering_box_catalog`` and ``read_full_catalog``.
    read_clustering_box_catalog : callable, optional
        Function to read the clustering catalog.
    **kwargs : dict
        Options for catalog, reconstruction, and summary statistics.
    """
    if isinstance(stats, str):
        stats = [stats]

    cache = cache or {}
    options = fill_box_fiducial_options(kwargs)
    catalog_options = options['catalog']
    tracers = list(catalog_options.keys())

    if get_box_catalog_fn is not None:
        read_clustering_box_catalog = functools.partial(read_clustering_box_catalog, get_box_catalog_fn=get_box_catalog_fn)

    with_recon = any('recon' in stat for stat in stats)
    with_catalogs = True
    data, shifted = {}, {}
    if with_catalogs:
        for tracer in tracers:
            _catalog_options = dict(catalog_options[tracer])
            data[tracer] = read_clustering_box_catalog(kind='data', **_catalog_options, concatenate=True)
            zsnap = data[tracer].attrs['zsnap']
            cmattrs = dict(boxsize=data[tracer].attrs['boxsize'], boxcenter=data[tracer].attrs['boxcenter'])

    # Reconstruction
    if with_recon:
        for tracer in tracers:
            recon_options = dict(options['recon'][tracer])
            recon_options.setdefault('zsnap', zsnap)
            # local sizes to select positions
            nran = recon_options.pop('nran', 1)
            recon_options['mattrs'] = cmattrs | recon_options.get('mattrs', {})
            data[tracer]['POSITION_REC'], randoms_rec_positions = compute_box_reconstruction(lambda: {'data': data[tracer]}, nran=nran, **recon_options)
            shifted[tracer] = [Catalog({'POSITION_REC': _positions, 'INDWEIGHT': np.ones_like(_positions[..., 0])}, mpicomm=data[tracer].mpicomm) for _positions in np.split(randoms_rec_positions, nran, axis=0)]

    fn_catalog_options = dict(catalog_options)

    def get_catalog_recon(catalog):
        return catalog.clone(POSITION=catalog['POSITION_REC'])

    # Summary statistics
    for recon in ['', 'recon_']:
        stat = f'{recon}particle2_correlation'
        if stat in stats:
            correlation_options = dict(options[stat])
            correlation_options['mattrs'] = cmattrs | correlation_options.get('mattrs', {})

            def get_data(tracer):
                if recon:
                    return {'data': get_catalog_recon(data[tracer]), 'shifted': [get_catalog_recon(random) for random in shifted[tracer]]}
                return {'data': data[tracer]}

            correlation = compute_box_particle2_correlation(*[functools.partial(get_data, tracer) for tracer in tracers], **correlation_options)
            fn = get_box_stats_fn(kind=stat, catalog=fn_catalog_options, **correlation_options)
            box_tools.write_stats(fn, correlation)

        funcs = {f'{recon}mesh2_spectrum': compute_box_mesh2_spectrum, f'{recon}mesh3_spectrum': compute_box_mesh3_spectrum}

        for stat, func in funcs.items():
            if stat in stats:
                spectrum_options = dict(options[stat])
                spectrum_options['mattrs'] = cmattrs | spectrum_options.get('mattrs', {})

                def get_data(tracer):
                    if recon:
                        cshifted = Catalog.concatenate(shifted[tracer])
                        return {'data': get_catalog_recon(data[tracer]), 'shifted': cshifted}
                    return {'data': data[tracer]}

                spectrum = func(*[functools.partial(get_data, tracer) for tracer in tracers], cache=cache, **spectrum_options)
                fn = get_box_stats_fn(kind=stat, catalog=fn_catalog_options, **spectrum_options)
                box_tools.write_stats(fn, spectrum)

    jax.experimental.multihost_utils.sync_global_devices('spectrum')  # such that spectrum ready for window

    # Window matrix
    funcs = {'window_mesh2_spectrum': compute_window_box_mesh2_spectrum}

    for stat, func in funcs.items():
        if stat in stats:
            window_options = dict(options[stat])
            window_options.setdefault('zsnap', data[tracers[0]].attrs['zsnap'])
            spectrum_fn = window_options.pop('spectrum', None)
            if spectrum_fn is None:
                spectrum_stat = stat.replace('window_', '')
                spectrum_fn = get_box_stats_fn(kind=spectrum_stat, catalog=fn_catalog_options, **options[spectrum_stat])
            spectrum = types.read(spectrum_fn)

            window = func(spectrum=spectrum, **window_options)
            fn = get_box_stats_fn(kind=stat, catalog=fn_catalog_options, **window_options)
            box_tools.write_stats(fn, window)

    # Covariance matrix
    funcs = {'covariance_mesh2_spectrum': compute_covariance_box_mesh2_spectrum}
    for stat, func in funcs.items():
        if stat in stats:
            covariance_options = dict(options[stat])
            covariance_options['mattrs'] = cmattrs | covariance_options.get('mattrs', {})
            theory_stat = stat.replace('covariance_', 'theory_')
            theory_fn = covariance_options.pop('theory', None)

            def _check_fn(fn, tracers, name=''):
                if len(tracers) == 1:
                    fn = {(tracer, tracer): fn for tracer in tracers}
                else:
                    raise ValueError(f'provide a dictionary of (tracer1, tracer2): {name} for tracer1, tracer2 in {tracers}')
                return fn

            def _read_tracer(fns, tracers2):
                if tracers2 not in fns: tracers2 = tracers2[::-1]
                return types.read(fns[tracers2])

            if theory_fn is None:
                products_fn = {}
                # Collect power spectrum and window
                for name in ['spectrum', 'window']:
                    kind_stat = stat.replace('covariance_', '') if name == 'spectrum' else stat.replace('covariance_', f'{name}_')
                    fn = covariance_options.pop(name, None)
                    if fn is None:
                        kw = options[kind_stat] | dict(auw=False, cut=False)
                        fn = {(tracer, tracer): get_box_stats_fn(kind=kind_stat, catalog=fn_catalog_options[tracer], **kw) for tracer in tracers}
                        if len(tracers) > 1:
                            fn[tuple(tracers)] = get_box_stats_fn(kind=kind_stat, catalog=fn_catalog_options, **kw)
                    elif not isinstance(fn, dict):
                        _check_fn(fn, tracers, name=name)
                    products_fn[name] = fn

                theory_fn = {}
                for tracers2 in itertools.combinations_with_replacement(tracers, r=2):
                    spectrum = _read_tracer(products_fn['spectrum'], tracers2)
                    window = _read_tracer(products_fn['window'], tracers2)
                    theory = run_preliminary_fit_mesh2_spectrum(data=spectrum, window=window)
                    theory_fn[tracers2] = get_box_stats_fn(kind=theory_stat, catalog=(fn_catalog_options[tracers2[0]] if tracers2[1] == tracers2[0] else {tracer: fn_catalog_options[tracer] for tracer in tracers2}))
                    box_tools.write_stats(theory_fn[tracers2], theory)
            else:
                _check_fn(theory_fn, tracers, name='theory')

            jax.experimental.multihost_utils.sync_global_devices('theory')  # such that theory ready for window
            fields = {tracer: box_tools.get_simple_tracer(tracer) for tracer in tracers}
            theory = {tuple(fields[tracer] for tracer in tracers2): _read_tracer(theory_fn, tracers2) for tracers2 in itertools.product(tracers, repeat=2)}
            theory = types.ObservableTree(list(theory.values()), fields=list(theory.keys()))
            covariance = func(theory=theory, **covariance_options)
            fn = get_box_stats_fn(kind=stat, catalog=fn_catalog_options, **covariance_options)
            box_tools.write_stats(fn, covariance)


def list_stats(stats, get_box_stats_fn=box_tools.get_box_stats_fn, **kwargs):
    """
    List measurements produced by :func:`compute_stats_from_options`.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to list.
    get_box_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='abacus-hf-v2', tracer='LRG', zsnap=0.500, imock=2)
            mesh2_spectrum = dict(ells=(0, 2, 4))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)])  # all arguments for compute_mesh3_spectrum
    """
    if isinstance(stats, str):
        stats = [stats]
    kwargs = fill_box_fiducial_options(kwargs)
    catalog_options = kwargs['catalog']

    toret = {stat: [] for stat in stats}
    for stat in stats:
        kw = dict(catalog=catalog_options, **kwargs[stat])
        fn = get_box_stats_fn(kind=stat, **kw)
        toret[stat].append((fn, kw))
    return toret