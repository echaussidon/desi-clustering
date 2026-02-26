import logging
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import default_mpicomm, _format_bitweights, compute_fkp_effective_redshift, combine_stats


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, add_data=tuple(), add_randoms=tuple(), **kwargs):
    """
    Prepare :class:`jaxpower.ParticleField` objects from data and randoms catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'randoms', 'shifted') catalogs.
        Each catalog must contain 'POSITION' and 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights and 'TARGETID'
        for randoms IDs to allow process-invariant random split in bispectrum normalization.
    mattrs : dict, optional
        Mesh attributes ('boxsize', 'meshsize' or 'cellsize', 'boxcenter') to define the :class:`ParticleField` objects. If ``None``, default attributes are used.
    kwargs : dict, optional
        Additional keyword arguments to pass to :class:`ParticleField`.

    Returns
    -------
    all_particles : list of dictionaries
        List of dictionaries of :class:`ParticleField`  'data' (optionally 'randoms', 'shifted') objects for each input catalog.
    """
    from jaxpower.mesh import get_mesh_attrs, ParticleField
    backend = 'mpi'
    mpicomm = kwargs['mpicomm']

    all_catalogs = [_get_data_randoms() for _get_data_randoms in get_data_randoms]

    # Define the mesh attributes; pass in positions only
    mattrs = get_mesh_attrs(*[catalog['POSITION'] for catalogs in all_catalogs for catalog in catalogs.values()], check=True, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    def collective_arange(local_size):
        sizes = mpicomm.allgather(local_size)
        return sum(sizes[:mpicomm.rank]) + np.arange(local_size)

    all_particles = []
    add = {'data': add_data, 'randoms': add_randoms}
    for catalogs in all_catalogs:
        particles = {}
        for name, catalog in catalogs.items():
            _add = {}
            indweights = catalog['INDWEIGHT']
            if name == 'data':
                bitweights = None
                if 'BITWEIGHT' in catalog and 'BITWEIGHT' in add[name]:
                    bitweights = _format_bitweights(catalog['BITWEIGHT'])
                    from cucount.jax import BitwiseWeight
                    iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
                    _add['BITWEIGHT'] = [indweights] + bitweights  # add individual weight (photometric, spectro systematics) without PIP
                    indweights = indweights * iip  # multiply by IIP to correct fiber assignment at large scales
                for column in add[name]:
                    if column != 'BITWEIGHT': _add[column] = catalog[column]
            elif name == 'randoms':
                if 'TARGETID' in catalog and 'IDS' in add[name]:
                    _add['IDS'] = catalog['TARGETID']
                for column in add[name]:
                    if column != 'IDS': _add[column] = catalog[column]
            particle = ParticleField(catalog['POSITION'], indweights, attrs=mattrs, exchange=True, backend=backend, **kwargs)
            for key, value in _add.items():
                if isinstance(value, list): value = [particle.exchange_direct(value, pad=0) for value in value]
                else: value = particle.exchange_direct(value, pad=0)
                particle.__dict__[key] = value
            particles[name] = particle
        all_particles.append(particles)
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*all_particles):
    """Return summary attributes from :class:`jaxpower.ParticleField` objects: total weight and size."""
    mattrs = next(iter(all_particles[0].values())).attrs
    # Creating FKP fields
    attrs = {}
    for particles in all_particles:
        for name in particles:
            if particles[name] is not None:
                if f'wsum_{name}' not in attrs:
                    #attrs[f'size_{name}'] = [[]]  # size is process-dependent
                    attrs[f'wsum_{name}'] = [[]]
               # attrs[f'size_{name}'][0].append(particles[name].size)
                attrs[f'wsum_{name}'][0].append(particles[name].sum())
    for name in ['boxsize', 'boxcenter', 'meshsize']:
        attrs[name] = mattrs[name]
    return attrs


def compute_mesh2_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', optimal_weights=None,
                           cache=None):
    r"""
    Compute the 2-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    cut : bool, optional
        If True, apply a theta-cut of (0, 0.05) in degrees.
    auw : ObservableTree, optional
        Angular upweights to apply. If ``None``, no angular upweights are applied.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles or dict of Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles. If `cut` or `auw` are provided, returns a dict with keys 'raw', 'cut', and/or 'auw'.
    """

    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    columns_optimal_weights = []
    if optimal_weights is not None:
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=['BITWEIGHT'] + columns_optimal_weights, add_randoms=columns_optimal_weights)

        if cache is None: cache = {}
        if edges is None: edges = {'step': 0.001}

        def _compute_spectrum_ell(all_particles, ells, fields=None):
            # Compute power spectrum for input given multipoles
            attrs = _get_jaxpower_attrs(*all_particles)
            attrs.update(los=los)
            mattrs = all_particles[0]['data'].attrs

            # Define the binner
            key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
            bin = cache.get(key, None)
            if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
                bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
            cache.setdefault(key, bin)

            # Computing normalization
            all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
            norm = compute_fkp2_normalization(*all_fkp, bin=bin, cellsize=10)

            # Computing shot noise
            all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
            del all_particles
            num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=fields)

            jax.block_until_ready((norm, num_shotnoise))
            if jax.process_index() == 0:
                logger.info('Normalization and shotnoise computation finished')

            results = {}
            # First compute the theta-cut pairs
            if cut is not None:
                sattrs = {'theta': (0., 0.05)}
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
                from jaxpower.particle2 import convert_particles
                all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
                close = compute_particle2(*all_particles, bin=pbin, los=los)
                close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=norm)
                close = close.to_spectrum(bin.xavg)
                results['cut'] = -close.value()

            # Then compute the AUW-weighted pairs
            with_bitweights = 'BITWEIGHT' in all_fkp[0].data.__dict__
            if auw is not None or with_bitweights:
                from cucount.jax import WeightAttrs
                from jaxpower.particle2 import convert_particles
                sattrs = {'theta': (0., 0.1)}
                bitwise = angular = None
                if with_bitweights:
                    # Order of weights matters
                    # fkp.data.__dict__['BITWEIGHT'] includes IIP in the first position
                    all_data = [convert_particles(fkp.data, weights=list(fkp.data.__dict__['BITWEIGHT']) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
                    bitwise = dict(weights=all_data[0].get('bitwise_weight'))  # sets nrealizations, etc.: fine to use the first
                    if jax.process_index() == 0:
                        logger.info(f'Applying PIP weights {bitwise}.')
                else:
                    all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
                if auw is not None:
                    angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
                    if jax.process_index() == 0:
                        logger.info(f'Applying AUW {angular}.')
                wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
                pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
                DD = compute_particle2(*all_data, bin=pbin, los=los)
                DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=pbin, fields=fields), norm=norm)
                results['auw'] = DD.value()

            jax.block_until_ready(results)
            if jax.process_index() == 0:
                logger.info(f'Particle-based calculation finished')

            kw = dict(resampler='tsc', interlacing=3, compensate=True)
            # out='real' to save memory
            all_mesh = [fkp.paint(**kw, out='real') for fkp in all_fkp]
            del all_fkp

            # JIT the mesh-based spectrum computation; helps with memory footprint
            jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
            #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
            spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
            spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
            spectrum = spectrum.clone(attrs=attrs)
            jax.block_until_ready(spectrum)
            if jax.process_index() == 0:
                logger.info('Mesh-based computation finished')

            # Add theta-cut and AUW contributes
            for name, value in results.items():
                results[name] = spectrum.clone(value=spectrum.value() + value)
            results['raw'] = spectrum

            return results

        if optimal_weights is None:
            results = _compute_spectrum_ell(all_particles, ells=ells)
        else:
            results = {}
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                fields = tuple(range(len(all_particles)))
                fields = fields + (fields[-1],) * (2 - len(fields))
                all_particles = tuple(all_particles) + (all_particles[-1],) * (2 - len(all_particles))

                def _get_optimal_weights(all_particles):
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    for all_weights in optimal_weights(ell, [{'INDWEIGHT': particles.weights} | {column: particles.__dict__[column] for column in columns_optimal_weights} for particles in all_particles]):
                        yield tuple(particles.clone(weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                names = list(all_particles[0].keys())
                for _all_particles in zip(*[_get_optimal_weights([particles[name] for particles in all_particles]) for name in names]):
                    # _all_particles is a list [(data1, data2), (randoms1, randoms2), [(shifted1, shifted2)]] of tuples of ParticleField with optimal weights applied
                    _all_particles = list(zip(*_all_particles))
                    # _all_particles is now a list of tuples [(data1, randoms1, shifted1), (data2, randoms2, shifted2)] with optimal weights applied
                    _all_particles = [dict(zip(names, _particles)) for _particles in _all_particles]
                    # _all_particles is now a list of dictionaries [{'data': data1, 'randoms': randoms1, 'shifted': shifted1}, {'data': data2, 'randoms': randoms2, 'shifted': shifted2}] with optimal weights applied
                    _result = _compute_spectrum_ell(_all_particles, ells=[ell], fields=fields)
                    for key in _result:  # raw, cut, auw
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                for key, value in result_ell.items():
                    results.setdefault(key, [])
                    results[key].append(combine_stats(value))  # sum 1<->2
            for key in results:
                results[key] = types.join(results[key])  # join multipoles

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def compute_window_mesh2_spectrum(*get_data_randoms, spectrum: types.Mesh2SpectrumPoles, optimal_weights: Callable=None, cut
: bool=None):
    r"""
    Compute the 2-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.

    Returns
    -------
    window : WindowMatrix or dict of WindowMatrix
        The computed 2-point spectrum window. If `auw` is provided, returns a dict with keys 'raw' and 'auw'.
    """
    # FIXME: data is not used, could be dropped, add auw
    from jaxpower import (create_sharding_mesh, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise,
                           compute_smooth2_spectrum_window, get_smooth2_window_bin_attrs, interpolate_window_function, split_particles)

    ells = spectrum.ells
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ellsin = [0, 2, 4]
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

    columns_optimal_weights = []
    if optimal_weights is not None:
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'] + columns_optimal_weights)
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles

        stop, step = -np.inf, np.inf
        for pole in spectrum:
            edges = pole.edges('k')
            stop = max(edges.max(), stop)
            step = min(np.nanmin(np.diff(edges, axis=-1)), step)
        edgesin = np.arange(0., 1.2 * stop, step)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        def _compute_window_ell(all_randoms, ells, isum=0, fields=None):
            all_randoms = list(all_randoms)
            seed = [(42, randoms.__dict__['IDS']) for randoms in all_randoms]  # for process invariance
            mattrs = all_randoms[0].attrs
            pole = spectrum.get(ells[0])
            bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
            # Get normalization from input power spectrum
            norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells], axis=0)
            results = {}
            correlations = []
            kw_window = get_smooth2_window_bin_attrs(ells, ellsin)
            jitted_compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0])
            # Window computed in configuration space, summing Bessel over the Fourier-space mesh
            coords = jnp.logspace(-3, 5, 4 * 1024)
            list_edges = []
            for scale in [1, 4]:
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                all_mesh = []
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    randoms = randoms.exchange(backend='mpi')
                    alpha = pole.attrs['wsum_data'][isum][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                    all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))
                edges = np.arange(0., mattrs2.boxsize.min() / 2., mattrs2.cellsize.min())
                list_edges.append(edges)
                sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw_window, basis='bessel')
                correlation = jitted_compute_mesh2_correlation(all_mesh, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
                del all_mesh
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                correlations.append(correlation)
            masks = [coords < edges[-1] for edges in list_edges[:-1]]
            masks.append((coords < np.inf))
            weights = []
            for mask in masks:
                if len(weights):
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            results['window_mesh2_correlation_raw'] = correlation = correlations[0].sum(correlations, weights=weights)

            window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
            observable = window.observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs), input_label=True)
            results['raw'] = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))  # just in case norm is k-dependent
            if cut:
                sattrs = {'theta': (0., 0.05)}
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, **kw_window)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, **kw_window)
                from jaxpower.particle2 import convert_particles
                all_particles = []
                for iran, randoms in enumerate(all_randoms):
                    alpha = pole.attrs['wsum_data'][isum][iran] / randoms.weights.sum()
                    all_particles.append(convert_particles(randoms.clone(weights=alpha * randoms.weights)))
                correlation = compute_particle2(*all_particles, bin=pbin, los=los)
                correlation = correlation.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=[np.mean(norm)] * len(sbin.ells))
                pole = next(iter(correlation))
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                results['window_mesh2_correlation_cut'] = correlation
                correlation = correlation.clone(value=results['window_mesh2_correlation_raw'].value() + correlation.value())
                window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
                results['cut'] = window.clone(observable=results['raw'].observable, value=window.value() / (norm[..., None] / np.mean(norm)))
            for key, result in results.items():
                if 'correlation' in key:
                    results[key] = types.ObservableTree([result], oells=[ells[0] if len(ells) == 1 else tuple(ells)])
            return results

        if optimal_weights is None:
            # Compute effective redshift
            fields = None
            seed = [(42, randoms.__dict__['IDS']) for randoms in all_randoms]
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields, return_fraction=True)
            results = _compute_window_ell(all_randoms, ells=ells, fields=fields)
            for key in results:
                if 'correlation' not in key:
                    observable = results[key].observable
                    observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                    results[key] = results[key].clone(observable=observable)
        else:
            results = {}
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                fields = tuple(range(len(all_randoms)))
                fields = fields + (fields[-1],) * (2 - len(fields))
                all_randoms = tuple(all_randoms) + (all_randoms[-1],) * (2 - len(all_randoms))

                def _get_optimal_weights(all_particles):
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    def clone(particles, weights):
                        toret = particles.clone(weights=weights)
                        toret.__dict__.update(particles.__dict__)  # to keep IDS
                        return toret
                    for all_weights in optimal_weights(ell, [{'INDWEIGHT': particles.weights} | {column: particles.__dict__[column] for column in columns_optimal_weights} for particles in all_particles]):
                        yield tuple(clone(particles, weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                for isum, all_randoms in enumerate(_get_optimal_weights(all_randoms)):
                    fields = None
                    seed = [(42, randoms.__dict__['IDS']) for randoms in all_randoms]
                    zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields, return_fraction=True)
                    _result = _compute_window_ell(all_randoms, ells=[ell], isum=isum, fields=fields)
                    for key in _result:  # raw, cut, auw
                        if 'correlation' not in key:
                            observable = _result[key].observable
                            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                            _result[key] = _result[key].clone(observable=observable)
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                for key, windows in result_ell.items():
                    results.setdefault(key, [])
                    window = combine_stats(windows)  # sum 1<->2
                    # Used power spectrum norm is for the sum of the two;
                    # just sum the two components
                    window = window.clone(value=sum(window.value() for window in windows))
                    results[key].append(combine_stats(windows))
            for key in results:
                if 'correlation' in key:
                    results[key] = types.join(results[key])
                else:
                    observables = [window.observable for window in results[key]]
                    observable = types.join(observables)
                    value = np.concatenate([window.value() for window in results[key]], axis=0)
                    results[key] = results[key][0].clone(value=value, observable=observable)  # join multipoles

    return results


def run_preliminary_fit_mesh2_spectrum(data: types.Mesh2SpectrumPoles, window: types.WindowMatrix, select: dict=None, theory: str='rept', fixed=tuple(), out: types.Mesh2SpectrumPoles=None):
    """
    Compute a smooth theory spectrum to assume when building the covariance.

    Parameters
    ----------
    data : Mesh2SpectrumPoles or None
        Measured spectrum multipoles used to build the covariance and (optionally)
        to set priors / initialize the fit. If None, the function will still
        construct an analytic covariance from `window` but cannot use data-driven
        priors.
    window : WindowMatrix
        Window matrix describing mode-coupling of the estimator. The window's
        observable axes are matched to the `data` before fitting.
    select : dict, optional
        If provided, a selection is applied to `data` via `data.select(**select)`
        prior to fitting (e.g. to restrict k-ranges or multipoles).
    theory : str, optional
        Theory to use in the fit, one of ['rept', 'kaiser'].
    out : Mesh2SpectrumPoles, optional
        If provided, returns a clone of these power spectrum multipoles with best fit theory values.

    Returns
    -------
    out : Mesh2SpectrumPoles
    """
    from jaxpower import MeshAttrs, compute_spectrum2_covariance
    smooth = data.select(k=(0.001, 10.))

    mattrs = MeshAttrs(**{name: data.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    covariance = compute_spectrum2_covariance(mattrs, data)  # Gaussian, diagonal covariance

    select = select or {'k': (0.02, 10.)}
    data = data.select(**select)
    window = window.at.observable.match(data)
    window = window.at.theory.select(k=(0.001, 1.2 * next(iter(data)).coords('k').max()))
    covariance = covariance.at.observable.match(data)
    z = window.observable.get(ells=0).attrs['zeff']

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    Theory = {'rept': REPTVelocileptorsTracerPowerSpectrumMultipoles, 'kaiser': KaiserTracerPowerSpectrumMultipoles}[theory]

    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=z)
    theory = Theory(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(data=data.value(concatenate=True), wmatrix=window.value(), ells=data.ells,
                                                         k=[pole.coords('k') for pole in data], kin=window.theory.get(ells=0).coords('k'),
                                                         ellsin=window.theory.ells, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())
    for param in fixed:
        likelihood.all_params[param].update(fixed=True)

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    params = profiles.bestfit.choice(index='argmax', input=True)
    if out is None:
        theory.init.update(k=smooth.get(0).coords('k'))
        poles = theory(**params)
        smooth = smooth.clone(value=poles.ravel())
    else:
        value = []
        for label, pole in out.items(level=1):
            theory.init.update(k=pole.coords('k'))
            value.append(theory(**params)[theory.ells.index(label['ells'])])
        smooth = out.clone(value=value)
    return smooth


def compute_covariance_mesh2_spectrum(*get_data_randoms, theory=None, fields=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    theory : Mesh2SpectrumPoles
        Theory 2-point spectrum multipoles.
    fields : tuple, list, optional
        Field names.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    from jaxpower import create_sharding_mesh, compute_fkp2_covariance_window, interpolate_window_function, compute_spectrum2_covariance, FKPField
    fftlog = True
    if fields is None:
        fields = list(range(1, 1 + len(get_data_randoms)))
    results = {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs = all_fkp[0].attrs
        kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
        kw.update(los='local', fields=fields)
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        windows = compute_fkp2_covariance_window(all_fkp, **kw, **kw_paint)
        if fftlog:
            coords = np.logspace(-2, 8, 8 * 1024)
            windows = windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
        results['window_covariance_mesh2_correlation'] = windows

    # delta is the maximum abs(k1 - k2) where the covariance will be computed (to speed up calculation)
    covariance = compute_spectrum2_covariance(windows, theory, flags=['smooth'] + (['fftlog'] if fftlog else []))
    # Update label names
    fields = covariance.observable.fields
    observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
    covariance = covariance.clone(observable=observable)
    results['raw'] = covariance
    return results


def compute_rotation_mesh2_spectrum(window: types.WindowMatrix, covariance: types.CovarianceMatrix, Minit: str='momt',
                                    data: types.Mesh2SpectrumPoles=None, theory: types.Mesh2SpectrumPoles=None, select: dict=None):
    """
    Compute the rotation to make the window matrix more diagonal.

    Parameters
    ----------
    window : WindowMatrix
        Window matrix.
    covariance : CovarianceMatrix
        Covariance of the measured spectrum.
    Minit : {'momt', ...}, optional
        Initialization method passed to rotation.setup(Minit=...). Defaults to 'momt'.
    data : Mesh2SpectrumPoles or None, optional
        Measured spectrum used to set priors for the rotation (if available).
    theory : Mesh2SpectrumPoles or None, optional
        Theory spectrum used together with `data` when setting priors.

    Returns
    -------
    rotation : WindowRotationSpectrum2
    """
    from jaxpower import WindowRotationSpectrum2
    observable = window.observable
    if data is not None:
        if select is not None:
            data = data.select(**select)
        observable = data
    window = window.at.observable.match(observable)
    if theory is not None:
        def interpolate_pole(ref, pole):
            return ref.clone(value=np.interp(ref.coords('k'), pole.coords('k'), pole.value()))

        theory = window.theory.map(lambda pole, label: interpolate_pole(pole, theory.get(ells=label['ells'])), input_label=True, level=1)
    covariance = covariance.at.observable.match(observable)
    rotation = WindowRotationSpectrum2(window=window, covariance=covariance, xpivot=0.1)
    rotation.setup(Minit=Minit)
    rotation.fit()
    if rotation.with_momt and data is not None:
        # To set up priors
        rotation.set_prior(data=data, theory=theory)
    return rotation


def compute_box_mesh2_spectrum(*get_data, ells=(0, 2, 4), edges=None, los='z', cache=None, mattrs=None):
    r"""
    Compute the 2-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_shotnoise, compute_box2_normalization, BinMesh2SpectrumPoles, compute_mesh2_spectrum, compute_fkp2_shotnoise)

    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        if cache is None: cache = {}
        if edges is None: edges = {'step': 0.001}
        attrs = _get_jaxpower_attrs(*all_particles)
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs

        # Define the binner
        key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
        bin = cache.get(key, None)
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        cache.setdefault(key, bin)

        # Computing normalization
        all_data = [particles['data'] for particles in all_particles]
        norm = compute_box2_normalization(*all_data, bin=bin)

        # Computing shot noise
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        del all_particles
        num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=None)

        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory
        all_mesh = []
        for fkp in all_fkp:
            mesh = fkp.paint(**kw, out='real')
            all_mesh.append(mesh - mesh.mean())
        del all_fkp
        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
        #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
        spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum


def compute_window_box_mesh2_spectrum(spectrum: types.Mesh2SpectrumPoles, zsnap: float=None):
    r"""
    Compute the 2-point spectrum window for a box (i.e., binning window) with :mod:`jaxpower`.

    Parameters
    ----------
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.

    Returns
    -------
    window : WindowMatrix
        The computed 2-point spectrum window.
    """
    from jaxpower import create_sharding_mesh, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_window

    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ells = spectrum.ells
    pole = spectrum.get(0)
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
        #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
        edgesin = bin.edges
        window = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ells, los=los, bin=bin)
        observable = window.observable
        if zsnap is not None:
            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zsnap, zsnap=zsnap)))
        window = window.clone(observable=observable)
    return window


def compute_covariance_box_mesh2_spectrum(theory: types.Mesh2SpectrumPoles=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance for a box with :mod:`jaxpower`.

    Parameters
    ----------
    theory : Mesh2SpectrumPoles, optional
        Theory spectrum used together with `spectrum` when setting priors.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    from jaxpower import create_sharding_mesh, MeshAttrs, compute_spectrum2_covariance
    # Add shotnoise to theory
    theory_sn = theory.map(lambda pole: pole.clone(num_shotnoise=pole.values('num_shotnoise') * 0.), level=2)
    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        covariance = compute_spectrum2_covariance(mattrs, theory_sn)  # Gaussian, diagonal covariance
    
        # Update label names
        fields = covariance.observable.fields
        observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
        covariance = covariance.clone(observable=observable)
    return covariance