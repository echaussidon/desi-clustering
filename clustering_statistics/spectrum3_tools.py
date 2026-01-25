import logging

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles, _get_jaxpower_attrs


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum(*get_data_randoms, mattrs=None,
                            basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='local',
                            buffer_size=0, cache=None):
    r"""
    Compute the 3-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects. If None, default attributes are used.
        See :func:`prepare_jaxpower_particles` for details.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'sugiyama-diagonal'.
    ells : list of tuples, optional
        List of multipole moments to compute. Default is [(0, 0, 0), (2, 0, 2)] (for the sugiyama basis).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.005 h/\mathrm{Mpc}` is used for the sugiyama basis, :math:`0.01 h/\mathrm{Mpc}` for the scoccimarro basis.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'local', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    buffer_size : int, optional
        Buffer size when binning; if the binning is multidimensional, increase for faster computation at the cost of memory.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
    attrs = _get_jaxpower_attrs(*all_particles)
    mattrs = all_particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh3_spectrum', None)
    if edges is None: edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault('bin_mesh3_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in all_particles]
    norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=[(42, fkp.randoms.__dict__['IDS']) for fkp in all_fkp],  # index for process invariance
                                      cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in all_particles]
    del all_particles
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)

    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')

    # out='real' to save memory
    meshes = [fkp.paint(**kw, out='real') for fkp in all_fkp]
    del all_fkp

    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])

    # out='real' to save memory
    spectrum = jitted_compute_mesh3_spectrum(*meshes, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    return spectrum



def compute_window_mesh3_spectrum(*get_data_randoms, spectrum, ibatch: tuple=None, computed_batches: list=None, buffer_size=10):
    r"""
    Compute the 3-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles.
    ibatch : tuple, optional
        To split the window function multipoles to compute in batches, provide (0, nbatches) for the first batch,
        (1, nbatches) for the second, etc; up to (nbatches - 1, nbatches).
        ``None`` to compute the final window matrix.
    computed_batches : list, optional
        The window function multipoles that have been computed thus far.

    Returns
    -------
    spectrum : WindowMatrix or dict of WindowMatrix
        The computed 3-point spectrum window.
    """
    # FIXME: data is not used, could be dropped, add auw
    from jaxpower import (BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation,
                           compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles, compute_fkp_effective_redshift)

    ells = spectrum.ells
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

    all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
    all_randoms = [particles[1] for particles in all_particles]
    del all_particles

    pole = next(iter(spectrum))
    ells, norm, edges, basis = spectrum.ells, pole.values('norm'), pole.edges('k'), pole.basis
    _, index = np.unique(pole.coords('k', center='mid')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')
    stop = bin.edges1d[0].max()
    step = np.diff(bin.edges1d[0], axis=-1).min()
    edgesin = np.arange(0., 1.5 * stop, step / 2.)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    fields = list(range(len(all_randoms)))
    fields += [fields[-1]] * (3 - len(all_randoms))
    seed = [(42, randoms.__dict__['IDS']) for randoms in all_randoms]
    zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed)

    correlations = []
    kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=fields, return_ellsin=True)
    jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])

    coords = jnp.logspace(-3, 5, 4 * 1024)
    scales = [1, 4]
    b, c = mattrs.boxsize.min(), mattrs.cellsize.min()
    edges = [np.concatenate([np.arange(11) * c, np.arange(11 * c, 0.3 * b, 4 * c)]),
            np.concatenate([np.arange(11) * scales[1] * c, np.arange(11 * scales[1] * c, 2 * b, 4 * scales[1] * c)])]

    ells = kw['ells']
    if ibatch is not None:
        start, stop = ibatch[0] * len(ells) // ibatch[1], (ibatch[0] + 1) * len(ells) // ibatch[1]
        kw['ells'] = ells[start:stop]
    if ells and not bool(computed_batches):
        # multigrid calculation
        for scale, edges in zip(scales, edges):
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=buffer_size)  # kcut=(0., mattrs2.knyq.min()))
            meshes = []
            for iran, randoms in enumerate(split_particles([randoms.clone(attrs=mattrs2, exchange=True, backend='mpi') for randoms in all_randoms] + [None] * (3 - len(all_randoms)),
                                                            seed=seed, fields=fields)):
                    alpha = pole.attrs[f'wsum_data{iran:d}'] / randoms.weights.sum()
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))
            correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
            correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
            correlations.append(correlation)

        coords = list(next(iter(correlations[0])).coords().values())
        limit = 0.25 * mattrs.boxsize.min()
        mask = (coords[0] < limit)[:, None] * (coords[1] < limit)[None, :]
        weights = [jnp.maximum(mask, 1e-6), jnp.maximum(~mask, 1e-6)]
        correlation = correlations[0].sum(correlations, weights=weights)

    if computed_batches:
        correlation = types.join(computed_batches)
        correlation = types.join([correlation.get(ells=[ell]) for ell in ells])  # reorder

    results = {}
    results['window_mesh3_correlation_raw'] = correlation
    if ibatch is None:
        window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
        observable = window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')),
                                                                    attrs=pole.attrs | dict(zeff=zeff)))
        window = window.clone(observable=observable, value=window.value / (norm[..., None] / np.mean(norm)))  # just in case norm is k-dependent
        results['raw'] = window
    return results