import time
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
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
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
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        attrs = _get_jaxpower_attrs(*all_particles)
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs
        # Define the binner
        if cache is None: cache = {}
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Computing normalization
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=[(42, fkp.randoms.__dict__['IDS']) for fkp in all_fkp],  # index for process invariance
                                          cellsize=10)

        # Computing shot noise
        all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
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
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)

        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')

    return spectrum


def _get_window_edges(mattrs, scales: tuple=(1, 4)):
    """Return window edges."""
    distmax, cellmin = np.sqrt(np.sum(mattrs.boxsize**2)), mattrs.cellsize.min()
    nsizes, cellsizes = [6] * 5 + [None], [cellmin * 2**i for i in range(6)]
    edges = []
    for scale in scales:
        edges_scale = []
        start = 0.
        for nsize, cellsize in zip(nsizes, cellsizes):
            cellsize = cellsize * scale
            if nsize is None:
                tmp = np.arange(start, distmax * scale / scales[-1] + cellsize, cellsize)
            else:
                tmp = start + np.arange(nsize) * cellsize
            if tmp.size:
                start = tmp[-1] + cellsize
                edges_scale.append(tmp)
        edges_scale = np.concatenate(edges_scale, axis=0)
        edges_scale = edges_scale[edges_scale < distmax + cellsize]
        edges.append(edges_scale)
    return edges


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
    from jaxpower import (create_sharding_mesh, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation,
                           compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles)
    ells = spectrum.ells
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    #mattrs['meshsize'] = 256
    los = spectrum.attrs['los']
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles
        mattrs = all_randoms[0].attrs

        pole = next(iter(spectrum))
        ells, edges, basis = spectrum.ells, pole.edges('k'), pole.basis
        norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in spectrum.ells])
        k, index = np.unique(pole.coords('k', center='mid_if_edges')[..., 0], return_index=True)
        edges = edges[index, 0]
        edges = np.insert(edges[:, 1], 0, edges[0, 0])
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')   # mask_edges useless if cellsize is large enough
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
        kw['ells'] = [ell for ell in kw['ells'] if all(ell <= 2 for ell in ell)]  # let's remove some terms...
        jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])
        #jitted_compute_mesh3_correlation = compute_mesh3_correlation

        coords = jnp.logspace(-3, 5, 1024)
        list_scales = [1, 4]
        list_edges = _get_window_edges(mattrs, scales=list_scales)

        ells = kw['ells']
        if ibatch is not None:
            start, stop = ibatch[0] * len(ells) // ibatch[1], (ibatch[0] + 1) * len(ells) // ibatch[1]
            kw['ells'] = ells[start:stop]
        if ells and not bool(computed_batches):
            # multigrid calculation
            for scale, edges in zip(list_scales, list_edges):
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}')
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
                #kw['ells'] = kw['ells'][-1:]
                #kw['ells'] = [(2, 0, 2)]
                #kw['ells'] = [(0, 0, 0)]
                #edges = edges[:5]
                buffer_size = len(edges) - 1
                #buffer_size = 0
                sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=buffer_size)  # kcut=(0., mattrs2.knyq.min()))
                meshes = []
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (3 - len(all_randoms)),
                                                               seed=seed, fields=fields)):
                    randoms = randoms.exchange(backend='mpi')
                    alpha = pole.attrs['wsum_data'][0][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))
                t0 = time.time()
                correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
                jax.block_until_ready(correlation)
                if jax.process_index() == 0:
                    logger.info(f"Computed windows {kw['ells']}, scale {scale}, in {time.time() - t0:.2f} s.")
                    #correlation.write(f'_tmp/window_mesh3_correlation_raw_{scale}.h5')
                correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
                correlations.append(correlation)

            coords = list(next(iter(correlations[0])).coords().values())
            masks = [(coords[0] < edges[-1])[:, None] * (coords[1] < edges[-1])[None, :] for edges in list_edges[:-1]]
            masks.append((coords[0] < np.inf)[:, None] * (coords[1] < np.inf)[None, :])
            weights = []
            for mask in masks:
                if len(weights):
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            correlation = correlations[0].sum(correlations, weights=weights)

        if computed_batches:
            correlation = types.join(computed_batches)
            correlation = types.join([correlation.get(ells=[ell]) for ell in ells])  # reorder

        jax.block_until_ready(correlation)
        if jax.process_index() == 0:
            logger.info('Window functions computed.')

        results = {}
        results['window_mesh3_correlation_raw'] = correlation
        if ibatch is None:
            if jax.process_index() == 0:
                logger.info('Building window matrix.')
            window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',), batch_size=4)
            observable = window.observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs | dict(zeff=zeff)), input_label=True)
            window = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))  # just in case norm is k-dependent
            results['raw'] = window
    return results



def compute_box_mesh3_spectrum(*get_data, mattrs=None,
                                basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='z',
                                buffer_size=0, cache=None):
    r"""
    Compute the 3-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return tuples of (data, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects.
        See :func:`prepare_jaxpower_particles` for details.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_shotnoise, compute_box3_normalization, BinMesh3SpectrumPoles, compute_mesh3_spectrum, compute_fkp3_shotnoise)

    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        if cache is None: cache = {}
        if edges is None: edges = {'step': 0.001}
        attrs = _get_jaxpower_attrs(*all_particles)
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs

        # Define the binner
        basis = 'sugiyama-diagonal' if all(isinstance(ell, tuple) for ell in ells) else 'scoccimarro'
        if cache is None: cache = {}
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Computing normalization
        all_data = [particles['data'] for particles in all_particles]
        norm = compute_box3_normalization(*all_data, bin=bin)

        # Computing shot noise
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        del all_particles
        num_shotnoise = compute_fkp3_shotnoise(*all_fkp, bin=bin, fields=None)

        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory
        all_mesh = []
        for fkp in all_fkp:
            mesh = fkp.paint(**kw, out='real')
            all_mesh.append(mesh - mesh.mean())
        del all_fkp
        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])
        #jitted_compute_mesh3_spectrum = compute_mesh3_spectrum
        spectrum = jitted_compute_mesh3_spectrum(*all_mesh, bin=bin, los=los)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum