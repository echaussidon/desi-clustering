import logging

import jax

from spectrum2_tools import prepare_jaxpower_particles, _get_jaxpower_attrs


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
    if bin is None or not np.all(bin.mattrs.meshsize, mattrs.meshsize) or not np.all(bin.mattrs.boxsize, mattrs.boxsize):
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