import logging

import jax

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles


logger = logging.getLogger('reconstruction')


def compute_reconstruction(get_data_randoms, mattrs=None, mode='recsym', bias=2.0, smoothing_radius=15.):
    """
    Compute density field reconstruction using :mod:`jaxrecon`.

    Parameters
    ----------
    get_data_randoms : callable
        Functions that return dict of 'data', 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    mode : {'recsym', 'reciso'}, optional
        Reconstruction mode. 'recsym' removes large-scale RSD from randoms, 'reciso' does not.
    bias : float, optional
        Linear bias of the tracer.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h for the density field.

    Returns
    -------
    data_positions_rec : np.ndarray
        Reconstructed data positions.
    randoms_positions_rec : np.ndarray
        Reconstructed randoms positions.
    """
    from jaxpower import create_sharding_mesh, FKPField
    from jaxrecon.zeldovich import IterativeFFTReconstruction, estimate_particle_delta

    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        particles = prepare_jaxpower_particles(get_data_randoms, mattrs=mattrs, return_inverse=True)[0]

        # Define FKP field = data - randoms
        fkp = FKPField(particles['data'], particles['randoms'])
        del particles
        delta = estimate_particle_delta(fkp, smoothing_radius=smoothing_radius)
        # Line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
        # In case of IterativeFFTParticleReconstruction, and multi-GPU computation, provide the size of halo regions in cell units. E.g., maximum displacement is ~ 40 Mpc/h => 4 * chosen cell size => provide halo_add=2
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        growth_rate = compute_fkp_effective_redshift(fkp.data, order=1, cellsize=None, func_of_z=cosmo.growth_rate)
        recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'halo_add', 'niterations'], donate_argnums=[0])(delta, growth_rate=growth_rate, bias=bias, los=None, halo_add=0)
        data_positions_rec = recon.read_shifted_positions(fkp.data.positions)
        assert mode in ['recsym', 'reciso']
        # RecSym = remove large scale RSD from randoms
        kwargs = {}
        if mode == 'recsym': kwargs['field'] = 'disp'
        randoms_positions_rec = recon.read_shifted_positions(fkp.randoms.positions, **kwargs)
        if jax.process_index() == 0:
            logger.info('Reconstruction finished.')

        data_positions_rec = fkp.data.exchange_inverse(data_positions_rec)
        randoms_positions_rec = fkp.randoms.exchange_inverse(randoms_positions_rec)
        if jax.process_index() == 0:
            logger.info('Exchange finished.')

    return data_positions_rec, randoms_positions_rec



def compute_box_reconstruction(get_data, mattrs=None, mode='recsym', zsnap=None, bias=2.0, smoothing_radius=15., nran=10, los='z'):
    """
    Compute density field reconstruction in box using :mod:`jaxrecon`.

    Parameters
    ----------
    get_data : callable
        Functions that return dict of 'data' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    mode : {'recsym', 'reciso'}, optional
        Reconstruction mode. 'recsym' removes large-scale RSD from randoms, 'reciso' does not.
    zsnap : float, optional
        Snapshot redshift, used to estimate the growth rate.
    bias : float, optional
        Linear bias of the tracer.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h for the density field.
    nran : int, optional
        Number of randoms per data particle to generate for reconstruction.
    los : str or array-like, optional
        Line-of-sight specification ('x', 'y', 'z',).

    Returns
    -------
    data_positions_rec : np.ndarray
        Reconstructed data positions.
    randoms_positions_rec : np.ndarray
        Reconstructed randoms positions.
    """
    from jaxpower import create_sharding_mesh, ParticleField, generate_uniform_particles
    from jaxrecon.zeldovich import IterativeFFTReconstruction, estimate_particle_delta

    mattrs = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        data = prepare_jaxpower_particles(get_data, mattrs=mattrs, return_inverse=True)[0]['data']
        mattrs = data.attrs
        delta = estimate_particle_delta(data, smoothing_radius=smoothing_radius)

        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        growth_rate = cosmo.growth_rate(zsnap)

        recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'halo_add', 'niterations'], donate_argnums=[0])(delta, growth_rate=growth_rate, bias=bias, los=los, halo_add=0)
        data_positions_rec = recon.read_shifted_positions(data.positions)
        randoms = generate_uniform_particles(mattrs, size=nran * data.size, seed=(42, 'index'), exchange=True, backend='mpi', return_inverse=True)
        assert mode in ['recsym', 'reciso']
        # RecSym = remove large scale RSD from randoms
        kwargs = {}
        if mode == 'recsym': kwargs['field'] = 'disp'
        randoms_positions_rec = recon.read_shifted_positions(randoms.positions, **kwargs)
        if jax.process_index() == 0:
            logger.info('Reconstruction finished.')

        data_positions_rec = data.exchange_inverse(data_positions_rec)
        randoms_positions_rec = randoms.exchange_inverse(randoms_positions_rec)
        if jax.process_index() == 0:
            logger.info('Exchange finished.')

    return data_positions_rec, randoms_positions_rec