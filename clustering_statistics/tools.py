import os
import logging
from pathlib import Path
import warnings
from collections.abc import Callable
import itertools
import functools

import numpy as np
from mpi4py import MPI
import jax
from jax import numpy as jnp
from mockfactory import Catalog, sky_to_cartesian, cartesian_to_sky, setup_logging
import lsstypes as types


logger = logging.getLogger('tools')


desi_dir = Path('/dvs_ro/cfs/cdirs/desi/')


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def default_mpicomm(func: Callable):
    """Wrapper to provide a default MPI communicator."""
    @functools.wraps(func)
    def wrapper(*args, mpicomm=None, **kwargs):
        if mpicomm is None:
            from mpi4py import MPI
            mpicomm = MPI.COMM_WORLD
        return func(*args, mpicomm=mpicomm, **kwargs)

    return wrapper


def load_footprint():
    #global footprint
    from regressis import footprint
    footprint = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
    return footprint


def join_tracers(tracers):
    """Given list/tuple of input tracers, return joined string with 'x' separator."""
    if not isinstance(tracers, str):
        return 'x'.join(tracers)
    return tracers


def get_simple_tracer(tracer):
    """Given import tracer, return simple tracer name; e.g. 'ELG_LOPnotqso' would result in 'ELG'."""
    if 'BGS' in tracer:
        return 'BGS'
    elif 'LRG+ELG' in tracer:
        return 'LRG+ELG'
    elif 'LRG' in tracer:
        return 'LRG'
    elif 'ELG' in tracer:
        return 'ELG'
    elif 'QSO' in tracer:
        return 'QSO'
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')


def get_lensing_options(sample):
    # get options for different lensing maps based on given sample (str)
    # following https://github.com/cosmodesi/DESI_Y3_x_CMB/tree/28bf7661a6ed02f81397d5db93d87344cd47d0d2/configs
    options = {'healpix_nside': 2048}
    if sample == 'act_dr6':
        base_dir = "/dvs_ro/cfs/projectdirs/act/www/dr6_lensing_v1/"
        options['file'] = os.path.join(base_dir,'maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits')
        options['is_cmb_mask'] = True
        options['galactic_coordinates'] = False
        return options
    if sample == 'planck_pr4':
        base_dir = "/dvs_ro/cfs/cdirs/cmb/data/planck2020/PR4_lensing/"
        options['file'] = os.path.join(base_dir, "mask.fits.gz")
        options['is_cmb_mask'] = False
        options['galactic_coordinates'] = True
        return options
    raise ValueError('unknown lensing sample {}'.format(sample))


def get_lensing_footprint(sample, threshold=0.1):
    # https://github.com/cosmodesi/DESI_Y3_x_CMB/blob/28bf7661a6ed02f81397d5db93d87344cd47d0d2/DESI_Y3_x_CMB/auxiliary/config_utils.py#L59
    import healpy as hp
    lensing_options = get_lensing_options(sample)
    mask_path = lensing_options['file']
    lensing_mask = hp.ud_grade(hp.read_map(mask_path, dtype=np.float32), lensing_options['healpix_nside']) # This is slow
    if lensing_options['is_cmb_mask']:
        lensing_mask *= lensing_mask

    uses_galactic_coords = lensing_options["galactic_coordinates"]
    if uses_galactic_coords:
        rotator = hp.Rotator(coord=['G','C'])
        lensing_mask = rotator.rotate_map_pixel(lensing_mask)
    lensing_mask = hp.reorder(lensing_mask,r2n=True)
    return lensing_mask > threshold


def select_region(ra, dec, region=None):
    """
    Return mask of corresponding R.A./Dec. region.

    Parameters
    ----------
    ra : array_like
        R.A. coordinates in degrees.
    dec : array_like
        Dec. coordinates in degrees.
    region : str, optional
        Region to select. Options are:
        - None, 'ALL', 'GCcomb': all-sky
        - 'NGC': North Galactic Cap
        - 'SGC': South Galactic Cap
        - 'N': Northern region (Dec > 32.375 and in NGC)
        - 'S': Southern region (Dec > -25 and in SGC)
        - 'SNGC': Southern part of NGC
        - 'SSGC': Southern part of SGC
        - 'NGCnoN': NGC excluding Northern region
        - 'DES': DES footprint
        - 'SnoDES': Southern region excluding DES footprint
        - 'SSGCnoDES': Southern part of SGC excluding DES footprint
        - 'SGCnoDES': SGC excluding DES footprint
        - 'ACT_DR6': ACT DR6 footprint
        - 'PLANCK_PR4': Planck PR4 footprint

    Returns
    -------
    mask : array_like
        Boolean mask array indicating the selected region.
    """
    import healpy as hp
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')

    # North, South, SGC, and NGC footprints
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if region == 'NGCnoN':
        return mask_ngc & (~mask_n)
    # if region == 'GCcomb_noNorth':
    #     return ~mask_n

    # DES footprint
    north, south, des = load_footprint().get_imaging_surveys()
    mask_des = des[hp.ang2pix(hp.get_nside(des), ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    if region == 'SGCnoDES':
        return (~mask_ngc) & (~mask_des)
    # if region == 'GCcomb_noDES':
    #     return ~mask_des

    # Other footprints
    act = get_lensing_footprint(region.lower())
    mask_act = act[hp.ang2pix(hp.get_nside(act), ra, dec, nest=True, lonlat=True)]
    if region == 'ACT_DR6':
        return mask_act
    planck = get_lensing_footprint(region.lower())
    mask_planck = planck[hp.ang2pix(hp.get_nside(planck), ra, dec, nest=True, lonlat=True)]
    if region == 'PLANCK_PR4':
        return mask_planck
    raise ValueError('unknown region {}'.format(region))


def _make_tuple(item, n=None):
    if not isinstance(item, (list, tuple)):
        item = (item,)
    item = tuple(item)
    if n is not None:
        item = item + (item[-1],) * (n - len(item))
    return item


def compute_fiducial_selection_weights(catalog, stat='mesh3_spectrum', tracer=None):
    """
    Apply fiducial selection weight to the catalog based on the statistic being computed.
    For the bispectrum (mesh3), the individual weights are scaled by NX^(-1/3) to make the
    effective redshift comparable to that of the power spectrum (mesh2).
    """
    if 'mesh3' in stat:
        catalog = catalog.clone(INDWEIGHT=catalog['INDWEIGHT'] * catalog['NX']**(-1. / 3.))
    return catalog


@jax.jit
def lininterp_1d(xp: jax.Array, y: jax.Array, xmin: jax.Array=0., step: jax.Array=1.):
    """
    xp are query points, y is a 1D array of samples spaced by `step` starting at `xmin`.

    Parameters
    ----------
    xp : array-like
        Query points.
    y : array-like
        Sampled values on a regular grid.
    xmin : float, optional
        Grid starting coordinate.
    step : float, optional
        Grid spacing.

    Returns
    -------
    array-like
        Interpolated values at xp using linear interpolation with clipping at boundaries.
    """
    fidx = (xp - xmin) / step
    idx = jnp.floor(fidx).astype(jnp.int16)
    fidx -= idx
    toret = (1 - fidx) * y[idx] + fidx * y[idx + 1]
    #return jnp.where((idx >= 0) & (idx < len(y)), toret, 0.)
    toret = jnp.where(idx < 0, y[0], toret)
    toret = jnp.where(idx > len(y) - 1, y[-1], toret)
    return toret


def get_interpolator_1d(x: jax.Array, y: jax.Array, order: int=1):
    """
    Return a 1D interpolator function for arrays x, y.

    For linear regular grids (constant spacing) a JIT-able linear interpolator is
    returned; for other cases or higher `order`, an interpolator based on interpax is used.

    Parameters
    ----------
    x : 1D array-like
        Grid points (monotonic).
    y : 1D array-like
        Values at grid points.
    order : int, optional
        Interpolation order (1 for linear). Higher orders use an external Interpolator.

    Returns
    -------
    callable
        A function that takes query points and returns interpolated values (JAX arrays).
    """
    xmin, xmax = x[0], x[-1]
    step = (xmax - xmin) / (len(x) - 1)
    is_lin = np.allclose(x, step * np.arange(len(x)) + xmin)

    if order == 1:
        if is_lin:

            def interp(xp):
                return lininterp_1d(xp, y=y, xmin=xmin, step=step)
        else:

            def interp(xp):
                return jnp.interp(xp, x, y)

    else:
        from interpax import Interpolator1D
        interpolator = Interpolator1D(x, jnp.asarray(y), method={1: 'linear', 3: 'cubic2'}[order], extrap=False, period=None)

        @jax.jit
        def interp(xp):
            # clip
            toret = interpolator(xp)
            #return jnp.where((xp >= xmin) & (xp <= xmax), toret, 0.)
            toret = jnp.where(xp < xmin, y[0], toret)
            toret = jnp.where(xp > xmax, y[-1], toret)
            return toret

    return interp


def compute_fiducial_png_weights(ell, catalog, tracer='LRG', p=1.):
    """Return total optimal weights for local PNG analysis."""
    from jax import numpy as jnp
    from cosmoprimo.fiducial import DESI
    from interpax import Interpolator1D

    def bias(z, tracer='QSO'):
        """Bias model for the different DESI tracer (measured from DR2 data (loa/v2))."""
        params = {'BGS_BRIGHT-21.35': (0.60646037, 0.52389492),
                  'LRG': (0.23553567, 1.3458994),
                  'ELG_LOPnotqso': (0.15066781, 0.59463735),
                  'ELGnotqso': (0.15487521, 0.59464828),
                  'ELG': (0.15487521, 0.59464828),
                  'QSO': (0.25207547, 0.71020952)}
        params.update({f'{key}_zcmb': value for key, value in params.items()})

        if tracer in params:
            alpha, beta = params[tracer]
        else:
            raise ValueError(f'Bias for {tracer} is not ready!')
        return alpha * (1 + z)**2 + beta

    cosmo = DESI()
    zstep = 0.001
    zgrid = np.arange(0., 10. + zstep, zstep)
    growth_factor = get_interpolator_1d(zgrid, cosmo.growth_factor(zgrid), order=1)
    growth_rate = get_interpolator_1d(zgrid, cosmo.growth_rate(zgrid), order=1)

    tracers = _make_tuple(tracer, n=2)
    catalogs = _make_tuple(catalog, n=2)
    ps = _make_tuple(p, n=2)

    def _get_weights(catalogs, tracers, ps):
        wtilde = bias(catalogs[0]['Z'], tracer=tracers[0]) - ps[0]
        w0 = growth_factor(catalogs[1]['Z']) * (bias(catalogs[1]['Z'], tracer=tracers[1]) + growth_rate(catalogs[1]['Z']) / 3)
        w2 = 2 / 3 * growth_factor(catalogs[1]['Z']) * growth_rate(catalogs[1]['Z'])
        return catalogs[0]['INDWEIGHT'] * wtilde, catalogs[1]['INDWEIGHT'] * {0: w0, 2: w2}[ell]

    yield _get_weights(catalogs, tracers, ps)
    if tracers[1] != tracers[0]:
        yield _get_weights(catalogs[::-1], tracers[::-1], ps[::-1])[::-1]


def propose_fiducial(kind, tracer, zrange=None, analysis='full_shape'):
    """
    Propose fiducial measurement parameters for given tracer and statistic kind.

    Parameters
    ----------
    kind : str
        Statistic kind. Options are 'zranges', 'nran', 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', 'recon'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'LRG+ELG', 'QSO'.
    zrange : tuple, optional
        Redshift range. If provided, it will override the default zrange.

    Returns
    -------
    params : dict
        Dictionary of proposed fiducial parameters for the specified statistic kind and tracer.
    """
    base = {'catalog': {}, 'particle2_correlation': {}, 'mesh2_spectrum': {}, 'mesh3_spectrum': {}}
    propose_fiducial = {
        'BGS': {'nran': 3, 'recon': {'bias': 1.5, 'smoothing_radius': 15., 'zrange': (0.1, 0.4)}},
        'LRG+ELG': {'nran': 13, 'recon': {'bias': 1.6, 'smoothing_radius': 15.}, 'zrange': (0.8, 1.1)},
        'LRG': {'nran': 10, 'recon': {'bias': 2.0, 'smoothing_radius': 15., 'zrange': (0.4, 1.1)}},
        'ELG': {'nran': 15, 'recon': {'bias': 1.2, 'smoothing_radius': 15., 'zrange': (0.8, 1.6)}},
        'QSO': {'nran': 4, 'recon': {'bias': 2.1, 'smoothing_radius': 30., 'zrange': (0.8, 2.1)}}
    }
    tracers = _make_tuple(tracer)
    tracer = join_tracers(tracers)
    tracer = get_simple_tracer(tracer)
    propose_fiducial = base | propose_fiducial[tracer]
    if 'png' in analysis:
        propose_weight = 'default-oqe' # use OQE weights by default
        propose_zranges = {'BGS': [(0.1, 0.4)], 'LRG': [(0.4, 1.1)], 'ELG': [(0.8, 1.6)], 'LRG+ELG': [(0.8, 1.1)], 'QSO': [(0.8, 3.5)]}
        propose_FKP_P0 = {'LRG': 5e4, 'ELG': 2e4, 'QSO': 3e4}
        propose_meshsizes = {'BGS': 700, 'LRG': 700, 'ELG': 700, 'LRG+ELG': 700, 'QSO': 700}
        propose_cellsize = 20.
    else:
        propose_weight = 'default-FKP'
        propose_zranges = {'BGS': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
                           'ELG': [(0.8, 1.1), (1.1, 1.6)], 'LRG+ELG': [(0.8, 1.1)], 'QSO': [(0.8, 2.1)]}
        propose_FKP_P0 = {'BGS': 7e3, 'LRG': 1e4, 'ELG': 4e3, 'LRG+ELG': 1e4, 'QSO': 6e3}
        propose_meshsizes = {'BGS': 750, 'LRG': 750, 'ELG': 960, 'LRG+ELG': 750, 'QSO': 1152}
        propose_cellsize = 7.5
    propose_fiducial.update(zranges=propose_zranges[tracer])
    propose_fiducial['catalog'].update(weight=propose_weight, nran=propose_fiducial['nran'], zranges=propose_zranges[tracer], FKP_P0=propose_FKP_P0[tracer])
    for stat in ['mesh2_spectrum', 'mesh3_spectrum']:
        propose_fiducial[stat]['mattrs'] = {'meshsize': propose_meshsizes[tracer], 'cellsize': propose_cellsize}
    if 'png' in analysis:
        propose_fiducial['mesh2_spectrum'].update(ells=(0, 2), optimal_weights=functools.partial(compute_fiducial_png_weights, tracer=tracers))
    else:
        propose_fiducial['mesh2_spectrum'].update(ells=(0, 2, 4))
        propose_fiducial['mesh3_spectrum'].update(ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal', selection_weights={tracer: functools.partial(compute_fiducial_selection_weights, tracer=tracer) for tracer in tracers})
    if 'protected' in analysis:
        propose_fiducial['mesh2_spectrum'].update(ells=(0,))
        propose_fiducial['mesh3_spectrum'].update(ells=[(0, 0, 0)])
    for stat in ['recon']:
        recon_cellsize = propose_fiducial[stat]['smoothing_radius'] / 3.
        primes, divisors = (2, 3, 5), (2,)
        propose_fiducial[stat]['mattrs'] = {'boxpad': 1.2, 'cellsize': recon_cellsize, 'primes': primes, 'divisors': divisors}
    for name in list(propose_fiducial):
        propose_fiducial[f'recon_{name}'] = propose_fiducial[name]  # same for post-recon measurements
    return propose_fiducial[kind]


def _unzip_catalog_options(catalog):
    """From a catalog dictionary with nran, zrange, ..., tracer, return {tracer: {nran:..., zrange: ...}}"""
    if 'tracer' in catalog:
        tracers = _make_tuple(catalog['tracer'])
        toret = {}
        for itracer, tracer in enumerate(tracers):
            toret[tracer] = dict(catalog) | dict(tracer=tracer)
            for key, value in list(toret[tracer].items()):
                if key == 'zrange':
                    toret[tracer][key] = value[itracer] if isinstance(value, tuple) and np.ndim(value[0]) else value
                else:
                    toret[tracer][key] = value[itracer] if isinstance(value, tuple) else value
    else:
        toret = dict(catalog)
    return toret


def _zip_catalog_options(catalog, squeeze=True):
    """From {tracer: {nran:..., zrange: ...}}, return {tracer: tuple or single tracer if same, nran: tuple or single number if same}"""
    tracers = tuple(catalog.keys())
    toret = {key: [] for tracer in tracers for key in catalog[tracer]}
    num = {key: 0 for tracer in tracers for key in catalog[tracer]}
    for tracer in tracers:
        for key in toret:
            value = catalog[tracer].get(key, None)
            if value not in toret[key]:
                toret[key].append(value)
                num[key] += 1
    toret = {key: tuple(value) if num[key] > 1 or not squeeze else value[0] for key, value in toret.items()}
    toret['tracer'] = tracers
    return toret


def fill_fiducial_options(kwargs, analysis='full_shape'):
    """Fill missing options with fiducial values."""
    options = {key: dict(value) for key, value in kwargs.items()}
    mattrs = options.pop('mattrs', {})
    options['catalog'] = _unzip_catalog_options(options['catalog'])
    tracers = tuple(options['catalog'].keys())
    for tracer in tracers:
        fiducial_options = propose_fiducial('catalog', tracer=tracer, analysis=analysis)
        options['catalog'][tracer] = fiducial_options | options['catalog'][tracer]
    recon_options = options.pop('recon', {})
    # recon for each tracer
    options['recon'] = {}
    for tracer in tracers:
        fiducial_options = propose_fiducial('recon', tracer=tracer, analysis=analysis)
        options['recon'][tracer] = fiducial_options | recon_options.get(tracer, recon_options)
        if mattrs: options['recon'][tracer]['mattrs'] = mattrs
        options['recon'][tracer]['nran'] = options['recon'][tracer].get('nran', options['catalog'][tracer]['nran'])
        assert options['recon'][tracer]['nran'] >= options['catalog'][tracer]['nran'], 'must use more randoms for reconstruction than clustering measurements'
    for recon in ['', 'recon_']:
        for stat in ['particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum']:
            stat = f'{recon}{stat}'
            fiducial_options = propose_fiducial(stat, tracer=tracers, analysis=analysis)
            options[stat] = fiducial_options | options.get(stat, {})
            if 'mesh' in stat:
                if mattrs: options[stat]['mattrs'] = mattrs
        for stat in ['window_mesh2_spectrum', 'window_mesh3_spectrum']:
            spectrum_options = options[stat.replace('window_', '')]
            spectrum_options = {key: value for key, value in spectrum_options.items() if key in ['selection_weights', 'optimal_weights']}
            options[stat] = spectrum_options | options.get(stat, {})
    return options


def _merge_options(options1, options2):
    """Merge two options dictionaries, after call to :func:`fill_fiducial_options`. Nested dictionaries are updated."""
    options = {key: dict(value) for key, value in options1.items()}
    for key, value in options2.items():
        if key not in options: options[key] = {k: None for k in value}
        if key in ['catalog', 'recon']:  # tracer division
            for tracer in value:
                options[key].setdefault(tracer, {})
                options[key][tracer].update(value[tracer])
        else:
            options[key].update(value)

    return options


def get_catalog_fn(version=None, cat_dir=None, kind='data', tracer='LRG',
                   region='NGC', weight='default-FKP', nran=10, imock=0, ext='h5', **kwargs):
    """
    Return catalog filename(s) for given parameters.

    Parameters
    ----------
    version : str
        Catalog version. Options are 'data-dr1-v1.5', 'data-dr2-v2', 'holi-v1-complete', 'holi-v1-altmtl'.
    cat_dir : str, Path, optional
        Directory containing the catalogs. If None, pre-registered paths will be used based on version.
    kind : str
        Catalog kind. Options are 'data', 'randoms', 'full_data', 'full_randoms'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'LRG+ELG', 'QSO'.
    region : str
        Region name. Options are 'NGC', 'SGC', 'N', 'S', 'ALL', 'NGCnoN', 'SGCnoDES'.
    weight : str
        Weight type. Options are 'default-FKP', 'defaut-bitwise-FKP', etc.
    nran : int
        Number of random catalogs.
    imock : int
        Mock index (for mock catalogs). Default is 0.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : str, Path, list
        Catalog filename(s).
        Multiple filenames are returned as a list when region is 'ALL' or when kind is 'randoms' or 'full_randoms'.
    """
    if region in ['N', 'NGC', 'NGCnoN']: region = 'NGC'
    elif region in ['SGC', 'SGCnoDES']: region = 'SGC'
    elif 'full' not in kind:
        if region in ['S', 'ALL']: regions = ['NGC', 'SGC']
        else: raise NotImplementedError(f'{region} is unknown')
        fn_lists =  [get_catalog_fn(version=version, cat_dir=cat_dir, kind=kind, tracer=tracer,
                                    region=region, weight=weight, nran=nran, imock=imock, ext=ext, **kwargs) for region in regions]
        # flatten list of lists (can append with nrand > 1 and region='ALL')
        if any(isinstance(fn_list, list) for fn_list in fn_lists):
            return [fn for fn_list in fn_lists for fn in (fn_list if isinstance(fn_list, list) else [fn_list])]
        else:
            return fn_lists

    if cat_dir is None:  # pre-registered paths
        if version == 'data-dr1-v1.5':
            cat_dir = desi_dir / f'survey/catalogs/Y1/LSS/iron/LSScats'
            if 'bitwise' in weight:
                cat_dir = cat_dir / 'v1.5pip'
            else:
                cat_dir = cat_dir / 'v1.5'
            ext = 'fits'
        elif version == 'data-dr2-v2':
            cat_dir = desi_dir / f'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2'
            if kind == 'parent_randoms':
                program = 'bright' if 'BGS' in tracer else 'dark'
                return [cat_dir / f'{program}_{iran}_full_noveto.ran.{ext}' for iran in range(nran)]
            if 'bitwise' in weight:
                data_dir = cat_dir / 'PIP'
            else:
                data_dir = cat_dir / 'nonKP'
            if kind == 'data':
                return data_dir / f'{tracer}_{region}_clustering.dat.fits'
            if kind == 'randoms':
                return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
            if kind == 'full_data':
                return cat_dir / f'{tracer}_full_HPmapcut.dat.fits'
            if kind == 'full_randoms':
                return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran)]
        elif version == 'holi-v1-complete':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            if kind == 'data':
                return cat_dir / f'{tracer}_complete_{region}_clustering.dat.h5'
            if kind == 'randoms':
                return [cat_dir / f'{tracer}_complete_{region}_{iran:d}_clustering.ran.h5' for iran in range(nran)]
            ext = 'fits' if 'full' in kind else 'h5'
        elif version == 'holi-v1-altmtl':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            ext = 'fits' if 'full' in kind else 'h5'
        elif version == 'glam-uchuu-v1-complete':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/GLAM-Uchuu_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            if kind == 'data':
                return cat_dir / f'{tracer}_complete_{region}_clustering.dat.h5'
            if kind == 'randoms':
                return [cat_dir / f'{tracer}_complete_{region}_{iran:d}_clustering.ran.h5' for iran in range(nran)]
            ext = 'h5'
        elif version == 'glam-uchuu-v1-altmtl':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/GLAM-Uchuu_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            ext = 'h5'
        elif version == 'abacus-2ndgen-complete':
            if 'BGS' in tracer:
                cat_dir = desi_dir / f'survey/catalogs/Y3/mocks/SecondGenMocks/AbacusSummitBGS_v2/mock{imock:d}'
            else:
                cat_dir = desi_dir / f'survey/catalogs/Y3/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{imock:d}'
            if kind == 'data':
                return cat_dir / f'{tracer}_complete_clustering.dat.fits'
            if kind == 'randoms':
                return [cat_dir / f'{tracer}_complete_{iran:d}_clustering.ran.fits' for iran in range(nran)]
        elif version == 'abacus-2ndgen-altmtl':
            if 'BGS' in tracer:
                cat_dir = desi_dir / f'survey/catalogs/Y3/mocks/SecondGenMocks/AbacusSummitBGS_v2/altmtl{imock:d}/kibo-v1/mock{imock:d}/LSScats'
            else:
                cat_dir = desi_dir / f'survey/catalogs/Y3/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl{imock:d}/kibo-v1/mock{imock:d}/LSScats'
            ext = 'fits'
        elif 'uchuu-hf' in version:
            if 'altmtl' in version:
                # Do not exist anymore?
                cat_dir =  Path(desi_dir / f'mocks/cai/Uchuu-SHAM/Y3-v2.0/{imock:04d}/altmtl/')
            else:
                cat_dir =  Path(desi_dir / f'mocks/cai/Uchuu-SHAM/Y3-v2.0/{imock:04d}/complete/')
            if kind == 'data':
                return Path(cat_dir / f'Uchuu-SHAM_{get_simple_tracer(tracer)}_Y3-v2.0_0000_clustering.dat.fits')
            if kind == 'randoms':
                return [cat_dir / f'Uchuu-SHAM_{get_simple_tracer(tracer)}_Y3-v2.0_0000_{iran}_clustering.ran.fits' for iran in range(nran)]
    cat_dir = Path(cat_dir)
    if kind == 'data':
        return cat_dir / f'{tracer}_{region}_clustering.dat.{ext}'
    if kind == 'randoms':
        return [cat_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.{ext}' for iran in range(nran)]
    if kind == 'full_data':
        return cat_dir / f'{tracer}_full_HPmapcut.dat.{ext}'
    if kind == 'full_randoms':
        return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.{ext}' for iran in range(nran)]
    if kind == 'single_full_randoms':
        return cat_dir / f'{tracer}_{nran:d}_full_HPmapcut.ran.{ext}'
    if kind == 'single_randoms':
        return cat_dir / f'{tracer}_{region}_{nran:d}_clustering.ran.{ext}'


def get_stats_fn(stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', kind='mesh2_spectrum', auw=None, cut=None, extra='', ext='h5', **kwargs):
    """
    Return measurement filename for given parameters.

    Parameters
    ----------
    stats_dir : str, Path
        Directory containing the measurements.
    version : str, optional
        Measurement version.
    kind : str
        Measurement kind. Options are 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', etc.
    tracer : str
        Tracer name.
    region : str
        Region name.
    zrange : tuple, optional
        Redshift range.
    auw : bool, optional
        Whether to include angular upweighting.
    cut : bool, optional
        Whether to include theta cut.
    weight : str
        Weight type. Options are 'default_FKP', 'defaut_FKP_bitwise', etc.
    imock : int, str, optional
        Mock index (for mock catalogs). If '*', return all existing mock filenames.
    extra : str, optional
        Extra string to append to filename.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : str, Path, list
        Measurement filename(s).
        Multiple filenames are returned as a list when imock is '*'.
    """
    _default_options = dict(version=None, tracer=None, region=None, zrange=None, weight=None, imock=None)
    catalog_options = kwargs.get('catalog', {})
    if not catalog_options:
        catalog_options = {key: kwargs.get(key, _default_options[key]) for key, value in _default_options.items()}
        catalog_options = _unzip_catalog_options(catalog_options)
    else:
        catalog_options = _unzip_catalog_options(catalog_options)
        _default_options.pop('tracer')
        catalog_options = {tracer: _default_options | catalog_options[tracer] for tracer in catalog_options}
    catalog_options = _zip_catalog_options(catalog_options, squeeze=False)
    imock = catalog_options['imock']
    if imock[0] and imock[0] == '*':
        fns = [get_stats_fn(stats_dir=stats_dir, kind=kind, auw=auw, cut=cut, ext=ext, catalog=catalog_options | dict(imock=(imock,)), **kwargs) for imock in range(1000)]
        return [fn for fn in fns if os.path.exists(fn)]

    stats_dir = Path(stats_dir)

    def join_if_not_none(f, key):
        items = catalog_options[key]
        if any(item is not None for item in items):
            return join_tracers(tuple(f(item) for item in items if item is not None))
        return ''

    def check_is_not_none(key):
        items = catalog_options[key]
        assert all(item is not None for item in items), f'provide {key}'
        return items

    version = join_if_not_none(str, 'version')
    if version: stats_dir = stats_dir / version
    tracer = join_tracers(check_is_not_none('tracer'))
    zrange = join_if_not_none(lambda zrange: f'z{zrange[0]:.1f}-{zrange[1]:.1f}', 'zrange')
    zrange = f'_{zrange}' if zrange else ''
    region = join_tracers(check_is_not_none('region'))
    weight = join_tracers(check_is_not_none('weight'))
    auw = '_auw' if auw else ''
    cut = '_thetacut' if cut else ''
    extra = f'_{extra}' if extra else ''
    imock = join_if_not_none(str, 'imock')
    imock = f'_{imock}' if imock else ''
    corr_type = 'smu'
    battrs = kwargs.get('battrs', None)
    if battrs is not None: corr_type = ''.join(list(battrs))
    kind = {'mesh2_spectrum': 'mesh2_spectrum_poles',
            'particle2_correlation': f'particle2_correlation_{corr_type}'}.get(kind, kind)
    if 'mesh3' in kind:
        basis = kwargs.get('basis', None)
        basis = f'_{basis}' if basis else ''
        kind = f'mesh3_spectrum{basis}_poles'
    basename = f'{kind}_{tracer}{zrange}_{region}_weight-{weight}{auw}{cut}{extra}{imock}.{ext}'
    return stats_dir / basename


def get_box_stats_fn(stats_dir='/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacushf/measurements',
                     kind='mesh2_spectrum', extra='', ext='h5', **kwargs):
    """
    Return measurement filename for box mocks with given parameters.

    Parameters
    ----------
    stats_dir : str, Path
        Directory containing the measurements.
    version : str, optional
        Measurement version. Default is 'v2'.
    kind : str
        Measurement kind. Options are 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', etc.
    tracer : str
        Tracer name.
    cosmo : str
        Cosmology label (e.g., 'c000').
    zrange : tuple, optional
        Redshift range of interest. This will be mapped to a specific box snapshot.
    hod : str, optional
        HOD flavor (e.g., 'base_B', 'base_dv'). Default is 'base' (baseline HOD).
    los : str, optional
        Line of sight direction (e.g., 'z'). Default is 'z'.
    imock : int, str, optional
        Mock index. If '*', return all existing mock filenames.
    extra : str, optional
        Extra string to append to filename.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : str, Path, list
        Measurement filename(s).
        Multiple filenames are returned as a list when imock is '*'.
    """
    _default_options = dict(version='v2', tracer=None, cosmo=None, zrange=None, hod='base', los='z', imock=None)
    catalog_options = kwargs.get('catalog', {})
    if not catalog_options:
        catalog_options = {key: kwargs.get(key, _default_options[key]) for key, value in _default_options.items()}
        catalog_options = _unzip_catalog_options(catalog_options)
    else:
        catalog_options = _unzip_catalog_options(catalog_options)
        _default_options.pop('tracer')
        catalog_options = {tracer: _default_options | catalog_options[tracer] for tracer in catalog_options}
    catalog_options = _zip_catalog_options(catalog_options, squeeze=False)
    imock = catalog_options['imock']

    if imock[0] and imock[0] == '*':
        fns = [get_box_stats_fn(stats_dir=stats_dir, kind=kind, ext=ext, catalog=catalog_options | dict(imock=(imock,)), **kwargs) for imock in range(1000)]
        return [fn for fn in fns if os.path.exists(fn)]

    stats_dir = Path(stats_dir)

    def join_if_not_none(f, key):
        items = catalog_options[key]
        if any(item is not None for item in items):
            return join_tracers(tuple(f(item) for item in items if item is not None))
        return ''

    def check_is_not_none(key):
        items = catalog_options[key]
        assert all(item is not None for item in items), f'provide {key}'
        return items

    version = join_if_not_none(str, 'version')
    if version: stats_dir = stats_dir / version
    tracer = join_tracers(check_is_not_none('tracer'))
    cosmo = join_tracers(check_is_not_none('cosmo'))
    zrange = join_if_not_none(lambda zrange: f'z{zrange[0]:.1f}-{zrange[1]:.1f}', 'zrange')
    zrange = f'_{zrange}' if zrange else ''
    hod = join_tracers(check_is_not_none('hod'))
    hod = f'_{hod}' if hod else ''
    los = join_tracers(check_is_not_none('los'))
    extra = f'_{extra}' if extra else ''
    imock = join_if_not_none(str, 'imock')
    imock = f'_{imock}' if imock else ''
    corr_type = 'smu'
    battrs = kwargs.get('battrs', None)
    if battrs is not None: corr_type = ''.join(list(battrs))
    kind = {'mesh2_spectrum': 'mesh2_spectrum_poles',
            'particle2_correlation': f'particle2_correlation_{corr_type}'}.get(kind, kind)
    if 'mesh3' in kind:
        basis = kwargs.get('basis', None)
        basis = f'_{basis}' if basis else ''
        kind = f'mesh3_spectrum{basis}_poles'
    basename = f'{kind}_{tracer}{zrange}_{cosmo}{hod}_los{los}{extra}{imock}.{ext}'
    return stats_dir / basename


def checks_if_exists_and_readable(get_fn, test_if_readable=True, **kwargs):
    """
    Return lists of existing, missing and not readable files for all combinations of input kwargs.
    Input :func:`get_fn` must provide the filename associated to the input kwargs.
    """
    def is_unreadable(fn):
        fn = str(fn)
        if any(fn.endswith(ext) for ext in ['hdf', 'h4', 'hdf4', 'he2', 'h5', 'hdf5', 'he5', 'h5py']):
            try:
                import hdf5plugin
                import h5py
                with h5py.File(fn, 'r', locking=False):
                    pass
                return False
            except Exception as exc:
                return exc
        elif fn.endswith('fits') or fn.endswith('fit') or fn.endswith('fts'):
            try:
                import fitsio
                with fitsio.FITS(fn) as file:
                    for i, hdu in enumerate(file): pass
                return False
            except Exception as exc:
                return exc
        else:
            warnings.warn(f'cannot check readability of file {fn} with unknown extension')
            return False

    names, values = zip(*kwargs.items())
    exists, missing, unreadable = [([], {name: [] for name in names}) for i in range(3)]
    unreadable = unreadable + ([],)  # add exceptions

    def _append(toret, fn, kwargs, exc=None):
        toret[0].append(fn)
        for name in names: toret[1][name].append(kwargs[name])
        if exc is not None: toret[2].append(exc)

    for values in itertools.product(*values):
        fn_kwargs = dict(zip(names, values))
        fn = get_fn(**fn_kwargs)
        if os.path.exists(fn):
            _append(exists, fn, fn_kwargs)
            if test_if_readable:
                exc = is_unreadable(fn)
                if exc:
                    _append(unreadable, fn, fn_kwargs, exc=exc)
        else:
            _append(missing, fn, fn_kwargs)
    return exists, missing, unreadable


# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def _format_bitweights(bitweights):
    if bitweights is None:
        return []
    if isinstance(bitweights, (tuple, list)):
        return list(bitweights)
    if bitweights.ndim == 2:
        return list(bitweights.T)
    return [bitweights]


@default_mpicomm
def _read_catalog(fn, mpicomm=None, **kwargs):
    """Wrapper around :meth:`Catalog.read` to read catalog(s)."""
    import numpy as np
    from mockfactory import Catalog
    import warnings
    one_fn = fn[0] if isinstance(fn, (tuple, list)) else fn
    if str(one_fn).endswith('.h5'):
        kwargs.setdefault('locking', False)  # fix -> Unable to synchronously open file (unable to lock file, errno = 524, error message = 'Unknown error 524')
        try:
            catalog = Catalog.read(fn, group='LSS', mpicomm=mpicomm, **kwargs)
        except KeyError:
            catalog = Catalog.read(fn, mpicomm=mpicomm, **kwargs)
    else:
        catalog = Catalog.read(fn, mpicomm=mpicomm)
    if str(one_fn).endswith('.fits'): catalog.get(catalog.columns())  # Faster to read all columns at once
    if 'WEIGHT' not in catalog:
        warnings.warn('WEIGHT not in catalog')
        catalog['WEIGHT'] = catalog.ones()
    if 'TARGETID' not in catalog:
        warnings.warn('TARGETID not in catalog')
        catalog['TARGETID'] = catalog.cindex()
    return catalog


def _compute_missing_power(ntile, bitweights, loc_assigned, method='missing_power'):
    """
    Compute "missing power weights", called "NTMP" in Davide's paper below.

    Reference
    ---------
    https://arxiv.org/pdf/2411.12025v2, Section 5.2.2


    Parameters
    ----------
    ntile : array_like
        NTILE values.
    bitweights : array_like, list
        Bitweights array or list of bitweights arrays.
    loc_assigned : array_like
        LOCATION_ASSIGNED boolean array.
    method : str
        Method to compute missing power. Options are 'missing_power' or 'zero_prob'.

    Returns
    -------
    toret : array_like
        Missing power weights per NTILE.
    """
    bitweights = _format_bitweights(bitweights)
    # Input: list of bitweights
    nbits = 8 * sum(weight.dtype.itemsize for weight in bitweights)
    recurr = popcount(*bitweights)
    wiip = (nbits + 1) / (recurr + 1)
    zero_prob = (recurr == 0) & (~loc_assigned)

    #print(np.sum(zerop_msk))
    sum_ntile = np.bincount(ntile)
    sum_zero_prob = np.bincount(ntile, weights=zero_prob)
    sum_loc_assigned = np.bincount(ntile, weights=loc_assigned)
    sum_wiip = np.bincount(ntile, weights=loc_assigned * wiip)
    mask_zero_ntile = sum_ntile == 0
    frac_zero_prob = np.divide(sum_zero_prob, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    frac_missing_power = np.divide(sum_ntile - sum_wiip, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    if method == 'missing_power':
        toret = 1 - frac_missing_power
    elif method == 'zero_prob':
        toret = 1 - frac_zero_prob
    else:
        raise NotImplementedError(f'unknown method {method}')
    return toret


def _compute_binned_weight(ntile, weight):
    """Compute weights per ntile."""
    sum_ntile = np.bincount(ntile)
    sum_weight = np.bincount(ntile, weights=weight)
    mask_zero_ntile = sum_ntile == 0
    return np.divide(sum_weight, sum_ntile, out=np.ones_like(sum_weight), where=~mask_zero_ntile)


def get_binned_weight(catalog, binned_weight):
    """Get values of binned weights."""
    toret = 1.
    for column, weight in binned_weight.items():
        toret *= weight[catalog[column]]  # e.g. completeness[ntile]
    return toret


def get_positions_from_rdz(catalog):
    """Return Cartesian positions from R.A., Dec., and redshift."""
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    dist = fiducial.comoving_radial_distance(catalog['Z'])
    catalog['POSITION'] = sky_to_cartesian(dist, catalog['RA'], catalog['DEC'], dtype=dist.dtype)
    return catalog


def expand_randoms(randoms, parent_randoms, data, from_randoms=('RA', 'DEC'), from_data=('Z',)):
    """
    Expand randoms by adding columns from parent randoms and data catalogs via TARGETID matches.

    Parameters
    ----------
    randoms : Catalog
        Catalog of randoms, containing at least the columns ['TARGETID', 'TARGETID_DATA', 'WEIGHT', 'NX'].
    parent_randoms : Catalog
        Path to randoms that are a superset of 'randoms' to expand and contain at least ['TARGETID', 'RA', 'DEC'].
    data : Catalog
        Data catalogs to take redshift information from (concatenation of NGC and SGC).
    from_randoms : list, tuple
        List of the column names to add to ``randoms`` from the parent random catalog via TARGETID match.
        If empty, no columns are added from ``parent_randoms``.
    from_data : list, tuple
        List of the column names to add to ``randoms`` from the data catalog via TARGETID_DATA to TARGETID match.
        If empty, no columns are added from ``data``.

    Returns
    -------
    randoms : Catalog
        Expanded randoms catalog.
    """
    import numpy as np
    if len(from_randoms) != 0:
        _, randoms_index, parent_index = np.intersect1d(randoms['TARGETID'], parent_randoms['TARGETID'], return_indices=True)
        randoms = randoms[randoms_index]
        for column in from_randoms:
            if column != 'TARGETID':
                randoms[column] = parent_randoms[column][parent_index]
    if len(from_data) != 0:
        if isinstance(data, (list, tuple)):  # NGC + SGC
            data = Catalog.concatenate([dd[list(from_data) + ['TARGETID']] for dd in data])
        else:
            data = data[list(from_data) + ['TARGETID']]
        data['TARGETID_DATA'] = data.pop('TARGETID')

        if data['TARGETID_DATA'].max() < int(1e9):  # faster method
            lookup = np.arange(1 + data['TARGETID_DATA'].max())
            lookup[data['TARGETID_DATA']] = np.arange(len(data))
            index = lookup[randoms['TARGETID_DATA']]
        else:
            sorted_index = np.argsort(data['TARGETID_DATA'])
            index_in_sorted = np.searchsorted(data['TARGETID_DATA'], randoms['TARGETID_DATA'], sorter=sorted_index)
            index = sorted_index[index_in_sorted]
        for column in data:
            if column != 'TARGETID':
                randoms[column] = data[column][index]
    return randoms


@default_mpicomm
def read_clustering_catalog(kind=None, concatenate=True, get_catalog_fn=get_catalog_fn, get_positions_from_rdz=get_positions_from_rdz,
                            expand=None, reshuffle=None, FKP_P0=None, binned_weight=None, return_all_columns=False, mpicomm=None, **kwargs):
    """
    Read clustering catalog (data or randoms) with given parameters.

    Parameters
    ----------
    kind : str
        Catalog kind. Options are 'data' or 'randoms'.
    concatenate : bool
        Whether to concatenate catalogs from different regions or multiple randoms.
    get_catalog_fn : callable
        Function to get catalog filenames.
    get_positions_from_rdz : callable
        Function to compute Cartesian positions from R.A., Dec., and redshift.
    expand : callable, dict, optional
        If callable, function to expand randoms catalog.
        If dict, parameters to expand randoms catalog via :func:`expand_randoms`.
        In this case, modified in-place to add the parent randoms catalog with key its file name.
    reshuffle : callable, dict, optional
        If callable, function to reshuffle randoms catalog.
        If dict, parameters to reshuffle randoms catalog via :func:`reshuffle_randoms`.
        In this case, modified in-place to add the merged data catalogs with key its file name.
    binned_weight : dict, optional
        Binned weights to apply. Keys are column names, values are weight arrays.
    mpicomm : MPI.Comm, optional
        MPI communicator.
    kwargs : dict
        Additional keyword arguments to pass to :func:`get_catalog_fn`.

    Returns
    -------
    catalog : Catalog, list
        Catalog object or list of Catalog objects (if ``concatenate`` is False).
        Contains 'RA', 'DEC', 'Z', 'NX', 'TARGETID', 'POSITION', 'INDWEIGHT' (individual weight), 'BITWEIGHT' columns.
    """
    assert kind in ['data', 'randoms'], 'provide kind (data or randoms)'
    zrange, region, weight_type, imock, tracer = (kwargs.get(key) for key in ['zrange', 'region', 'weight', 'imock', 'tracer'])
    assert weight_type is not None, 'provide weight'
    reshuffle_condition = kind == 'randoms' and (isinstance(reshuffle, dict) or (reshuffle is not None))
    if reshuffle_condition:
        # if randoms are going to be reshuffled, all regions are needed so we force it.
        fns = get_catalog_fn(kind=kind,  **(kwargs | dict(region='ALL')))
    else:
        fns = get_catalog_fn(kind=kind, **kwargs)
    if not isinstance(fns, (tuple, list)): fns = [fns]
    if region in ['S', 'ALL'] or reshuffle_condition:
        # group in pairs (this assumes that fns is a list with the first half corresponds to filenames
        # of one region and the second half to another (e.g., NGC and SGC)
        fns = list(zip(fns[:len(fns)//2],fns[len(fns)//2:])) if len(fns)>1 else fns
    exists = {f: os.path.exists(f) for fn in fns for f in (fn if isinstance(fn, (list,tuple)) else [fn])}
    if not all(exists.values()):
        raise IOError(f'Catalogs {[fn for fn, ex in exists.items() if not ex]} do not exist!')

    if kind == 'randoms' and isinstance(expand, dict):
        from_data = expand.get('from_data', ['Z','WEIGHT_SYS','FRAC_TLOBS_TILES'])
        from_randoms = expand.get('from_randoms', ['RA','DEC','NTILE'])
        parent_randoms_fn = expand['parent_randoms_fn']
        if not isinstance(parent_randoms_fn, (tuple, list)):
            parent_randoms_fn = [parent_randoms_fn]
        if mpicomm.rank == 0:
            logger.info('Expanding randoms')
        parent_randoms = []
        for ifn, fn in enumerate(parent_randoms_fn):
            if fn not in expand:
                irank = ifn % mpicomm.size
                expand[fn] = _read_catalog(fn, mpicomm=MPI.COMM_SELF) if mpicomm.rank == irank else None
            parent_randoms.append(expand[fn])
        data_fn = expand.get('data_fn', None)
        if data_fn is None:
            data_fn = [get_catalog_fn(kind='data', **(kwargs | dict(region=region))) for region in ['NGC', 'SGC']]
        data = _read_catalog(data_fn, mpicomm=MPI.COMM_SELF)

        def expand(catalog, ifn):
            return expand_randoms(catalog, parent_randoms=parent_randoms[ifn], data=data, from_randoms=from_randoms, from_data=from_data)
    else:
        expand = None

    if kind == 'randoms' and isinstance(reshuffle, dict):
        merged_data_fn = reshuffle['merged_data_fn']
        if not isinstance(merged_data_fn, (tuple, list)):
            raise ValueError(f"Merged data filenames must be a tuple or list containing the filenames for NGC and SGC")
        if mpicomm.rank == 0:
            logger.info('Reshuffling randoms')
        merged_data = _read_catalog(merged_data_fn, mpicomm=MPI.COMM_SELF)
        data_fn = reshuffle.get('data_fn', None)
        if data_fn is None:
            data_fn = [get_catalog_fn(kind='data', **(kwargs | dict(region=region))) for region in ['NGC', 'SGC']]
        data = _read_catalog(data_fn, mpicomm=MPI.COMM_SELF)

        def reshuffle(catalog, seed):
            return reshuffle_randoms(tracer, catalog, merged_data=merged_data, data=data, seed=seed)
    else:
        reshuffle = None

    catalogs = [None] * len(fns)
    # if mpicomm.rank == 0: logger.info(f'length of filenames {len(fns)} and fns are {fns}')
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = _read_catalog(fn, mpicomm=MPI.COMM_SELF)
            if expand is not None:
                catalog = expand(catalog, ifn)
            if reshuffle is not None:
                if mpicomm.rank == 0:
                    from time import time
                    t0=time()
                    logger.info('Reshuffling randoms started.')
                catalog = reshuffle(catalog, 100 * imock + ifn)
                if mpicomm.rank == 0:
                    logger.info(f'Reshuffling randoms completed in {time() - t0:2.1f} s')
            columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_COMP', 'WEIGHT_FKP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'BITWEIGHTS', 'FRAC_TLOBS_TILES', 'NTILE', 'NX', 'TARGETID']
            columns = [column for column in columns if column in catalog.columns()]
            catalog = catalog[columns]

            if zrange is not None:
                catalog = catalog[(catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])]
            if 'bitwise' in weight_type:
                catalog = catalog[(catalog['FRAC_TLOBS_TILES'] != 0)]
            if region is not None:
                catalog = catalog[select_region(catalog['RA'], catalog['DEC'], region)]

            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT'].copy()
        bitwise_weights = None

        if 'bitwise' in weight_type:
            if kind == 'data':
                individual_weight = catalog['WEIGHT'] / catalog['WEIGHT_COMP']
                bitwise_weights = catalog['BITWEIGHTS']
            elif kind == 'randoms':
                individual_weight = catalog['WEIGHT'] * get_binned_weight(catalog, binned_weight['missing_power'])

        if 'FKP' in weight_type.upper():
            if mpicomm.rank == 0: logger.info('Multiplying individual weights by WEIGHT_FKP')
            if FKP_P0 is not None:
                catalog['WEIGHT_FKP'] = 1. / (1. + catalog['NX'] * FKP_P0)
            individual_weight *= catalog['WEIGHT_FKP']

        if 'noimsys' in weight_type:
            # this assumes that the WEIGHT column contains WEIGHT_SYS
            if mpicomm.rank == 0: logger.info('Dividing individual weights by WEIGHT_SYS')
            individual_weight /= catalog['WEIGHT_SYS']

        if 'comp' in weight_type:
            individual_weight *= get_binned_weight(catalog, binned_weight['completeness'])

        if not return_all_columns:
            catalog = catalog[[column for column in ['RA', 'DEC', 'Z', 'NX', 'TARGETID'] if column in catalog]]

        catalog['INDWEIGHT'] = individual_weight
        for column in catalog:
            if not np.issubdtype(catalog[column].dtype, np.integer):
                catalog[column] = catalog[column].astype('f8')
        if bitwise_weights is not None:
            catalog['BITWEIGHT'] = bitwise_weights
        catalog = get_positions_from_rdz(catalog)
        rdzw.append(catalog)
    if concatenate:
        if len(rdzw) > 1: return rdzw[0]
        return Catalog.concatenate(rdzw)
    else:
        return rdzw


@default_mpicomm
def read_full_catalog(kind, wntile=None, concatenate=True,
                     get_catalog_fn=get_catalog_fn, mpicomm=None, attrs_only=False, **kwargs):
    """
    Read full data or randoms catalog with given parameters.

    Parameters
    ----------
    kind : str
        Catalog kind. Options are 'parent_data', 'fibered_data', 'parent_randoms', 'fibered_randoms'.
    wntile : Path, str, default=None
        Filename of precomputed wntile weights. If None, compute from data clustering catalog (using :func:`get_catalog_fn`).
    concatenate : bool
        Whether to concatenate catalogs from different regions or multiple randoms.
    get_catalog_fn : callable
        Function to get catalog filenames.
    mpicomm : MPI.Comm, optional
        MPI communicator.
    kwargs : dict
        Additional keyword arguments to pass to :func:`get_catalog_fn`.

    Returns
    -------
    catalog : Catalog, list
        Catalog object or list of Catalog objects (if ``concatenate`` is False).
        Contains 'RA', 'DEC', 'TARGETID', 'INDWEIGHT' (individual weight), 'BITWEIGHT' columns.
    """
    assert kind in ['parent_data', 'fibered_data', 'parent_randoms', 'fibered_randoms'], 'provide kind'
    region, weight_type = (kwargs.get(key) for key in ['region', 'weight'])
    fns = get_catalog_fn(kind='full_data' if 'data' in kind else 'full_randoms', **kwargs)
    if not isinstance(fns, (tuple, list)): fns = [fns]

    exists = {os.path.exists(fn): fn for fn in fns}
    if not all(exists):
        raise IOError(f'Catalogs {[fn for ex, fn in exists.items() if not ex]} do not exist!')

    def get_wntile(wntile):
        if wntile is None:
            clustering_data_fn = get_catalog_fn(kind='data', **kwargs)
        else:
            clustering_data_fn = wntile
        toret = None
        if mpicomm.rank == 0:
            catalog = _read_catalog(clustering_data_fn, mpicomm=MPI.COMM_SELF)
            toret = _compute_binned_weight(catalog['NTILE'], catalog['WEIGHT'] / catalog['WEIGHT_COMP'])
        return mpicomm.bcast(toret, root=0)

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = _read_catalog(fn, mpicomm=MPI.COMM_SELF)
            columns = ['RA', 'DEC', 'LOCATION_ASSIGNED', 'BITWEIGHTS', 'NTILE', 'WEIGHT_NTILE', 'FRACZ_TILELOCID', 'FRAC_TLOBS_TILES']
            columns = [column for column in columns if column in catalog.columns()]
            catalog = catalog[columns]
            if 'BITWEIGHTS' in catalog:
                catalog.attrs['missing_power'] = {column: _compute_missing_power(catalog[column], catalog['BITWEIGHTS'], catalog['LOCATION_ASSIGNED']) for column in ['NTILE']}
            catalog.attrs['completeness'] = {column: _compute_binned_weight(catalog[column], catalog['FRACZ_TILELOCID'] * catalog['FRAC_TLOBS_TILES']) for column in ['NTILE']}
            if 'fibered' in kind:
                mask = catalog['LOCATION_ASSIGNED']
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    if attrs_only:
        for irank, catalog in catalogs:
            attrs = catalog.attrs if mpicomm.rank == irank else None
            if mpicomm.size > 1:
                attrs = mpicomm.bcast(catalog.attrs if mpicomm.rank == irank else None, root=irank)
            return attrs

    rdw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        if 'WEIGHT_NTILE' in catalog:
            individual_weight = catalog['WEIGHT_NTILE']
        else:
            individual_weight = get_binned_weight(catalog, {'NTILE': get_wntile(wntile)})
        bitwise_weights = None
        if 'fibered' in kind and 'data' in kind:
            if 'bitwise' in weight_type:
                individual_weight /= get_binned_weight(catalog, catalog.attrs['missing_power'])
                bitwise_weights = catalog['BITWEIGHTS']
            else:
                individual_weight /= (catalog['FRACZ_TILELOCID'] * catalog['FRAC_TLOBS_TILES'])
                bitwise_weights = None
        catalog = catalog[['RA', 'DEC']]
        catalog['INDWEIGHT'] = individual_weight
        for column in catalog:
            catalog[column] = catalog[column].astype('f8')
        if bitwise_weights is not None: catalog['BITWEIGHT'] = bitwise_weights
        rdw.append(catalog)
    if concatenate:
        if len(rdw) > 1: return rdw[0]
        return Catalog.concatenate(rdw)
    else:
        return rdw


def write_stats(fn, stats):
    """Write summary statistics to file from process 0 only."""
    import jax
    if jax.process_index() == 0:
        logger.info(f'Writing to {fn}')
        stats.write(fn)


def possible_combine_regions(regions):
    """Return potential combinations of regions."""
    regions = sorted(regions)
    region_combs = {'GCcomb': ['NGC', 'SGC'],
                    'NS': ['N', 'S'],
                    'GCcomb_noN': ['NGCnoN', 'SGC'],
                    'GCcomb_noDES': ['NGC', 'SGCnoDES']}
    combs = {}
    for _region_comb, _regions in region_combs.items():
        if all(region in _regions for region in regions):
            combs[_region_comb] = _regions
    return combs


def compute_fkp_effective_redshift(*fkps, cellsize=10., order=2, split=None, fields=None, func_of_z=lambda x: x,
                                   resampler='cic', return_fraction=False):
    """
    Return effective redshift given input :class:`FKPField` of :class:`ParticleField` fields.

    Parameters
    ----------
    cellsize : float, default=10.
        Cellsize to use for mesh assignment in Mpc/h.
    order : int, default=2
        Weight redshift by the mesh density to that power (2 for 2PCF, 3 for 3PCF).
    split : int, tuple, optional
        Random seed to split particles to obtain ``order`` samples.
    func_of_z : callable, default=lambda x: x
        Optionally, function of redshift. E.g. growth rate.
    resampler : str, default='cic'
        Resampler to use for mesh assignment.
    fields : tuple, default=None
        Field identifiers; pass e.g. [0, 0] if two fields sharing the same positions are given as input;
        disjoint random subsamples will be selected.

    Returns
    -------
    zeff : float
        Effective redshift.
    """
    # FIXME
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from jaxpower import split_particles, FKPField
    from jaxpower.mesh import _iter_meshes

    fiducial = DESI()
    zstep = 0.005
    zgrid = np.arange(0., 1100 + zstep, zstep)
    rgrid = fiducial.comoving_radial_distance(zgrid)
    d2z = get_interpolator_1d(rgrid, func_of_z(zgrid), order=1)

    fkps_none =  list(fkps) + [None] * (order - len(fkps))

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    randoms = [get_randoms(fkp) for fkp in fkps_none]

    def compute_fkp_normalization_z(*particles, cellsize=cellsize, split=split, fields=fields):
        if split is not None:
            particles = split_particles(*particles, seed=split, fields=fields)
        reduce = 1
        for mesh in _iter_meshes(*particles, resampler=resampler, cellsize=cellsize, compensate=False, interlacing=0):
            reduce *= mesh
        rsum = reduce.sum()
        if not return_fraction: reduce /= rsum
        distance = jnp.sqrt(sum(xx**2 for xx in mesh.attrs.xcoords(kind='position', sparse=True)))
        reduce *= d2z(distance)
        if not return_fraction: return reduce.sum()
        return reduce.sum(), rsum

    return compute_fkp_normalization_z(*randoms)


def combine_stats(observables):
    """Combine input observables (e.g. NGC and SGC); of :mod:`lsstypes` type."""
    import copy
    observables = list(observables)
    observable = types.sum(observables)

    def _combine_attrs(observables):
        attrs = copy.deepcopy(observables[0].attrs)
        for name in ['size_data', 'size_randoms', 'size_shifted',
                     'wsum_data', 'wsum_randoms', 'wsum_shifted']:
            if name in attrs:
                # hdf5 converts to arrays, retransform to list
                attrs[name] = sum([list(observable.attrs[name]) for observable in observables], start=[])
        name = 'zeff'
        if name in attrs:
            norm_zeff = [observable.attrs[f'norm_{name}'] for observable in observables]
            attrs[name] = np.average([observable.attrs[name] for observable in observables], weights=norm_zeff)
            attrs[f'norm_{name}'] = np.sum(norm_zeff)
        return attrs

    def combine_attrs(observable, observables):
        return types.tree_map(lambda observables: observables[0].clone(attrs=_combine_attrs(observables[1:])), [observable] + observables, level=None)

    if isinstance(observable, types.WindowMatrix):
        window = observable
        observable = window = window.clone(observable=combine_attrs(window.observable, [window.observable for window in observables]))
    else:
        observable = combine_attrs(observable, [observable for observable in observables])

    return observable


def merge_catalogs(output, inputs, factor=1., seed=42, read_catalog=_read_catalog, **kwargs):
    import numpy as np
    from mockfactory import Catalog
    inputs = list(inputs)
    ncatalogs = len(inputs)
    catalogs = []
    columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'MASK']
    columns += ['WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'NX']
    rng = np.random.RandomState(seed=seed)
    from pyrecon.utils import MemoryMonitor
    with MemoryMonitor() as mem:
        for fn in inputs:
            catalog = read_catalog(fn, **kwargs)
            mask = rng.uniform(0., 1., catalog.size) < factor / ncatalogs
            catalog.get(catalog.columns())
            columns = [col for col in columns if col in catalog.columns()]
            catalogs.append(catalog[columns][mask])
            mem()
    catalog = Catalog.concatenate(catalogs, intersection=True)
    catalog.write(output)


def merge_randoms_catalogs(output, inputs, parent_randoms_fn=None, factor=1., seed=42,
                           read_catalog=_read_catalog, expand_randoms=expand_randoms, **kwargs):
    import numpy as np
    from mockfactory import Catalog

    if parent_randoms_fn is not None:
        parent_randoms = read_catalog(parent_randoms_fn)
        def expand(catalog):
            catalog = expand_randoms(catalog, parent_randoms=parent_randoms, data=None, from_randoms=('RA','DEC'), from_data=())
            if catalog.csize == 0:
                raise ValueError(f'Catalog size after expansion is {catalog.csize}')
            return catalog
    else:
        expand = None

    inputs = list(inputs)
    ncatalogs = len(inputs)
    rng = np.random.RandomState(seed=seed)
    from pyrecon.utils import MemoryMonitor
    concatenate = None
    # columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'MASK']
    columns = ['RA', 'DEC', 'NX', 'TARGETID', 'TARGETID_DATA', 'WEIGHT']

    # def get_uid(ra, dec):
    #     factor = 1000000
    #     return np.rint(ra * factor) + 360 * factor * np.rint((dec + 90.) * factor)

    def get_uid(ra, dec):
        return ra + 1j * dec

    with MemoryMonitor() as mem:
        for ifn, fn in enumerate(inputs):
            print(ifn,fn)
            catalog = read_catalog(fn, **kwargs)
            catalog.get(catalog.columns())
            if expand is not None:
                print(f'Expanding {fn}')
                catalog = expand(catalog)
            columns = [col for col in columns if col in catalog.columns()]
            if concatenate is None:
                mask = rng.uniform(0., 1., catalog.size) < factor / ncatalogs
                concatenate = catalog[columns][mask]
            else:
                csize = catalog.size
                mask = np.isin(get_uid(catalog['RA'], catalog['DEC']), get_uid(concatenate['RA'], concatenate['DEC']))
                print(mask.sum(), mask.sum() / mask.size, factor / ncatalogs, ncatalogs)
                catalog = catalog[~mask]
                if not catalog.csize: break
                print(factor * csize / catalog.size / ncatalogs)
                mask = rng.uniform(0., 1., catalog.size) < factor * csize / catalog.size / ncatalogs
                concatenate = Catalog.concatenate(concatenate, catalog[columns][mask])
            mem()
    concatenate.write(output)


def reshuffle_randoms(tracer, randoms, merged_data, data, seed):
    # get weights
    data_wtotp = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']
    data_wcomp = data_wtotp / data['WEIGHT']

    merged_data_wtotp = merged_data['WEIGHT_COMP'] * merged_data['WEIGHT_SYS'] * merged_data['WEIGHT_ZFAIL']
    merged_data_wcomp = merged_data_wtotp / merged_data['WEIGHT']

    merged_data_ftile = data_ftile = 1. # ok for FFA

    merged_data_ftile_wcomp = merged_data_ftile / merged_data_wcomp
    merged_data_nz = merged_data['NX'] / merged_data_ftile_wcomp

    P0 = np.rint(np.mean((1. / merged_data['WEIGHT_FKP'] - 1.) / merged_data['NX']))

    if 'Z' not in randoms:
        randoms['Z'] = randoms.zeros() # place holder, since will be filled with 'Z' from merged_data anyway
    randoms_ntile = randoms['NTILE']
    randoms_wcomp = _compute_binned_weight(data['NTILE'], data_wcomp)[randoms_ntile]

    # print(seed)
    rng = np.random.RandomState(seed=seed)

    regions = ['N', 'S']
    if tracer == 'QSO':
        regions = ['N', 'SnoDES', 'DES']

    randoms['NZ'] = randoms.zeros()

    randoms_ftile = 1.  # ok for FFA
    sum_data_weights, sum_randoms_weights = [], []

    for region in regions:
        mask_data = select_region(data['RA'], data['DEC'], region=region)
        mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
        mask_merged_data = select_region(merged_data['RA'], merged_data['DEC'], region=region)

        # Shuffle z
        index = rng.choice(mask_merged_data.sum(), size=mask_randoms.sum())
        randoms['Z'][mask_randoms] = merged_data['Z'][mask_merged_data][index]
        randoms['WEIGHT'][mask_randoms] = (merged_data_wtotp * randoms_ftile)[mask_merged_data][index]
        randoms['NZ'][mask_randoms] = merged_data_nz[mask_merged_data][index]

        sum_data_weights.append(data_wtotp[mask_data].sum())
        sum_randoms_weights.append(randoms['WEIGHT'][mask_randoms].sum())

    if np.any(randoms['Z'] == 0.):
        raise ValueError('Something went wrong! Place holder of z = 0. remain after reshuffling.')
    # Renormalize randoms / data here
    sum_data_weights, sum_randoms_weights = np.array(sum_data_weights), np.array(sum_randoms_weights)
    alphas = sum_data_weights / sum_randoms_weights / (sum(sum_data_weights) / sum(sum_randoms_weights))
    # logger.info('alpha before renormalization: {}'.format(alphas))

    for region, alpha in zip(regions, alphas):
        mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
        randoms['WEIGHT'][mask_randoms] *= alpha

    randoms['WEIGHT'] /= randoms_wcomp
    randoms['NX'] = randoms['NZ'] * (randoms_ftile / randoms_wcomp)
    # randoms['WEIGHT_FKP'] = 1. / (1. + randoms['NX'] * P0)
    del randoms['NZ']

    alphas = []
    sum_data_weights, sum_randoms_weights = [], []
    for region in regions:
        mask_data = select_region(data['RA'], data['DEC'], region=region)
        mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
        sum_data_weights.append(data['WEIGHT'][mask_data].sum())
        sum_randoms_weights.append(randoms['WEIGHT'][mask_randoms].sum())

    sum_data_weights, sum_randoms_weights = np.array(sum_data_weights), np.array(sum_randoms_weights)
    alphas = sum_data_weights / sum_randoms_weights / (sum(sum_data_weights) / sum(sum_randoms_weights))
    # logger.info('alpha after renormalization & reweighting: {}'.format(alphas))

    # for iregion, region in enumerate(cregions):
    #     randoms[select_region(randoms['RA'], randoms['DEC'], region=region)][output_columns].write(output_randoms_fn[iregion])
    return randoms
