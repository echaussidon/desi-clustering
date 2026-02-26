import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .tools import (default_mpicomm, Catalog, join_tracers, _unzip_catalog_options, _merge_catalog_options, _zip_catalog_options,
_read_catalog, get_simple_tracer, _merge_options, _make_tuple, desi_dir, write_stats, setup_logging)


def get_zrange_from_snap(tracer, zsnap=None, version='abacus-2ndgen'):
    """
    Return redshift range from snapshot for given tracer and version.
    If zsnap is None, return dict of zsnap: zrange for all snapshots of the tracer and version.
    """
    tracer = get_simple_tracer(tracer)
    zrange = {}
    if version in ['abacus-2ndgen', 'ezmock-dr1']:
        if tracer == 'BGS':
            zrange[0.200] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.800] = (0.6, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.325] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.400] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'abacus-hf-v1':
        if tracer == 'BGS':
            zrange[0.300] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.725] = (0.6, 0.8)
            zrange[0.950] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.475] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.400] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'abacus-hf-v2':
        if tracer == 'BGS':
            zrange[0.300] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.725] = (0.6, 0.8)
            zrange[0.950] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.475] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.550] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'uchuu-hf':
        if tracer == 'BGS':
            zrange[0.190] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.490] = (0.4, 0.6)
            zrange[0.700] = (0.6, 0.8)
            zrange[0.940] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.940] = (0.8, 1.1)
            zrange[1.430] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.430] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if zsnap is None:
        return zrange
    return zrange[zsnap]


def get_zsnap_from_z(tracer, z, version='abacus-2ndgen'):
    """Return zsnap from redshift (range)."""
    zranges = get_zrange_from_snap(tracer, zsnap=None, version=version)
    z = np.array(z)
    for zsnap, zrange in zranges.items():
        if np.all((z >= zrange[0] * (1 - 1e-6)) & (z <= zrange[1] * (1 + 1e-6))):
            return zsnap
    raise ValueError(f'input z not found in any snapshot {z}')


def propose_box_fiducial(kind, tracer, version='abacus-hf-v2'):
    """
    Propose fiducial measurement parameters for given tracer and statistic kind.

    Parameters
    ----------
    kind : str
        Statistic kind. Options are 'catalog', 'zsnaps', 'particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum', 'recon'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'QSO'.

    Returns
    -------
    params : dict
        Dictionary of proposed fiducial parameters for the specified statistic kind and tracer.
    """
    base = {'catalog': {}, 'particle2_correlation': {}, 'mesh2_spectrum': {}, 'mesh3_spectrum': {}}
    propose_fiducial = {
        'BGS': {'recon': {'bias': 1.5, 'smoothing_radius': 15.}},
        'LRG+ELG': {'recon': {'bias': 1.6, 'smoothing_radius': 15.}},
        'LRG': {'recon': {'bias': 2.0, 'smoothing_radius': 15.}},
        'ELG': {'recon': {'bias': 1.2, 'smoothing_radius': 15.}},
        'QSO': {'recon': {'bias': 2.1, 'smoothing_radius': 30.}}
    }
    tracers = _make_tuple(tracer)
    tracer = join_tracers(tracers)
    tracer = get_simple_tracer(tracer)
    propose_fiducial = base | propose_fiducial[tracer]
    propose_fiducial['recon']['nran'] = 10
    propose_fiducial['catalog'] = {'hod': '', 'los': 'z'}
    if 'abacus' in version:
        propose_fiducial['catalog'].update({'cosmo': '000'})
    if 'abacus-hf' in version:
        hod = 'base'
        if 'BGS' in tracer or 'LRG' in tracer:
            hod = 'base_B'
        elif 'ELG' in tracer:
            hod = 'base_conf_nfwexp'
        propose_fiducial['catalog'].update({'hod': hod})
    propose_fiducial['zsnaps'] = list(get_zrange_from_snap(tracer, zsnap=None, version=version))
    for stat in ['mesh2_spectrum', 'mesh3_spectrum']:
        propose_fiducial[stat]['mattrs'] = {'meshsize': 512}
    propose_fiducial['mesh2_spectrum'].update(ells=(0, 2, 4))
    propose_fiducial['mesh3_spectrum'].update(ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
    for stat in ['recon']:
        propose_fiducial[stat]['mattrs'] = {'cellsize': 5.}
    for name in list(propose_fiducial):
        propose_fiducial[f'recon_{name}'] = propose_fiducial[name]  # same for post-recon measurements
    return propose_fiducial[kind]


def fill_box_fiducial_options(kwargs):
    """Fill missing options with fiducial values."""
    options = {key: dict(value) for key, value in kwargs.items()}
    mattrs = options.pop('mattrs', {})
    options['catalog'] = _unzip_catalog_options(options['catalog'])
    tracers = tuple(options['catalog'].keys())
    for tracer in tracers:
        fiducial_options = propose_box_fiducial('catalog', tracer=tracer)
        options['catalog'][tracer] = fiducial_options | options['catalog'][tracer]
    los_options = dict(los=options['catalog'][tracers[0]]['los'])
    recon_options = options.pop('recon', {})
    # recon for each tracer
    options['recon'] = {}
    for tracer in tracers:
        fiducial_options = propose_box_fiducial('recon', tracer=tracer)
        options['recon'][tracer] = fiducial_options | los_options | recon_options.get(tracer, recon_options)
        if mattrs: options['recon'][tracer]['mattrs'] = mattrs
    for recon in ['', 'recon_']:
        for stat in ['particle2_correlation', 'mesh2_spectrum', 'mesh3_spectrum']:
            stat = f'{recon}{stat}'
            fiducial_options = propose_box_fiducial(stat, tracer=tracers)
            options[stat] = fiducial_options | los_options | options.get(stat, {})
            if 'mesh' in stat:
                if mattrs: options[stat]['mattrs'] = mattrs
        for stat in ['window_mesh2_spectrum']:
            options[stat] = options.get(stat, {})
        for stat in ['covariance_mesh2_spectrum']:
            spectrum_options = options[stat.replace('covariance_', '')]
            spectrum_options = {key: value for key, value in spectrum_options.items() if key in ['mattrs']}
            options[stat] = spectrum_options | options.get(stat, {})
    return options


def get_box_catalog_fn(version: str='abacus-hf-v2', cat_dir: str=None, kind='data', tracer: str=None, cosmo: str=None, zsnap: str=None,
                       hod: str='', los: str='z', imock: int=None, extra: str='', **kwargs):
    """
    Return catalog file name(s) for given parameters.

    Parameters
    ----------
    version : str
        Catalog version. Options are ['abacus-2ndgen', 'abacus-hf-v1', 'abacus-hf-v2'].
    cat_dir : str, Path, optional
        Directory containing the catalogs. If None, pre-registered paths will be used based on version.
    kind : str, optional
        Catalog kind. Options are 'data'.
    tracer : str
        Tracer name. Options are 'BGS', 'LRG', 'ELG', 'QSO'.
    cosmo : str
        Cosmology label (e.g., '000').
    zsnap : float
        Redshift of the box snapshot.
    hod : str, optional
        HOD flavor (e.g., 'base', 'base_B', 'base_dv').
    los : str, optional
        Line of sight direction ('x', 'y', 'z'). Default is 'z'.
    imock : int
        Mock index (for mock catalogs). Default is 0.
    extra : str, optional
        Extra string to append to filename.
    ext : str
        File extension. Default is 'h5'.

    Returns
    -------
    fn : Path
        Catalog file name(s).
    """
    if version == 'abacus-2ndgen':
        stracer = get_simple_tracer(tracer)
        cat_dir = desi_dir / f'cosmosim/SecondGenMocks/CubicBox/{stracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{imock:03d}/'
        return cat_dir / f'{stracer}_real_space.fits'
    if version == 'abacus-hf-v1':
        cat_dir = desi_dir / f'mocks/cai/abacus_HF/{"DR2_v2.0" if version == "v2" else "DR2_v1.0"}/AbacusSummit_base_c{cosmo}_ph{imock:03d}/Boxes'
        stracer = get_simple_tracer(tracer)
        sznap = f'{zsnap:.3f}'.replace('.', 'p')
        return cat_dir / f'abacus_HF_{stracer}_{sznap}_DR2_v1.0_AbacusSummit_base_c000_ph{imock:03d}_clustering.dat.fits'
    if version == 'abacus-hf-v2':
        stracer = {'ELG_LOP': 'ELG'}.get(tracer, tracer)
        sznap = f'{zsnap:.3f}'.replace('.', 'p')
        cat_dir = desi_dir / f'mocks/cai/abacus_HF/DR2_v2.0/AbacusSummit_base_c000_ph{imock:03d}/Boxes/{stracer}'
        return cat_dir / f'abacus_HF_{stracer}_{sznap}_DR2_v2.0_AbacusSummit_base_c000_ph{imock:03d}_{hod}_clustering.dat.h5'
    if version == 'uchuu-hf':
        stracer = {'ELG_LOP': 'ELG'}.get(tracer, tracer)
        sznap = (f'{zsnap:.1f}' if zsnap in [0.7, 1.9] else f'{zsnap:.2f}').replace('.', 'p')
        if 'LRG' in stracer: sznap = f'z{sznap}'
        cat_dir = desi_dir / f'mocks/cai/Uchuu-SHAM/Y3-v2.0/{imock:04d}/Boxes/{stracer}/'
        return cat_dir / f'Uchuu-{stracer}_box_{sznap}.h5'
    raise ValueError(f'version {version} not recognized')


@default_mpicomm
def read_clustering_box_catalog(kind='data', los='z', mpicomm=None, get_box_catalog_fn=get_box_catalog_fn, **kwargs):
    """
    Read data clustering catalog with given parameters.

    Parameters
    ----------
    kind : str
        Catalog kind. Options are 'data'.
    los : str
        Line of sight direction ('x', 'y', 'z') to apply RSD. Default is 'z'.
    mpicomm : MPI.Comm, optional
        MPI communicator.
    kwargs : dict
        Additional keyword arguments to pass to :func:`get_box_catalog_fn`.

    Returns
    -------
    catalog : Catalog
        Contains 'POSITION', 'INDWEIGHT' (individual weight) columns.
    """
    mpiroot = 0
    catalog = None
    boxsize, scalev = None, None
    zsnap = kwargs.get('zsnap', None)
    version = kwargs.get('version', '')
    tracer = get_simple_tracer(kwargs.get('tracer', ''))
    cosmo = kwargs.get('cosmo', '000')

    def read_catalog(fn):
        kwargs = {}
        if 'uchuu' in version:
            if 'LRG' in tracer: kwargs['group'] = 'galaxies'
            elif 'QSO' in tracer: kwargs['group'] = '/'
            else: kwargs['group'] = 'df'
        return _read_catalog(fn, mpicomm=MPI.COMM_SELF, **kwargs)

    def recenter(positions, boxsize):
        if 'uchuu-hf' in version:
            return positions - boxsize / 2.
        return positions

    if mpicomm.rank == mpiroot:
        fn = get_box_catalog_fn(kind=kind, los=los, **kwargs)
        catalog = read_catalog(fn)
        boxsize = catalog.header.get('BOXSIZE', 2000.)
        scalev = catalog.header.get('VELZ2KMS', None)

    if mpicomm.size > 1:
        catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=mpiroot)
    for name in catalog.columns():
        catalog[name.upper()] = catalog.pop(name)

    boxsize, scalev = mpicomm.bcast((boxsize, scalev), root=mpiroot)

    if scalev is None:
        from cosmoprimo.fiducial import AbacusSummit
        cosmo = AbacusSummit(cosmo)
        a = 1.0 / (1.0 + zsnap)
        E = cosmo.efunc(zsnap)
        scalev = 100.0 * a * E

    positions = np.column_stack([catalog[name] for name in ['X', 'Y', 'Z']])
    positions = recenter(positions, boxsize)

    vsmear = catalog.get('VSMEAR', catalog.zeros()) / scalev
    velocities = np.column_stack([catalog[name] for name in ['VX', 'VY', 'VZ']]) / scalev

    vlos = los
    if isinstance(los, str):
        vlos = [0.0] * 3
        vlos["xyz".index(los)] = 1.0
    vlos = np.array(vlos)

    positions = positions + (np.sum(velocities * vlos, axis=-1) + vsmear)[..., None] * vlos[None, :]
    positions = (positions + boxsize / 2.0) % boxsize - boxsize / 2.0
    attrs = {'boxsize': boxsize * np.ones(3), 'boxcenter': np.zeros(3), 'zsnap': zsnap, 'los': los}
    catalog = Catalog({'POSITION': positions, 'INDWEIGHT': np.ones_like(positions[..., 0])},
                      attrs=attrs, mpicomm=mpicomm)
    return catalog


def get_box_stats_fn(stats_dir=Path(os.getenv('SCRATCH', '.')) / 'measurements',
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
        Cosmology label (e.g., '000').
    zsnap : float
        Redshift of the box snapshot.
    hod : str, optional
        HOD flavor (e.g., 'base_B', 'base_dv'). Default is 'base' (baseline HOD).
    los : str, optional
        Line of sight direction ('x', 'y', 'z'). Default is 'z'.
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
    _default_options = dict(version=None, tracer=None, cosmo=None, hod=None, zsnap=None, los='z', imock=None)
    catalog_options = kwargs.pop('catalog', {})
    if not catalog_options:
        kwargs_options = {key: kwargs[key] for key, value in _default_options.items()}
        catalog_options = _unzip_catalog_options(kwargs_options)
    else:
        catalog_options = _merge_catalog_options(catalog_options, {key: kwargs.pop(key) for key in list(kwargs) if key in _default_options}, zipped=[None, True])
        for tracer in catalog_options:
            catalog_options[tracer].setdefault('imock', None)
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
    cosmo = join_if_not_none(str, 'cosmo')
    cosmo = f'_c{cosmo}' if cosmo else ''
    zsnap = join_tracers([f'{z:.3f}' for z in check_is_not_none('zsnap')])
    hod = join_if_not_none(str, 'hod')
    hod = f'_hod-{hod}' if hod else ''
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
    basename = f'{kind}_{tracer}_z{zsnap}{cosmo}{hod}_los-{los}{extra}{imock}.{ext}'
    return stats_dir / basename
