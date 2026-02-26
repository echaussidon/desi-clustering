import logging

import numpy as np
import jax

import lsstypes as types
from .tools import _format_bitweights


logger = logging.getLogger('correlation2')


def compute_angular_upweights(*get_data):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'fibered_data', 'parent_data' catalogs. Each catalog must contain 'RA', 'DEC', 'INDWEIGHT', and optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree with 'DD' leaf.
    """
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    from lsstypes import ObservableLeaf, ObservableTree

    with jax.make_mesh((jax.device_count(),), axis_names=('x',), axis_types=(jax.sharding.AxisType.Auto,)):
        all_fibered_data, all_parent_data = [], []

        def get_rdw(catalog):
            positions = (catalog['RA'], catalog['DEC'])
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        for _get_data in get_data:
            _data = _get_data()
            fibered_data = Particles(*get_rdw(_data['fibered_data']), positions_type='rd', exchange=True)
            parent_data = Particles(*get_rdw(_data['parent_data']), positions_type='rd', exchange=True)
            all_fibered_data.append(fibered_data)
            all_parent_data.append(parent_data)

        theta = 10**np.arange(-5, -1 + 0.1, 0.1)  # TODO: update
        battrs = BinAttrs(theta=theta)
        bitwise = None
        if all_fibered_data[0].get('bitwise_weight'):
            bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
        wattrs = WeightAttrs(bitwise=bitwise)

        def get_counts(*particles):
            #setup_logging('error')
            autocorr = len(particles) == 1
            weight = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs)['weight']
            if autocorr:
                norm = wattrs(particles[0]).sum()**2 - wattrs(*(particles * 2)).sum()
            else:
                norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
            # No need to remove auto-pairs, as edges[0] > 0
            return weight / norm
            #return Count2(counts=weight, norm=norm, theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])

        DDfibered = get_counts(*all_fibered_data)
        wattrs = WeightAttrs()
        DDparent = get_counts(*all_parent_data)

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    return auw


def compute_particle2_correlation(*get_data_randoms, auw=None, cut=None, battrs: dict=None):
    """
    Compute two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        Each catalog must contain 'POSITION', 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights.
        Randoms and shifted catalogs can be lists of catalogs (for multiple randoms/shifted).
    auw : ObservableTree, optional
        Angular upweights to apply. If None, no angular upweights are applied.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) in degress.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs. If None, default bins are used. See cucount.jax.BinAttrs.

    Returns
    -------
    correlation : Count2Correlation
        Two-point correlation function as a Count2Correlation object.
    """
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, count2, setup_logging
    from lsstypes import Count2, Count2Correlation

    with jax.make_mesh((jax.device_count(),), axis_names=('x',), axis_types=(jax.sharding.AxisType.Auto,)):
        all_data, all_randoms, all_shifted = [], [], []

        def get_pw(catalog):
            positions = catalog['POSITION']
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        def get_all_particles(catalog):
            if not isinstance(catalog, (tuple, list)):
                catalog = [catalog]  # list of randoms
            return [Particles(*get_pw(catalog), exchange=True) for catalog in catalog]

        for _get_data_randoms in get_data_randoms:
            # data, randoms (optionally shifted) are tuples (positions, weights)
            _catalogs = _get_data_randoms()
            data = get_all_particles(_catalogs['data'])[0]  # data is not a list of catalogs
            randoms = get_all_particles(_catalogs['randoms'])
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'])
            else:
                shifted = [None] * len(randoms)
            all_data.append(data)
            all_randoms.append(randoms)
            all_shifted.append(shifted)
        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)
        sattrs = None
        if cut is not None:
            sattrs = SelectionAttrs(theta=(0., 0.05))
            if jax.process_index() == 0:
                logger.info(f'Applying theta-cut {sattrs}.')
        bitwise = angular = None
        if data.get('bitwise_weight'):
            bitwise = dict(weights=data.get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
        if auw is not None:
            angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
            if jax.process_index() == 0:
                logger.info(f'Applying AUW {angular}.')
        wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
        mattrs = None  # automatic setting for mesh

        # Helper to convert to lsstypes Count2
        def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray, attrs: dict) -> Count2:
            coords = battrs.coords()
            edges = battrs.edges()
            edges = {f'{k}_edges': v for k, v in edges.items()}
            return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

        # Hepler to get counts as Count2
        def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
            if wattrs is None: wattrs = WeightAttrs()
            autocorr = len(particles) == 1
            counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs, sattrs=sattrs)['weight']
            attrs = {'wsum': [wattrs(particles[0]).sum()]}
            if autocorr:
                auto_sum = wattrs(*(particles * 2)).sum()
                norm = wattrs(particles[0]).sum()**2 - auto_sum
                # Correct auto-pairs
                zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
                counts = counts.at[zero_index].add(-auto_sum)
            else:
                norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
                attrs['wsum'].append(wattrs(particles[1]).sum())
            return to_lsstypes(battrs, counts, norm, attrs=attrs)

        DD = get_counts(*all_data, wattrs=wattrs)
        data = data.clone(weights=wattrs(data))  # clone data, with IIP weights (in case we provided bitwise weights)

        DS, SD, SS, RR = [], [], [], []
        iran = 0
        for all_randoms, all_shifted in zip(zip(*all_randoms, strict=True), zip(*all_shifted, strict=True), strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            iran += 1
            RR.append(get_counts(*all_randoms))
            if all(shifted is not None for shifted in all_shifted):
                SS.append(get_counts(*all_shifted))
            else:
                all_shifted = all_randoms
                SS.append(RR[-1])
            DS.append(get_counts(all_data[0], all_shifted[-1]))
            SD.append(get_counts(all_shifted[0], all_data[-1]))

    DS, SD, SS, RR = (types.sum(XX) for XX in [DS, SD, SS, RR])
    correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    return correlation



def compute_box_particle2_correlation(*get_data, battrs: dict=None, mattrs: dict=None):
    """
    Compute two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        Each catalog must contain 'POSITION', 'INDWEIGHT'.
        Shifted catalogs can be lists of catalogs (for multiple randoms/shifted).
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs. If None, default bins are used. See cucount.jax.BinAttrs.
    mattrs : dict, array, optional
        Mesh attributes; typically a dictionary with 'boxsize' and 'boxcenter'.

    Returns
    -------
    correlation : Count2Correlation
        Two-point correlation function as a Count2Correlation object.
    """
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, count2, count2_analytic, setup_logging
    from lsstypes import Count2, Count2Correlation

    with jax.make_mesh((jax.device_count(),), axis_names=('x',), axis_types=(jax.sharding.AxisType.Auto,)):
        all_data, all_shifted = [], [], []

        def get_pw(catalog):
            positions = catalog['POSITION']
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        def get_all_particles(catalog):
            if not isinstance(catalog, (tuple, list)):
                catalog = [catalog]  # list of randoms
            return [Particles(*get_pw(catalog), exchange=True) for catalog in catalog]

        for _get_data in get_data:
            # data (optionally shifted) are tuples (positions, weights)
            _catalogs = _get_data()
            data = get_all_particles(_catalogs['data'])[0]  # data is not a list of catalogs
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'])
            else:
                shifted = [None]
            all_data.append(data)
            all_shifted.append(shifted)
        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)
        wattrs = WeightAttrs()
        mattrs = mattrs or {}
        mattrs = MeshAttrs(*all_data, battrs=battrs, **mattrs)

        # Helper to convert to lsstypes Count2
        def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray, attrs: dict) -> Count2:
            coords = battrs.coords()
            edges = battrs.edges()
            edges = {f'{k}_edges': v for k, v in edges.items()}
            return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

        # Hepler to get counts as Count2
        def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
            if wattrs is None: wattrs = WeightAttrs()
            autocorr = len(particles) == 1
            counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
            attrs = {'wsum': [wattrs(particles[0]).sum()]}
            if autocorr:
                auto_sum = wattrs(*(particles * 2)).sum()
                norm = wattrs(particles[0]).sum()**2 - auto_sum
                # Correct auto-pairs
                zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
                counts = counts.at[zero_index].add(-auto_sum)
            else:
                norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
                attrs['wsum'].append(wattrs(particles[1]).sum())
            return to_lsstypes(battrs, counts, norm, attrs=attrs)

        DD = get_counts(*all_data, wattrs=wattrs)
        data = data.clone(weights=wattrs(data))  # clone data, with IIP weights (in case we provided bitwise weights)
        RR = to_lsstypes(battrs, count2_analytic(battrs=battrs, mattrs=mattrs), norm=1., attrs={})

        DS, SD, SS = [], [], []
        iran = 0
        for all_shifted in zip(*all_shifted, strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            iran += 1
            if all(shifted is not None for shifted in all_shifted):
                SS.append(get_counts(*all_shifted))
                DS.append(get_counts(all_data[0], all_shifted[-1]))
                SD.append(get_counts(all_shifted[0], all_data[-1]))

    if iran:
        DS, SD, SS = (types.sum(XX) for XX in [DS, SD, SS])
        correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    else:
        correlation = Count2Correlation(estimator='natural', DD=DD, RR=RR)
    return correlation