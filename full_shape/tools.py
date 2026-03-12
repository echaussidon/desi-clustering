"""
In this file we propose a dictionary-like interface to building :mod:`desilike` likelihoods.

Important functions are:
- :func:`generate_likelihood_options_helper`, helper to generate dictionary of options
- :func:`get_stats`: read clustering statistics from disk
- :func:`get_theory`: build desilike theory
- :func:`get_single_likelihood` return single desilike likelihood (that can be summed up with others)
- :func:`get_fits_fn`: proposed path to output fits.

Dictionary of options are organized as follows:
- list of dictionaries, one for each independent (summed) likelihood;
- each of this dictionary is {'observables': [observable1, observable2, ...], 'covariance': covariance options}
- each of observable1, observable2, ... is a dictionary that specifies how to build the desilike
observable (data, theory, window): ``{'stat': {'kind': ..., 'basis': ..., 'select': {...}},
'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}``.
"""


import os
import logging
from pathlib import Path

import numpy as np
import scipy as sp
import lsstypes as types
from lsstypes.utils import get_hartlap2007_factor, get_percival2014_factor

from clustering_statistics.tools import (float2str, get_full_tracer, get_simple_tracer, _make_tuple,
                                         get_simple_stats, _unzip_catalog_options, setup_logging)
from clustering_statistics import tools as clustering_tools


logger = logging.getLogger('tools')


def get_cosmology():
    """
    Construct and return a :mod:`desilike` :class:`Cosmoprimo` calculator and the DESI fiducial.

    Returns
    -------
    (cosmo, fiducial)
        - cosmo: :class:`desilike.theories.Cosmoprimo` instance with configured priors.
        - fiducial: :class:`cosmoprimo.fiducial.DESI` instance used as fiducial cosmology.
    """
    from desilike.theories import Cosmoprimo
    from cosmoprimo.fiducial import DESI

    fiducial = DESI()
    cosmo = Cosmoprimo(engine='class', fiducial=fiducial)
    # Free parameters h, omega_cdm, omega_b, logA with uniform priors
    # n_s and tau_reio are fixed
    # A Gaussian prior on omega_b.
    params = {
        'H0':       {'derived': True},
        'Omega_m':  {'derived': True},
        'sigma8_m': {'derived': True},
        'tau_reio': {'fixed': True},
        'n_s':      {'fixed': True},
        'omega_b':  {'fixed': False, 'prior': {'dist': 'norm',    'loc': 0.02237,  'scale': 0.00037}},
        'h':        {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.5,  0.9]}},
        'omega_cdm':{'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.05, 0.2]}},
        'logA':     {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [2.0,  4.0]}},
    }
    for name, config in params.items():
        if name in cosmo.init.params:
            cosmo.init.params[name].update(**config)
        else:
            cosmo.init.params[name] = config
    return cosmo, fiducial



def _get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=True, sigma8_fid=1.):
    """
    Build a dictionary of parameter priors.

    Parameters
    ----------
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2_spectrum', 'mesh2_spectrum'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float, optional
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        :meth:`Parameter.update` (e.g. ``{'fixed': True}`` or ``{'prior': {...}}``).
    """
    params = {}
    scale_eft = 12.5
    scale_sn0 = 2.0
    scale_sn2 = 5.0

    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7. * sigma8_fid**2, 'scale': 5}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3p'] = {'fixed': True}
            else:
                params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23. / 42. * sigma8_fid**4, 'scale': sigma8_fid**4},
                                 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_pp'] = {'fixed': True}
            else:
                params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}

    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3'] = {'fixed': True}
            else:
                params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}, 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_p'] = {'fixed': True}
            else:
                params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            shotnoise = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    return params


def get_theory(stat: str, theory: dict, z: float, cosmo, fiducial):
    """
    Return a configured theory desilike calculator for the requested statistic.

    Parameters
    ----------
    stat : str
        Statistic name, e.g. 'mesh2_spectrum' or 'mesh3_spectrum'.
    theory : dict
        Theory options dict containing at least 'model' and possibly other keys.
    z : float
        Effective redshift.
    cosmo, fiducial : objects
        Cosmology calculator and fiducial cosmology to pass to templates.

    Returns
    -------
    theory : BaseCalculator
        Initialized theory object from desilike for the requested statistic.
    """
    from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles)
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)
    theory_options = theory
    theory = None
    if 'mesh2_spectrum' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis', 'b3_coev']}
            theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid)
            for name, config in params.items():
                theory.init.params[name].update(**config)
            if theory_options['marg']:
                for param in theory.init.params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
    elif 'mesh3_spectrum' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis']}
            theory = FOLPSv2TracerBispectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], sigma8_fid=sigma8_fid)
            for name, config in params.items():
                theory.init.params[name].update(**config)
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory


def pack_stats(stats, **labels):
    """
    Pack a list of stat-like objects into a single :class:`types.ObservableTree` or :class:`types.WindowMatrix`.

    Parameters
    ----------
    stats : list[ObservableLike, WindowMatrix]
        List of statistics objects to pack.
    labels : mapping
        Labels to attach to the resulting container.
        E.g. ``observables=['spectrum2', 'spectrum3'], tracers=[('LRG', 'LRG'), ('LRG', 'LRG', 'LRG')]``
        for the combined power spectrum and bispectrum.

    Returns
    -------
    Packed types.ObservableTree or types.WindowMatrix
    """
    if isinstance(stats[0], types.ObservableLike):
        return types.ObservableTree(stats, **labels)
    elif isinstance(stats[0], types.WindowMatrix):
        windows = stats
        values = [window.value() for window in windows]
        observables = [window.observable for window in windows]
        theories = [window.theory for window in windows]
        return types.WindowMatrix(
            value=sp.linalg.block_diag(*values),
            observable=pack_stats(observables, **labels),
            theory=pack_stats(theories, **labels),
        )
    else:
        raise ValueError(f'unrecognized type {stats[0]}')


def unpack_stats(stats):
    """
    Unpack packed stats structures into individual windows/observables.

    Parameters
    ----------
    stats : types.ObservableLike | types.WindowMatrix | types.GaussianLikelihood
        If ObservableLike, returns a list of observables
        If WindowMatrix, returns a list of window matrices
        If GaussianLikelihood, returns a tuple[list of observables, list of window matrices, covariance]

    Returns
    -------
    Unpacked statistics.
    """
    if isinstance(stats, types.ObservableLike):
        return stats.flatten(level=1)  # iter over labels
    elif isinstance(stats, types.WindowMatrix):
        window = stats
        windows = []
        for label in window.observable.labels(level=1):
            windows.append(window.at.observable.get(**label).at.theory.get(**label))
        return windows
    elif isinstance(stats, types.GaussianLikelihood):
        likelihood = stats
        return (unpack_stats(likelihood.observable), unpack_stats(likelihood.window), likelihood.covariance)


def _infer_effective_nparams(observables: list[dict]) -> int:
    """Infer effective free-parameter count for covariance corrections.

    Uses a fixed effective count by fit content:
      - 7 for single-stat fits (mesh2-only or mesh3-only)
      - 9 for joint mesh2+mesh3 fits
    """
    stats = {obs['stat']['kind'] for obs in observables}
    has_mesh2 = any('mesh2_spectrum' in stat for stat in stats)
    has_mesh3 = any('mesh3_spectrum' in stat for stat in stats)
    return 9 if (has_mesh2 and has_mesh3) else 7


def _get_covariance_correction_factor(covariance: types.CovarianceMatrix,
                                      observables: list[dict],
                                      covariance_options: dict,
                                      default_corrections=('hartlap', 'percival')):
    """Return multiplicative covariance correction factor and per-term metadata."""
    corrections = covariance_options.get('corrections', default_corrections)
    if isinstance(corrections, str):
        corrections = [corrections]
    corrections = [str(corr).lower() for corr in (corrections or [])]

    nobs = int(covariance.attrs.get('nobs', 0))
    nbins = int(covariance.value().shape[0])
    factor = 1.
    metadata = {
        'nobs': nobs,
        'nbins': nbins,
        'corrections': tuple(corrections),
    }

    if 'hartlap' in corrections:
        hartlap = get_hartlap2007_factor(nobs, nbins)
        factor /= hartlap
        metadata['hartlap_factor'] = float(hartlap)

    if 'percival' in corrections:
        nparams = covariance_options.get('nparams', None)
        if nparams is None:
            nparams = _infer_effective_nparams(observables)
        percival = get_percival2014_factor(nobs, nbins, nparams)
        factor *= percival
        metadata['percival_factor'] = float(percival)
        metadata['nparams'] = int(nparams)

    return factor, metadata


def get_stats(observables: list[dict], covariance: dict=None, unpack: bool=False,
              get_stats_fn=clustering_tools.get_stats_fn, cache_dir: str | Path=None):
    """
    Load and assemble measurement products (data, windows, covariance).

    This function:
      - reads per-statistic measurement files determined by `observables`;
      - optionally caches the assembled likelihood;
      - constructs mock-based covariance from available mock files;
      - returns a :class:`types.GaussianLikelihood` (or unpacked components if requested).

    Parameters
    ----------
    observables : list[dict]
        List of observable option dicts describing which stats to load.
    covariance : dict or None
        Options used to locate covariance/mock files.
    unpack : bool, optional
        If ``True`` return unpacked (data, windows, covariance) rather than a :class:`types.GaussianLikelihood`.
    get_stats_fn : callable
        Function used to locate stats files.
    cache_dir : str or Path, optional
        Directory to use for caching assembled likelihoods.

    Returns
    -------
    types.GaussianLikelihood or tuple
    """
    observables_options = observables
    covariance_options = covariance or {}
    cache_fn = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        _str_from_options = str_from_likelihood_options(
            {'observables': observables_options, 'covariance': covariance_options},
            level={'catalog': 100, 'select': 100, 'covariance': 100},
        )
        cache_fn = cache_dir / 'prepared_stats' / f'{_str_from_options}.h5'
        if cache_fn.exists():
            logger.info(f'Reading cached stats {cache_fn}.')
            likelihood = types.read(cache_fn)
            if unpack:
                return unpack_stats(likelihood)
            return likelihood

    # Helper: iterate over (stat, tracer) combinations
    def iter_stat_tracer_combinations(observables_options, **kwargs):
        """
        Yield (stat, labels, file_kwargs, observable_options) for each requested observable.

        Compact helper for iterating the user-provided observables and producing file kwargs
        and labeling information used when reading files.
        """
        for observable_options in observables_options:
            stat = observable_options['stat']['kind']
            tracers = _make_tuple(observable_options['catalog']['tracer'])
            version = observable_options['catalog'].get('version', None)
            full_tracer = get_full_tracer(tracers, version=version)
            nfields = 3 if 'mesh3' in stat else 2
            simple_tracers = get_simple_tracer(tracers)
            simple_tracers += (simple_tracers[-1],) * (nfields - len(simple_tracers))
            labels = {
                'observables': get_simple_stats(stat),
                'tracers': simple_tracers,
            }
            kw = {}
            if nfields == 3:
                kw['basis'] = observable_options['stat']['basis']
            file_kw = kw | observable_options['catalog'] | {'tracer': full_tracer} | kwargs
            yield stat, labels, file_kw, dict(observable_options)

    def _apply_select(observable: types.ObservableTree, select: dict=None):
        """
        Apply a selection (k-range, ell selection) to an observable.

        The selection dict keys are multipoles (ell) and values are slice-like
        specifications or (min, max, [step]) tuples.
        """
        if select is None:
            return observable
        for ell, limit in select.items():
            pole = observable.get(ells=ell)
            coord_name = list(pole.coords())[0]
            if len(limit) == 3:
                step = limit[2]
                edge = pole.edges(coord_name)[0]
                rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                observable = observable.at(ells=ell).select(**{coord_name: slice(0, None, rebin)})
            observable = observable.at(ells=ell).select(**{coord_name: tuple(limit[:2])})
        observable = observable.get(ells=list(select))
        return observable

    # Loading data, window
    loaded_data, loaded_window, joint_labels = [], [], {'observables': [], 'tracers': []}
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        fn = get_stats_fn(kind=stat, **file_kw)
        if isinstance(fn, list):
            fns = fn
            mocks = [types.read(fn) for fn in fns if fn.exists()]
            data = types.mean(mocks)
        else:
            data = types.read(fn)
        data = _apply_select(data, select=kw['stat'].get('select', None))
        for field, value in labels.items():
            joint_labels[field].append(value)
        file_kw = dict(file_kw)
        imock = file_kw.get('imock', None)
        if imock is not None:
            file_kw['imock'] = 0
        fn = get_stats_fn(kind=f'window_{stat}', **file_kw)
        window = types.read(fn).at.observable.match(data)
        loaded_data.append(data)
        loaded_window.append(window)

    data = pack_stats(loaded_data, **joint_labels)
    window = pack_stats(loaded_window, **joint_labels)

    # Mock-based covariance
    all_fns = []
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        file_kw = file_kw | {'imock': '*'} | covariance_options
        file_kw['tracer'] = get_full_tracer(file_kw['tracer'], version=file_kw['version'])
        all_fns.append(get_stats_fn(kind=stat, **file_kw))
    all_fns = list(zip(*all_fns, strict=True))  # get a list of list of file names
    mocks = []
    for fns in all_fns:
        if all(fn.exists() for fn in fns):
            mock = types.ObservableTree([types.read(fn) for fn in fns], **joint_labels)
            mocks.append(mock)
    covariance = types.cov(mocks)
    covariance.attrs['nobs'] = len(mocks)
    covariance = covariance.at.observable.match(data)

    factor, metadata = _get_covariance_correction_factor(covariance, observables_options, covariance_options)
    if factor != 1.:
        covariance = covariance.clone(value=covariance.value() * factor)
    covariance.attrs['covariance_correction_factor'] = float(factor)
    for name, value in metadata.items():
        covariance.attrs[name] = value
    if metadata['corrections']:
        info = f"Applied covariance corrections {metadata['corrections']} with factor {factor:.6f}"
        if 'hartlap_factor' in metadata:
            info += f", hartlap={metadata['hartlap_factor']:.6f}"
        if 'percival_factor' in metadata:
            info += f", percival={metadata['percival_factor']:.6f}, nparams={metadata['nparams']}"
        logger.info(info)

    likelihood = types.GaussianLikelihood(
        observable=data,
        window=window,
        covariance=covariance,
    )
    if cache_fn is not None:
        likelihood.write(cache_fn)
    if unpack:
        return unpack_stats(likelihood)
    return likelihood


def get_single_likelihood(likelihood_options, stats: types.GaussianLikelihood=None,
                          cosmo=None, fiducial=None, get_stats_fn=clustering_tools.get_stats_fn, cache_dir:str | Path=None):
    """
    Build a single :mod:`desilike` Gaussian likelihood from provided options.

    Parameters
    ----------
    likelihood_options : dict
        Options containing 'observables' list and 'covariance' dict.
    stats : types.GaussianLikelihood or None
        Preloaded measurements (if ``None`` they will be loaded via :func:`get_stats`).
    cosmo, fiducial : optional
        Cosmology objects; if not provided, constructed via get_cosmology.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.

    Returns
    -------
    ObservablesGaussianLikelihood
    """
    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, TracerSpectrum3PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance = likelihood_options.get('covariance', {})
    if cosmo is None:
        cosmo, fiducial = get_cosmology()
    if stats is None:
        stats = get_stats(observables_options, covariance=covariance, unpack=False, get_stats_fn=get_stats_fn, cache_dir=cache_dir)
    data, windows, covariance = unpack_stats(stats)
    labels = covariance.observable.labels(level=1)
    observables = []
    for observable_options, data, window, label in zip(observables_options, data, windows, labels, strict=True):
        stat = observable_options['stat']['kind']
        if 'mesh2_spectrum' in stat:
            cls = TracerSpectrum2PolesObservable
        elif 'mesh3_spectrum' in stat:
            cls = TracerSpectrum3PolesObservable
        else:
            raise NotImplementedError(stat)
        for label, pole in window.observable.items(level=None):
            z = pole.attrs['zeff']
        theory = get_theory(stat, theory=observable_options['theory'], z=z, cosmo=cosmo, fiducial=fiducial)
        theory_params = theory.init.params
        observable = cls(data=data, window=window, theory=theory)
        observable()
        if observable_options['emulator'] is not None:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            cache_dir = Path(cache_dir)
            filename = cache_dir / _str_from_observable_options(observable_options, level={'theory': 100, 'catalog': 2}) / 'emulator.npy'
            filename.parent.mkdir(parents=True, exist_ok=True)
            from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
            if filename.exists():
                logger.info(f'Reading cached emulator {filename}')
                emulated_pt = EmulatedCalculator.load(filename)
            else:
                logger.info(f'Fitting emulator {filename}')
                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=observable_options['emulator'].get('order', 3)),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(filename)
            theory.init.update(pt=emulated_pt)
            theory.init.params.update(theory_params)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables, covariance=covariance.value())


def get_likelihood(likelihoods_options: dict | list[dict], cosmo=None, fiducial=None, get_stats_fn=clustering_tools.get_stats_fn, cache_dir:str | Path=None):
    """
    Build a desilike :class:`SumLikelihood, summed over all tracers.

    Parameters
    ----------
    likelihoods_options : dict, list[dict]
        List of options {'observables': [observable_options, ...], 'covariance': {}}.
    cosmo, fiducial : optional
        Cosmology objects; if not provided, constructed via get_cosmology.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.

    Returns
    -------
    SumLikelihood
    """
    from desilike.likelihoods import SumLikelihood
    if cosmo is None:
        cosmo, fiducial = get_cosmology()
    if isinstance(likelihoods_options, dict):
        likelihoods_options = [likelihoods_options]
    likelihoods = [get_single_likelihood(likelihood_options,
                  cosmo=cosmo, fiducial=fiducial, get_stats_fn=get_stats_fn, cache_dir=cache_dir) for likelihood_options in likelihoods_options]
    return SumLikelihood(likelihoods)


def propose_fiducial_observable_options(stat, tracer=None, zrange=None):
    """Propose fiducial fitting options for given statistics and tracer."""
    propose_fiducial = {'stat': {'kind': stat},
                        'catalog': {'weight': 'default-FKP'},
                        'theory': {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'lor', 'marg': True},
                        'emulator': {},
                        'window': {}}
    propose_stat = {'mesh2_spectrum': {'select': {0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}},
                      'mesh3_spectrum': {'select': {(0, 0, 0): [0.02, 0.12, 0.005], (2, 0, 2): [0.02, 0.08, 0.005]},
                                         'basis': 'sugiyama-diagonal'}}
    propose_theory = {'mesh2_spectrum': {'b3_coev': True},
                      'mesh3_spectrum': {}}
    for _stat in propose_stat:
        if _stat in stat:
            propose_fiducial['stat'].update(propose_stat[_stat])
            propose_fiducial['theory'].update(propose_theory[_stat])
    return propose_fiducial


def propose_fiducial_covariance_options():
    """Return dictionary of default covariance options."""
    return {'version': 'holi-v1-altmtl', 'corrections': ['hartlap', 'percival']}


def propose_fiducial_sampler_options(sampler=None):
    """Return dictionary of default sampler configuration."""
    if sampler is None:
        sampler = 'emcee'
    fiducial_options = {'sampler': sampler, 'init': {},' run': {}, 'nchains': 4}
    return fiducial_options


def propose_fiducial_profiler_options(profiler=None):
    """Return dictionary of default profiler configuration."""
    if profiler is None:
        profiler = 'minuit'
    fiducial_options = {'profiler': profiler, 'init': {}, 'maximize': {}}
    return fiducial_options


def fill_fiducial_observable_options(options):
    """Fill missing observable options with fiducial values."""
    options = dict(options)
    stat = options['stat']['kind']
    tracer, zrange = (options['catalog'][name] for name in ['tracer', 'zrange'])
    fiducial_options = propose_fiducial_observable_options(stat, tracer, zrange)
    options = fiducial_options | options
    for key, value in options.items():
        options[key] = fiducial_options[key] | value
    return options


def fill_fiducial_likelihood_options(options):
    """Fill missing likelihood options with fiducial values."""
    if isinstance(options, dict):
        options = dict(options)
        options['observables'] = [fill_fiducial_observable_options(options) for options in options['observables']]
        options['covariance'] = propose_fiducial_covariance_options() | (options.get('covariance', {}) or {})
        return options
    return type(options)(fill_fiducial_likelihood_options(opts) for opts in options)


def fill_fiducial_options(options):
    """Fill missing options with fiducial values."""
    options = dict(options)
    likelihoods = options.get('likelihoods', None)
    if likelihoods is not None:
        options['likelihoods'] = fill_fiducial_likelihood_options(options['likelihoods'])
    for name in ['sampler', 'profiler']:
        options.setdefault(name, {})
        options[name] = globals()[f'propose_fiducial_{name}_options'](options[name].get(name)) | options[name]
    return options


def generate_likelihood_options_helper(stats=('mesh2_spectrum', 'mesh3_spectrum'),
                                       tracer='LRG', zrange=(0.4, 0.6), region='GCcomb',
                                       version='abacus-2ndgen-complete', covariance='holi-v1-altmtl'):
    """
    Convenience helper that builds a minimal dictionary of likelihood options.

    Parameters
    ----------
    stats : list
        List of statistics in the joint likelihood, from ['mesh2_spectrum', 'mesh3_spectrum']
    tracer : str, tuple
        Tracers to fit.
    zrange : tuple
        Redshift range.
    region : str
        Sky region.
    version : str
        Version of data to use.
    covariance : str
        Version of covariance mocks to use.

    Returns
    -------
    likelihood_options : dict
        Dictionary with keys ['observables', 'covariance'].
        'covariance' is a dictionary specifying how to construct the covariance matrix.
        'observables' contains a list of dictionary (one for each observable), with keys:
        {'stat': {'kind': ..., 'basis': ..., 'select': {...}}, 'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}

    """
    if isinstance(stats, str):
        stats = [stats]
    observables = []
    tracer, zrange = get_full_tracer_zrange(tracer)
    # FIXME
    mock_dir = Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    for stat in stats:
        catalog = {'version': version, 'tracer': tracer, 'zrange': zrange, 'region': region}
        if 'data' not in version:
            catalog['imock'] = '*'  # read all available mocks
            catalog.setdefault('stats_dir', mock_dir)
        observables.append({'stat': {'kind': stat},
                            'catalog': catalog})
    covariance = {'version': covariance, 'stats_dir': mock_dir}
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})


def get_full_tracer_zrange(tracerz=None, zrange=None):
    """
    Translate simple tracer labels, (e.g. LRG1),
    to full tracer and zrange tuples ('LRG', (0.4, 0.6)).

    Parameters
    ----------
    tracerz : str, tuple, list, None
        If None returns the mapping table. If tracerz is a string returns
        (tracer, zrange) or for compound tracer strings returns zipped tuples.

    Returns
    -------
    tracer, zrange
    """
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _get_full_tracer_zrange(tracerz, zrange=zrange):
        if 'x' in tracerz:
            return list(zip(*[_get_full_tracer_zrange(t, zrange=zrange) for t in tracerz.split('x')]))
        if tracerz in translate_zrange:
            # Return tracer and z-range from translate_zrange
            tracer = tracerz[:-1]
            zrange = translate_zrange[tracerz]
        else:
            # Not in translate_zrange
            tracer = tracerz
        return tracer, zrange

    if isinstance(tracerz, str):
        return _get_full_tracer_zrange(tracerz)
    else:  # tuple/list of tracers
        return type(tracerz)(zip(*map(_get_full_tracer_zrange, tracerz)))


def _get_level(level: int | dict=None):
    """Compact helper to normalise verbosity level for string helpers."""
    _default_level = {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0}
    if level is None: level = {}
    if not isinstance(level, dict):
        level = {name: level for name in _default_level}
    level = _default_level | level
    return level


def _str_from_observable_options(options: dict, level: int=None) -> str:
    """Return string identifier given input observable options, with ``level`` of details."""
    level = _get_level(level)
    out_str = []

    # First, catalog
    catalog = _unzip_catalog_options(options['catalog'])

    def _str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer in catalog:
            stracer = get_simple_tracer(tracer)
            catalog_options = catalog[tracer]
            found = False
            for tracerz, zrange in translate_tracerz.items():
                if tracerz.startswith(stracer) and np.allclose(catalog_options['zrange'], zrange):
                    stracer = tracerz  # e.g. LRG1
                    found = True
                    break
            tracer_catalog_str = [stracer]
            if not found or level['catalog'] >= 2:
                tracer_catalog_str.append(_str_zrange(catalog_options['zrange']))
            if level['catalog'] >= 3:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            catalog_str.append('-'.join(tracer_catalog_str))
        out_str.append('x'.join(catalog_str))

    # Then, stat and select, e.g. S2-ell0-0.02-0.2-ell2-0.02-0.2
    translate_stat_name = {'S2': ['mesh2_spectrum'],
                      'S3': ['mesh3_spectrum'],
                      'BAOR': ['bao', 'recon'],
                      'C2R': ['particle2_correlation', 'recon']}
    stat_options = options['stat']
    stat = stat_options['kind']
    if level['stat'] >= 1:
        found = None
        for name in translate_stat_name:
            if all(t in stat for t in translate_stat_name[name]):
                found = name
                break
        if found is None:
            raise ValueError(f'could not find shot naame for {stat}')
        out_str.append(found)
    if level['stat'] >= 2:
        select_str = []
        for ell, limits in stat_options.get('select', {}).items():
            if isinstance(limits, (list, tuple)):
                if isinstance(ell, (list, tuple)):
                    ell = ''.join([str(ell) for ell in ell])
                else:
                    ell = str(ell)
                prec = dict(prec_min=2, prec_max=3) if name.startswith('S') else dict(prec_min=0, prec_max=0)
                select_str.append(f'ell{ell}-' + '-'.join(float2str(lim, **prec) for lim in limits))
        select_str = '-'.join(select_str)
        out_str.append(select_str)

    if level['theory'] > 0:
        out_str.append('th')
        out_str.append(options['theory']['model'])

    return '-'.join(out_str)


def str_from_likelihood_options(likelihood_options, level: int=None):
    """
    Return a compact string identifier for likelihood options.

    Parameters
    ----------
    likelihood_options : dict
        Dictionary with keys 'observables', 'covariance'.
    level : dict
        "Verbosity level". Default is {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0}.
        Increase for more details.
        Covariance level behavior:
        - > 0: include covariance version
        - >= 3: include covariance corrections and optional nparams
    """
    level = _get_level(level)
    out_str = []
    for options in likelihood_options['observables']:
        out_str.append(_str_from_observable_options(options, level=level))
    if level['covariance'] > 0:
        covariance = likelihood_options.get('covariance', {}) or {}
        covariance_str = ['cov-' + covariance.get('version', 'none')]
        if level['covariance'] >= 3:
            corrections = covariance.get('corrections', None)
            if isinstance(corrections, str):
                corrections = [corrections]
            corrections = sorted(str(corr).lower() for corr in (corrections or []))
            if corrections:
                covariance_str.append('corr-' + '-'.join(corrections))
            nparams = covariance.get('nparams', None)
            if nparams is not None:
                covariance_str.append(f'nparams-{int(nparams)}')
        out_str.append('-'.join(covariance_str))
    return '+'.join(out_str)


def get_fits_fn(fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits', kind='chain', likelihoods: list=None,
                sampler: dict=None, profiler: dict=None, ichain: int=None,
                level=None, extra='', ext='npy'):
    """
    Construct a file path for fit outputs based on likelihood and run options.

    Parameters
    ----------
    fits_dir : str, Path
        Base directory for fit outputs.
    kind : str
        Fitting product. Options are 'chain', 'profiles', etc.
    likelihoods : list
        Likelihood options used to build the filename.
    ichain : int or None
        Optional chain index appended to filename.
    extra : str, optional
        Extra suffix to include in the path.
    ext : str, optional
        File extension. Default is 'npy'.

    Returns
    -------
    fn : Path
        Fit file name.
    """
    fits_dir = Path(fits_dir)
    _str_from_options = [str_from_likelihood_options(likelihood_options, level=level) for likelihood_options in likelihoods]
    _str_from_options = '_'.join(_str_from_options)
    extra = f'_{extra}' if extra else ''
    ichain = '_{ichain:d}' if ichain is not None else ''
    return fits_dir / f'{_str_from_options}{extra}' / f'{kind}{ichain}.{ext}'