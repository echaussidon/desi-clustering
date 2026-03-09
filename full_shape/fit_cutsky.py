"""
fit_cutsky.py
=============
Full-shape perturbation theory fits to DR2 cutsky mocks.

Supports three fit modes selected via --fit_type:
  ps    — power spectrum only  (mesh2_spectrum)
  bs    — bispectrum only      (mesh3_spectrum)
  joint — joint PS + BS fit    (mesh2_spectrum + mesh3_spectrum)

The data vector, window matrix, and covariance are assembled by
``prepare_fiducial_likelihoods`` from fiducial_likelihoods.py.
Theory models, emulators, and MCMC sampling are handled by desilike.
"""

import os
os.environ["FOLPS_BACKEND"] = "jax"

import argparse
import itertools
from pathlib import Path

import numpy as np

from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import (
    REPTVelocileptorsTracerPowerSpectrumMultipoles,
    DirectPowerSpectrumTemplate,
)
from desilike.theories.galaxy_clustering.full_shape import (
    FOLPSv2TracerPowerSpectrumMultipoles,
    FOLPSv2TracerBispectrumMultipoles,
)
from desilike.observables.galaxy_clustering import (
    TracerPowerSpectrumMultipolesObservable,
    TracerBispectrumMultipolesObservable,
)
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from desilike.profilers import MinuitProfiler
from desilike.samplers import EmceeSampler
from desilike import setup_logging, ParameterCollection
import lsstypes as types

from full_shape.fiducial_likelihoods import prepare_fiducial_likelihoods

# ─── Fit-mode configuration ──────────────────────────────────────────────────

# Maps --fit_type to the stats list passed to prepare_fiducial_likelihoods.
FIT_TYPE_STATS = {
    'ps':    ['mesh2_spectrum'],
    'bs':    ['mesh3_spectrum'],
    'joint': ['mesh2_spectrum', 'mesh3_spectrum'],
}

BS_ELL_LABEL_TO_TUPLE = {
    '000': (0, 0, 0),
    '202': (2, 0, 2),
}

# ─── Module-level constants ──────────────────────────────────────────────────

# Maps short tracer labels (used on the command line) to the (tracer, zrange)
# pair expected by prepare_fiducial_likelihoods.
TRACER_CONFIG: dict = {
    'LRG1': ('LRG',     (0.4, 0.6)),
    'LRG2': ('LRG',     (0.6, 0.8)),
    'LRG3': ('LRG',     (0.8, 1.1)),
    'QSO':  ('QSO',     (0.8, 2.1)),
    'ELG':  ('ELG_LOP', (1.1, 1.6)),
}

# Effective redshifts at which theory is evaluated.
TRACER_REDSHIFTS = {
    'BGS':  0.295,
    'LRG1': 0.5094,
    'LRG2': 0.7054,
    'LRG3': 0.9264,
    'ELG':  1.3442,
    'QSO':  1.4864,
}

# Fiducial sigma_8(z_eff) values used as prior centres in the physical basis.
TRACER_SIGMA8 = {
    'BGS':  0.69376997,
    'LRG1': 0.62404056,
    'LRG2': 0.53930063,
    'LRG3': 0.50823223,
    'ELG':  0.43197292,
    'QSO':  0.41825647,
}


def _is_physical_basis(prior_basis):
    return prior_basis in ('physical', 'physical_prior_doc', 'physical_aap', 'tcm_chudaykin') or str(prior_basis).startswith('physical_')


# ─── Cosmology ───────────────────────────────────────────────────────────────

def get_cosmology():
    """Return a desilike Cosmoprimo calculator and the DESI fiducial cosmology.

    The calculator has free parameters h, omega_cdm, omega_b, logA with
    uniform priors; n_s and tau_reio are fixed.  A Gaussian prior is placed
    on omega_b from simulations.

    Returns
    -------
    cosmo : Cosmoprimo
    fiducial : cosmoprimo.Cosmology
    """
    cosmo = Cosmoprimo(engine='class')
    cosmo.init.params['H0'] = dict(derived=True)
    cosmo.init.params['Omega_m'] = dict(derived=True)
    cosmo.init.params['sigma8_m'] = dict(derived=True)
    fiducial = DESI()

    param_config = {
        'tau_reio': dict(fixed=True),
        'n_s':      dict(fixed=True),
        'omega_b':  dict(fixed=False, prior={'dist': 'norm',    'loc': 0.02237,  'scale': 0.00037}),
        'h':        dict(fixed=False, prior={'dist': 'uniform', 'limits': [0.5,  0.9]}),
        'omega_cdm':dict(fixed=False, prior={'dist': 'uniform', 'limits': [0.05, 0.2]}),
        'logA':     dict(fixed=False, prior={'dist': 'uniform', 'limits': [2.0,  4.0]}),
    }
    for name, cfg in param_config.items():
        cosmo.params[name].update(**cfg)

    return cosmo, fiducial


# ─── Nuisance priors ─────────────────────────────────────────────────────────

def get_nuisance_priors(
    fit_type,
    prior_basis,
    width_EFT=12.5,
    width_SN0=2.0,
    width_SN2=5.0,
    pt_model='folpsD',
    b3_coev=True,
    sigma8_fid=None,
):
    """Build a dict of nuisance parameter prior configurations.

    Parameters
    ----------
    fit_type : str
        'ps', 'bs', or 'joint'.  Determines which stochastic parameters are
        included (BS stochastic terms are only added for 'bs' and 'joint').
    prior_basis : str
        'physical' or 'physical_prior_doc' uses physical bias parameters
        (b1p, b2p, …).  Any other value uses the standard Eulerian basis
        (b1, b2, …).
    width_EFT : float
        Standard deviation of Gaussian priors on PS counter-terms.
    width_SN0 : float
        Standard deviation of Gaussian prior on shot-noise monopole.
    width_SN2 : float
        Standard deviation of Gaussian prior on shot-noise quadrupole.
    pt_model : str
        PT model tag. When 'EFT', FoG parameters are fixed.
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float or None
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        ``Parameter.update()`` (e.g. ``{'fixed': True}`` or
        ``{'prior': {...}}``).
    """
    params = {}
    include_bs_stochastic = (fit_type in ('bs', 'joint'))

    if _is_physical_basis(prior_basis):
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2/7 * sigma8_fid**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed': True}
        else:
            params['b3p'] = {
                'prior': {'dist': 'norm', 'loc': 23/42 * sigma8_fid**4, 'scale': sigma8_fid**4},
                'fixed': False,
            }
        # ── PS counter-terms and shot noise ───────────────────────────────
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        # ── FoG damping ───────────────────────────────────────────────────
        if pt_model == 'EFT':
            params['X_FoG_pp'] = {'fixed': True}
            params['X_FoG_bp'] = {'fixed': True}
        else:
            params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
            params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
        # ── BS stochastic parameters (only for bs / joint) ────────────────
        if include_bs_stochastic:
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed': True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        # ── PS counter-terms and shot noise ───────────────────────────────
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        # ── FoG damping ───────────────────────────────────────────────────
        if pt_model == 'EFT':
            params['X_FoG_p'] = {'fixed': True}
            params['X_FoG_b'] = {'fixed': True}
        else:
            params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
            params['X_FoG_b'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
        # ── BS stochastic parameters (only for bs / joint) ────────────────
        if include_bs_stochastic:
            Ppoisson = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson * 4}}
    return params


# ─── Theory ──────────────────────────────────────────────────────────────────

def get_theory(
    tracer,
    fit_type,
    fiducial,
    cosmo,
    pt_model='folpsD',
    prior_basis='physical',
    width_EFT=12.5,
    width_SN0=2.0,
    width_SN2=5.0,
    b3_coev=True,
    A_full_status=False,
    damping='lor',
):
    """Build PS and / or BS theory models for the requested fit type.

    Parameters
    ----------
    tracer : str
        Short tracer label, e.g. 'LRG2'.
    fit_type : str
        'ps', 'bs', or 'joint'.
    fiducial : cosmoprimo.Cosmology
        DESI fiducial cosmology.
    cosmo : Cosmoprimo
        Cosmology calculator.
    pt_model : str
        PT model: 'folpsD', 'EFT', or 'rept_velocileptors'.
    prior_basis : str
        'physical' or standard Eulerian basis.
    width_EFT, width_SN0, width_SN2 : float
        Prior widths for counter-terms and shot noise.
    b3_coev : bool
        Fix b3 to its co-evolution value.
    A_full_status : bool
        Use the full A-matrix in FOLPS.
    damping : str
        FoG damping kernel ('lor', 'exp', 'vdg').

    Returns
    -------
    theories : dict
        Subset of {'ps': ..., 'bs': ...} according to fit_type.
    """
    z = TRACER_REDSHIFTS[tracer]
    sigma8_fid = fiducial.sigma8_z(z)
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

    theories = {}

    # ── Power spectrum theory ─────────────────────────────────────────────
    if fit_type in ('ps', 'joint'):
        if pt_model == 'rept_velocileptors':
            ps_theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=prior_basis,
            )
        else:
            ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=prior_basis,
                A_full=A_full_status,
                b3_coev=b3_coev,
                damping=damping,
                sigma8_fid=sigma8_fid,
                h_fid=fiducial.h,
            )
        theories['ps'] = ps_theory

    # ── Bispectrum theory ─────────────────────────────────────────────────
    if fit_type in ('bs', 'joint'):
        bs_theory = FOLPSv2TracerBispectrumMultipoles(
            template=template,
            prior_basis=prior_basis,
            A_full=A_full_status,
            damping=damping,
            sigma8_fid=sigma8_fid,
            h_fid=fiducial.h,
        )
        theories['bs'] = bs_theory

    # ── Apply nuisance priors to the theory that owns each parameter ──────
    # get_nuisance_priors returns a plain dict[str, dict], so **p unpacks
    # correctly into Parameter.update() keyword arguments.
    params = get_nuisance_priors(
        fit_type, prior_basis, width_EFT, width_SN0, width_SN2,
        pt_model=pt_model, b3_coev=b3_coev, sigma8_fid=sigma8_fid,
    )
    for name, p in params.items():
        for theory in theories.values():
            if name in theory.params:
                theory.params[name].update(**p)

    return theories


# ─── Likelihood components extraction ────────────────────────────────────────

def _extract_ps_components(lk):
    """Extract PS observable, covariance, and window from a lsstypes GaussianLikelihood.

    All returned objects are lsstypes instances so that desilike receives
    full k-grid and observable metadata.

    Parameters
    ----------
    lk : types.GaussianLikelihood

    Returns
    -------
    obs : ObservableTree   PS data vector (spectrum2 branch).
    cov : CovarianceMatrix PS covariance sub-block.
    window : WindowMatrix  PS window matrix.
    """
    obs = lk.observable
    if isinstance(obs, types.ObservableTree):
        obs = obs.get(observables='spectrum2')

    cov = lk.covariance
    if isinstance(cov.observable, types.ObservableTree):
        cov = cov.at.observable.get(observables='spectrum2')
    cov = cov.at.observable.match(obs)

    window = lk.window
    if isinstance(window, types.WindowMatrix):
        if isinstance(window.observable, types.ObservableTree):
            try:
                window = window.at.observable.get(observables='spectrum2')
            except ValueError:
                pass
        if isinstance(window.theory, types.ObservableTree):
            try:
                window = window.at.theory.get(observables='spectrum2')
            except ValueError:
                pass
        window = window.at.observable.match(obs)

    return obs, cov, window


def _extract_bs_components(lk):
    """Extract BS observable, covariance, and window from a lsstypes GaussianLikelihood.

    All returned objects are lsstypes instances so that desilike receives
    full k-grid and observable metadata.

    Parameters
    ----------
    lk : types.GaussianLikelihood

    Returns
    -------
    obs : ObservableTree   BS data vector (spectrum3 branch).
    cov : CovarianceMatrix BS covariance sub-block.
    window : WindowMatrix  BS window matrix.
    """
    obs = lk.observable
    if isinstance(obs, types.ObservableTree):
        obs = obs.get(observables='spectrum3')

    cov = lk.covariance
    if isinstance(cov.observable, types.ObservableTree):
        cov = cov.at.observable.get(observables='spectrum3')
    cov = cov.at.observable.match(obs)

    window = lk.window
    
    if isinstance(window, types.WindowMatrix):
        if isinstance(window.observable, types.ObservableTree):
            try:
                window = window.at.observable.get(observables='spectrum3')
            except ValueError:
                pass
        if isinstance(window.theory, types.ObservableTree):
            try:
                window = window.at.theory.get(observables='spectrum3')
            except ValueError:
                pass
        window = window.at.observable.match(obs)

    return obs, cov, window

# ─── Observable builder ──────────────────────────────────────────────────────

def get_observables(
    tracer,
    region='GCcomb',
    fit_type='joint',
    data='abacus-2ndgen-complete',
    covariance='holi-v1-altmtl',
    weight='default-FKP',
    pt_model='folpsD',
    prior_basis='physical',
    width_EFT=12.5,
    width_SN0=2.0,
    width_SN2=5.0,
    b3_coev=True,
    A_full_status=False,
    damping='lor',
    cuts_kwargs=None,
):
    """Build desilike observables for a given tracer and fit mode.

    Calls ``prepare_fiducial_likelihoods`` to get data, window, and covariance,
    then extracts the relevant arrays and constructs desilike observable objects.

    Parameters
    ----------
    tracer : str
        Short tracer label, e.g. 'LRG2'.
    region : str
        Sky region: 'NGC', 'SGC', or 'GCcomb'.
    fit_type : str
        'ps' (power spectrum only), 'bs' (bispectrum only), or
        'joint' (PS + BS combined fit).
    data : str
        Data product identifier forwarded to prepare_fiducial_likelihoods.
    covariance : str
        Covariance mock set identifier.
    weight : str
        Weighting scheme.
    pt_model : str
        PT model: 'folpsD', 'EFT', or 'rept_velocileptors'.
    prior_basis : str
        Bias parametrisation basis.
    width_EFT, width_SN0, width_SN2 : float
        Prior widths for counter-terms and shot noise.
    b3_coev : bool
        Fix b3 to its co-evolution value.
    A_full_status : bool
        Use the full A-matrix in FOLPS.
    damping : str
        FoG damping kernel.

    Returns
    -------
    obs_dict : dict
        Keys: whichever of 'ps', 'bs' are active, plus 'cov' (full joint
        covariance as a numpy array) and 'theories'.
    """
    tracer_label, zrange = TRACER_CONFIG[tracer]
    stats = FIT_TYPE_STATS[fit_type]

    print(f'  Loading precomputed likelihood  (fit_type={fit_type})')
    lk = prepare_fiducial_likelihoods(
        tracer=tracer_label,
        zrange=zrange,
        region=region,
        weight=weight,
        stats=stats,
        data=data,
        covariance=covariance,
        cuts_kwargs=cuts_kwargs,
    )

    # Full covariance matrix (needed for ObservablesGaussianLikelihood)
    full_cov = lk.covariance.value()

    cosmo, fiducial = get_cosmology()
    theories = get_theory(
        tracer, fit_type, fiducial, cosmo,
        pt_model=pt_model, prior_basis=prior_basis,
        width_EFT=width_EFT, width_SN0=width_SN0, width_SN2=width_SN2,
        b3_coev=b3_coev, A_full_status=A_full_status, damping=damping,
    )

    obs_dict = {'theories': theories, 'cov': full_cov}

    # ── Power spectrum observable ─────────────────────────────────────────
    if fit_type in ('ps', 'joint'):
        obs_ps, cov_ps, wmat_ps = _extract_ps_components(lk)
        ps_obs = TracerPowerSpectrumMultipolesObservable(
            data=obs_ps,
            covariance=cov_ps,
            window=wmat_ps,
            theory=theories['ps'],
        )
        obs_dict['ps'] = ps_obs

    # ── Bispectrum observable ─────────────────────────────────────────────
    if fit_type in ('bs', 'joint'):
        obs_bs, cov_bs, wmat_bs = _extract_bs_components(lk)
        bs_obs = TracerBispectrumMultipolesObservable(
            data=obs_bs,
            covariance=cov_bs,
            window=wmat_bs,
            theory=theories['bs'],
        )
        obs_dict['bs'] = bs_obs

    return obs_dict


# ─── Emulator ────────────────────────────────────────────────────────────────

def set_emulator(observables, emulator_dir='./emulators', order=4):
    """Fit or load Taylor emulators for each active theory component.

    Emulator files are stored as
    ``{emulator_dir}/{tracer}_{comp}_emulator.npy``.

    Parameters
    ----------
    observables : dict
        ``{tracer: obs_dict}`` as returned by :func:`get_observables`.
    emulator_dir : str or Path
        Directory where emulator files are saved / loaded from.
    order : int
        Taylor expansion order.

    Returns
    -------
    observables : dict
        Same dict with PT calculators replaced by emulated versions.
    """
    emulator_dir = Path(emulator_dir)

    for tracer, obs_dict in observables.items():
        theories = obs_dict['theories']
        for comp in ('ps', 'bs'):
            if comp not in obs_dict:
                continue  # component not active for this fit type

            obs = obs_dict[comp]
            filename = emulator_dir / f'{tracer}_{comp}_emulator.npy'
            filename.parent.mkdir(parents=True, exist_ok=True)

            if filename.exists():
                print(f'  Loading {comp.upper()} emulator for {tracer}: {filename}')
                emulated_pt = EmulatedCalculator.load(filename)
            else:
                print(f'  Fitting {comp.upper()} emulator for {tracer}: {filename}')
                theory_calc = obs.theory
                emulator = Emulator(
                    theory_calc.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=order),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(filename)

            theories[comp].init.update(pt=emulated_pt)

    return observables


# ─── Analytic marginalisation ────────────────────────────────────────────────

def set_analytic_marginalization(observables, prior_basis, derived='.marg'):
    """Mark PS EFT counter-terms and shot-noise parameters for analytic marginalisation.

    Only PS parameters are marginalised analytically.  BS stochastic
    parameters (c1, c2, Pshot, Bshot) are always sampled explicitly.

    Parameters
    ----------
    observables : dict
        ``{tracer: obs_dict}`` as returned by :func:`get_observables`.
    prior_basis : str
        Determines which parameter names to marginalise.

    Returns
    -------
    observables : dict
    """
    if _is_physical_basis(prior_basis):
        marg_params = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        marg_params = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for obs_dict in observables.values():
        if 'ps' not in obs_dict:
            continue  # BS-only fit; nothing to marginalise analytically
        ps_obs = obs_dict['ps']
        for param in marg_params:
            ps_obs.theory.init.params[param].update(derived=derived)

    return observables


# ─── Likelihood ──────────────────────────────────────────────────────────────

def get_likelihood(observables, fit_type):
    """Build a desilike GaussianLikelihood, summed over all tracers.

    For 'joint' fits, a single ObservablesGaussianLikelihood is built per
    tracer using the full joint PS+BS covariance.  For single-stat fits,
    only the relevant component is passed.

    Parameters
    ----------
    observables : dict
        ``{tracer: obs_dict}`` as returned by :func:`get_observables`.
    fit_type : str
        'ps', 'bs', or 'joint'.

    Returns
    -------
    SumLikelihood
    """
    likelihoods = []

    for obs_dict in observables.values():
        active = [obs_dict[comp] for comp in ('ps', 'bs') if comp in obs_dict]
        # For joint fits, pass the full joint covariance so the PS-BS
        # cross-covariance block is included.  For single-stat fits the
        # per-observable covariance (already embedded in each observable)
        # is sufficient and no extra argument is needed.
        kw = {'covariance': obs_dict['cov']} if fit_type == 'joint' else {}
        likelihoods.append(
            ObservablesGaussianLikelihood(observables=active, **kw)
        )

    return SumLikelihood(likelihoods)


# ─── MCMC ────────────────────────────────────────────────────────────────────

def sample(likelihood, filename, GR_criteria=0.1):
    """
    Run an Emcee MCMC chain until the Gelman-Rubin criterion is met.

    Parameters
    ----------
    likelihood : SumLikelihood
        Joint likelihood to sample.
    filename : Path, str
        File path prefix for saving the chain.
    GR_criteria : float
        Gelman-Rubin convergence threshold on the maximum eigenvalue.
    """
    for param in likelihood.all_params.select(solved=True):
        param.update(derived='.marg')
    sampler = EmceeSampler(likelihood, save_fn=filename)
    sampler.run(check={'max_eigen_gr': GR_criteria})


def profile(likelihood, filename):
    """
    Run likelihood profiling.

    Parameters
    ----------
    likelihood : SumLikelihood
        Joint likelihood to profile.
    """
    for param in likelihood.all_params.select(solved=True):
        param.update(derived='.best')
    profiler = MinuitProfiler(likelihood, save_fn=filename)
    profiler.maximize()
    print(profiler.profiles.to_stats(tablefmt='pretty'))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit DESI cutsky clustering statistics (PS, BS, or joint).',
    )
    parser.add_argument(
        '--todo', type=str, default='sample',
        choices=['profile', 'sample'], nargs='*',
        help='Run best fit (maximize) and / or sample.',
    )
    parser.add_argument(
        '--fit_type', type=str, default='joint',
        choices=['ps', 'bs', 'joint'],
        help='Statistics to fit: ps (power spectrum), bs (bispectrum), '
             'or joint (PS + BS). Default: joint.',
    )
    parser.add_argument(
        '--tracers', nargs='+', type=str, default=['LRG2'],
        help='Tracer labels (default: LRG2).',
    )
    parser.add_argument(
        '--region', type=str, default='GCcomb',
        help='Sky region (default: GCcomb).',
    )
    parser.add_argument(
        '--data', type=str, default='abacus-2ndgen-complete',
        help='Data product identifier (default: abacus-2ndgen-complete).',
    )
    parser.add_argument(
        '--covariance', type=str, default='holi-v1-altmtl',
        help='Covariance mock set (default: holi-v1-altmtl).',
    )
    parser.add_argument(
        '--pt_model', type=str, default='folpsD',
        help='PT model: folpsD, EFT, rept_velocileptors. Default: folpsD.',
    )
    parser.add_argument(
        '--prior_basis', type=str, default='physical_aap',
           help='Bias parametrisation: physical, physical_aap, physical_prior_doc, '
               'tcm_chudaykin, or standard. Default: physical_aap.',
    )
    parser.add_argument(
        '--width_EFT', type=float, default=12.5,
        help='Gaussian prior width on EFT counter-terms (default: 12.5).',
    )
    parser.add_argument(
        '--width_SN0', type=float, default=2.0,
        help='Gaussian prior width on shot-noise monopole (default: 2.0).',
    )
    parser.add_argument(
        '--width_SN2', type=float, default=5.0,
        help='Gaussian prior width on shot-noise quadrupole (default: 5.0).',
    )
    parser.add_argument(
        '--ps_ells', nargs='+', type=int, default=[0, 2],
        help='PS multipoles to keep (default: 0 2).',
    )
    parser.add_argument(
        '--ps_kmin', type=float, default=0.02,
        help='Minimum PS k for data/cov cuts (default: 0.02).',
    )
    parser.add_argument(
        '--ps_kmax', type=float, default=0.20,
        help='Maximum PS k for data/cov cuts (default: 0.20).',
    )
    parser.add_argument(
        '--ps_stride', type=int, default=5,
        help='PS k-bin subsampling stride; set to 1 for no thinning (default: 5).',
    )
    parser.add_argument(
        '--bs_kmin', type=float, default=0.02,
        help='Minimum BS k for data/cov cuts (default: 0.02).',
    )
    parser.add_argument(
        '--bs_ells', nargs='+', type=str, default=['000', '202'],
        choices=['000', '202'],
        help='BS multipoles to keep (default: 000 202).',
    )
    parser.add_argument(
        '--bs_kmax_b0', type=float, default=0.12,
        help='Maximum BS k for (0,0,0) multipole (default: 0.12).',
    )
    parser.add_argument(
        '--bs_kmax_b2', type=float, default=0.08,
        help='Maximum BS k for (2,0,2) multipole (default: 0.08).',
    )
    parser.add_argument(
        '--emulator_dir', type=str, default='./emulators',
        help='Directory for Taylor emulator files (default: ./emulators).',
    )
    parser.add_argument(
        '--emulator_order', type=int, default=3,
        help='Taylor emulator expansion order (default: 3).',
    )
    parser.add_argument(
        '--GR_criteria', type=float, default=0.1,
        help='Gelman-Rubin convergence threshold (default: 0.3).',
    )
    parser.add_argument(
        '--no_emulator', action='store_true',
        help='Skip emulator fitting / loading and use the full PT calculation.',
    )
    parser.add_argument(
        '--no_analytic_marg', action='store_true',
        help='Disable analytic marginalisation of PS nuisance parameters.',
    )
    out_dir = Path(os.getenv('SCRATCH')) / 'fits'
    parser.add_argument('--out_dir', type=str, default=out_dir,
                       help=f'Output directory for fitting results, default is {out_dir}')
    args = parser.parse_args()

    setup_logging()

    # ── Build observables for each tracer ─────────────────────────────────
    bs_ells = tuple(BS_ELL_LABEL_TO_TUPLE[label] for label in args.bs_ells)
    cuts_kwargs = {
        'mesh2_spectrum': {
            'ells': tuple(args.ps_ells),
            'kmin': args.ps_kmin,
            'kmax': args.ps_kmax,
            'rebin': args.ps_stride,
        },
        'mesh3_spectrum': {
            'ells': bs_ells,
            'kmin': args.bs_kmin,
            'kmax_b0': args.bs_kmax_b0,
            'kmax_b2': args.bs_kmax_b2,
        },
    }

    observables = {}
    for tracer in args.tracers:
        print(f'\n{"=" * 60}')
        print(f'Building observable: tracer={tracer}  region={args.region}  '
              f'fit_type={args.fit_type}')
        print(f'{"=" * 60}')
        observables[tracer] = get_observables(
            tracer=tracer,
            region=args.region,
            fit_type=args.fit_type,
            data=args.data,
            covariance=args.covariance,
            pt_model=args.pt_model,
            prior_basis=args.prior_basis,
            width_EFT=args.width_EFT,
            width_SN0=args.width_SN0,
            width_SN2=args.width_SN2,
            cuts_kwargs=cuts_kwargs,
        )

    # ── Optionally fit / load emulators ───────────────────────────────────
    if not args.no_emulator:
        observables = set_emulator(
            observables,
            emulator_dir=args.emulator_dir,
            order=args.emulator_order,
        )

    # # ── Optionally set analytic marginalisation on PS nuisance params ─────
    if not args.no_analytic_marg and args.fit_type in ('ps', 'joint'):
        observables = set_analytic_marginalization(observables, args.prior_basis)

    # ── Build likelihood and run MCMC ─────────────────────────────────────
    likelihood = get_likelihood(observables, args.fit_type)

    tracer_str = '_'.join(args.tracers)
    chain_fn = args.out_dir / f'chain_{args.fit_type}_{tracer_str}_{args.region}'
    profiles_fn = args.out_dir / f'profiles_{args.fit_type}_{tracer_str}_{args.region}.npy'

    if 'sample' in args.todo:
        sample(likelihood, chain_fn, GR_criteria=args.GR_criteria)

    if 'profile' in args.todo:
        profile(likelihood, profiles_fn)
