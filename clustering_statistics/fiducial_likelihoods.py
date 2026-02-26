import numpy as np

from . import tools

import lsstypes as types



def prepare_fiducial_likelihoods(tracer: str | tuple | list='LRG',
                                 zrange: tuple=(0.4, 0.6),
                                 region: str='GCcomb',
                                 weight: str='default-FKP',
                                 stats: tuple | list | dict=['mesh2_spectrum'],
                                 data: str='abacus-2ndgen-complete',
                                 covariance: str='holi-v1-altmtl',
                                 rotation: bool | str=False):
    """
    Return pre-defined likelihood assembled from precomputed measurements.

    Parameters
    ----------
    tracer : str, tuple, or list
        Tracer label (e.g. 'LRG', 'ELG', ...), a joint tracer ('LRG', 'QSO'),
        or a list of tracers to include in the likelihood.
    zrange : tuple(float, float)
        Redshift range used to select the measurements.
    region : str
        Region (typically NGC, SGC, or GCcomb) used to select the measurements.
    weight : str
        Weighting scheme name used to select the appropriate measurements.
    stats : sequence or dict
        Statistics to include. If a sequence (default ['mesh2_spectrum']) each
        element is treated as a stat key; if a dict it is interpreted as {stat: kw},
        where kw are extra keywords used to select the measurements (e.g. version).
    data : str
        Data product identifier used to read the measured observables (e.g. mock
        set name or data release).
    covariance : str
        Version of the mocks. If set to 'holi', the covariance is estimated from 1000 mocks.
    rotation : bool or str
        If ``True``, or 'marg', apply precomputed window rotation, with covariance matrix
        including the marginalization. Else marginalization priors are included in the window matrix,
        as 'observablemarg' "observables" labels.

    Returns
    -------
    likelihood : GaussianLikelihood
    """
    tracers = tracer
    if not isinstance(tracers, list):
        tracers = [tracers]
    tracers = [tools._make_tuple(tracer) for tracer in tracers]
    if not isinstance(stats, dict):
        stats = {stat: {} for stat in stats}

    likelihood = {}

    def iter_stats(stats, **kwargs):
        base_kw = dict(zrange=zrange, region=region, weight=weight, basis='sugiyama-diagonal') | kwargs
        version = base_kw.get('version')
        for stat in stats:
            for tracer in tracers:
                stat_tracer = tracer
                if 'ELG' in tracer and 'altmtl' in version:
                    stat_tracer = 'ELG_LOPnotqso'
                simple_tracers = tuple(tools.get_simple_tracer(tr) for tr in tracer)
                simple_tracers += (simple_tracers[-1],) * ((3 if 'mesh3' in stat else 2) - len(simple_tracers))
                labels = {'observables': tools.get_simple_stats(stat), 'tracers': simple_tracers}
                yield (stat, tracer), labels, base_kw | dict(tracer=stat_tracer) | stats[stat]

    def read_stats(stats, **kwargs):
        observables, labels, not_exists = [], {'observables': [], 'tracers': []}, []
        for (stat, tracer), _labels, kw in iter_stats(stats, **kwargs):
            fn = tools.get_stats_fn(kind=stat, **kw)
            if fn.exists():
                observables.append(types.read(fn))
                for name, label in _labels.items(): labels[name].append(label)
            else:
                not_exists.append((stat, tracer))
        if observables:
            assert not not_exists, f'these measurements do not exist: {not_exists}'
            if isinstance(observables[0], types.WindowMatrix):
                # Let's build the joint window matrix
                value, observable, theory = [], [], []
                for window in observables:
                    value.append(window.value)
                    observable.append(window.observable)
                    theory.append(window.theory)
                from scipy import linalg
                value = linalg.block_diag(value)
                observable = types.ObservableTree(observable, **labels)
                theory = types.ObservableTree(theory, **labels)
                return types.WindowMatrix(value=value, theory=theory, observable=observable)
            else:
                return types.ObservableTree(observables, **labels)
        else:
            return None

    at_rotations = []
    marg_rotation = isinstance(rotation, bool) or ('marg' in rotation)
    # First, data vector
    if 'abacus' in data:
        kwargs = dict(version=data, stats_dir=tools.desi_dir / 'mocks/cai/LSS/DA2/mocks/desipipe')
        list_observables = []
        for imock in range(25):
            observables = read_stats(stats, imock=imock, **kwargs)
            if observables is not None:
                list_observables.append(observables)
        likelihood['observable'] = types.mean(list_observables)
        # Second, window matrix
        likelihood['window'] = read_stats(stats, imock=0, **kwargs)
        # Apply rotation, if requested
        if rotation:
            priors = []
            for (stat, tracer), at, kw in iter_stats(stats, imock=0, **kwargs):
                if 'mesh2_spectrum' in stat:
                    fn = tools.get_stats_fn(kind=f'rotation_mesh2_spectrum', **kw)
                    rotation = types.read(fn)
                    at_rotations.append((at, rotation))
                    likelihood['observable'] = likelihood['observable'].match(rotation.observable)
                    likelihood['window'] = likelihood['window'].at.observable.match(likelihood['observable'])
                    likelihood['window'], likelihood['observable'] = rotation.rotate(window=likelihood['window'], data=likelihood['observable'], at=at, prior_data=True)
                    if not marg_rotation:
                        priors.append(rotation.prior(at=at))  # this is a types.WindowMatrix
            # If no analytic marginalization, include k-template in the window matrix
            if priors:
                window = likelihood['window']
                value = [window.value()]
                theory = [window.theory]
                for prior, (at, _) in zip(priors, at_rotations):
                    value.append(prior.value())
                    labels = dict(at)
                    labels['observables'] = labels['observables'] + 'marg'  # add marg label
                    theory.append(types.ObservableTree([prior.theory], **labels))
                value = np.concatenate(value, axis=-1)
                likelihood['window'] = window.clone(value=value, theory=theory)
    else:
        raise NotImplementedError

    # Third, covariance matrix
    if 'holi' in covariance:
        kwargs = dict(version=covariance, stats_dir=tools.desi_dir / 'mocks/cai/LSS/DA2/mocks/desipipe')
        list_observables = []
        for imock in range(1000):
            observables = read_stats(stats, imock=imock, **kwargs)
            if observables is not None:
                list_observables.append(observables)
        likelihood['covariance'] = types.cov(list_observables).at.observable.match(likelihood['observable'])
        likelihood['covariance'].attrs['nobs'] = len(list_observables)
        # Apply rotation, if requested
        if rotation:
            for at, rotation in at_rotations:
                likelihood['covariance'] = rotation.rotate(covariance=likelihood['covariance'], at=at, prior_cov=marg_rotation)
    else:
        raise NotImplementedError

    # Match window and covariance to data, just in case
    for name in ['window', 'covariance']:
        likelihood[name] = likelihood[name].at.observable.match(likelihood['observable'])

    # Build the likelihood object
    return types.GaussianLikelihood(**likelihood)
