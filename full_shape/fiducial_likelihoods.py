import numpy as np
from scipy import linalg

from clustering_statistics import tools
import lsstypes as types


def _match_observable(obj, target):
    """
    Match ``obj``'s observable coordinates to those of ``target``.

    ``ObservableTree`` exposes ``.match()`` directly, while ``WindowMatrix``
    and ``CovarianceMatrix`` require going through ``.at.observable.match()``.
    This helper dispatches to the correct call so callers don't need to care
    about the underlying type.

    For a joint (multi-stat) fit the ``WindowMatrix`` already carries an
    ``ObservableTree`` observable that covers all stats; in that case we match
    against the full joint target rather than stripping it to one branch.
    """
    if isinstance(obj, types.ObservableTree):
        return obj.match(target)
    if isinstance(obj, types.WindowMatrix) and isinstance(target, types.ObservableTree):
        # Only strip the ObservableTree to its first branch when the window covers
        # a single stat.  A joint window has its observable constructed explicitly
        # as types.ObservableTree (exact type), while a single-stat window has a
        # concrete subclass (e.g. Mesh2SpectrumPoles) that also inherits from
        # ObservableTree but lacks the 'observables' tree dimension.
        # Use an exact-type check (not isinstance) to distinguish them.
        if type(obj.observable) is not types.ObservableTree:
            target = next(iter(target))
    return obj.at.observable.match(target)


def prepare_fiducial_likelihoods(
    tracer: str | tuple | list = 'LRG',
    zrange: tuple = (0.4, 0.6),
    region: str = 'GCcomb',
    weight: str = 'default-FKP',
    stats: tuple | list | dict = ['mesh2_spectrum'],
    data: str = 'abacus-2ndgen-complete',
    covariance: str = 'holi-v1-altmtl',
    rotation: bool | str = False,
    cuts_kwargs: dict | None = None,
):
    """
    Return a pre-defined GaussianLikelihood assembled from precomputed measurements.

    Parameters
    ----------
    tracer : str, tuple, or list
        Tracer label (e.g. 'LRG', 'ELG'), a joint tracer tuple ('LRG', 'QSO'),
        or a list of tracers to combine in the likelihood.
    zrange : tuple(float, float)
        Redshift range used to select measurements.
    region : str
        Sky region label, typically 'NGC', 'SGC', or 'GCcomb'.
    weight : str
        Weighting scheme used to select measurements.
    stats : sequence or dict
        Statistics to include. A plain sequence (e.g. ['mesh2_spectrum']) uses
        default selection keywords; a dict maps {stat: extra_kw} to pass
        additional selection keywords (e.g. version) per statistic.
    data : str
        Data product identifier for the measured observables (e.g. mock set name
        or data release label).
    covariance : str
        Mock set used to estimate the covariance matrix. If 'holi' is in the
        string, 1000 mocks are used.
    rotation : bool or str
        Controls window-function rotation:
        - False (default): no rotation applied.
        - True or 'marg': apply rotation with analytic marginalization folded
          into the covariance matrix.
        - any other str without 'marg': apply rotation with marginalization
          priors appended to the window matrix instead.

    Returns
    -------
    likelihood : types.GaussianLikelihood
    """
    # ------------------------------------------------------------------
    # Normalise inputs
    # ------------------------------------------------------------------
    # Always work with a list of tracers, each stored as a tuple.
    tracers = tracer if isinstance(tracer, list) else [tracer]
    tracers = [tools._make_tuple(t) for t in tracers]

    # Always work with a dict so every stat can carry extra keywords.
    if not isinstance(stats, dict):
        stats = {stat: {} for stat in stats}

    # Analytic marginalization is used when rotation=True or 'marg' in rotation.
    use_analytic_marg = (rotation is True) or (isinstance(rotation, str) and 'marg' in rotation)

    # ------------------------------------------------------------------
    # Helper: iterate over (stat, tracer) combinations
    # ------------------------------------------------------------------
    def iter_stat_tracer_combos(stats, **kwargs):
        """
        Yield (key, labels, file_kwargs) for every (stat, tracer) combination.

        key        : (stat, tracer) — identifies the combination.
        labels     : {'observables': ..., 'tracers': ...} for ObservableTree.
        file_kwargs: keyword arguments to pass to tools.get_stats_fn.
        """
        base_kw = dict(
            zrange=zrange,
            region=region,
            weight=weight,
            basis='sugiyama-diagonal',
        ) | kwargs
        version = base_kw.get('version')

        for stat in stats:
            for tracer in tracers:
                # ELG requires a different internal label for altmtl mocks.
                file_tracer = tracer
                if 'ELG' in tracer and version is not None and 'altmtl' in version:
                    file_tracer = 'ELG_LOPnotqso'

                # Pad simple_tracers to the number of fields expected by the statistic.
                n_fields = 3 if 'mesh3' in stat else 2
                simple_tracers = tuple(tools.get_simple_tracer(tr) for tr in tracer)
                simple_tracers += (simple_tracers[-1],) * (n_fields - len(simple_tracers))

                labels = {
                    'observables': tools.get_simple_stats(stat),
                    'tracers': simple_tracers,
                }
                file_kw = base_kw | {'tracer': file_tracer} | stats[stat]
                yield (stat, tracer), labels, file_kw

    # ------------------------------------------------------------------
    # Helper: read and assemble an ObservableTree (or WindowMatrix)
    # ------------------------------------------------------------------
    def read_observables(stats, stat_to_kind=None, **kwargs):
        """
        Read all files for the given stat/tracer combinations and assemble
        them into a single ObservableTree or WindowMatrix.

        Returns None if no files exist.
        """
        loaded, joint_labels, missing = [], {'observables': [], 'tracers': []}, []

        if stat_to_kind is None:
            stat_to_kind = lambda stat: stat

        for key, labels, kw in iter_stat_tracer_combos(stats, **kwargs):
            fn = tools.get_stats_fn(kind=stat_to_kind(key[0]), **kw)
            if fn.exists():
                loaded.append(types.read(fn))
                for field, value in labels.items():
                    joint_labels[field].append(value)
            else:
                missing.append(key)

        if not loaded:
            return None

        assert not missing, (
            f'Some measurements are missing and cannot be combined: {missing}'
        )

        # If the first item is a WindowMatrix, build a block-diagonal joint matrix.
        if isinstance(loaded[0], types.WindowMatrix):
            if len(loaded) == 1:
                return loaded[0]
            return _combine_window_matrices(loaded, joint_labels)
        else:
            return types.ObservableTree(loaded, **joint_labels)

    def _combine_window_matrices(windows, labels):
        """Stack a list of WindowMatrix objects into a single block-diagonal one."""
        values = [w.value() for w in windows]
        observables = [w.observable for w in windows]
        theories = [w.theory for w in windows]
        return types.WindowMatrix(
            value=linalg.block_diag(*values),
            observable=types.ObservableTree(observables, **labels),
            theory=types.ObservableTree(theories, **labels),
        )

    def _apply_stat_cuts(observable_tree):
        """Apply requested k/multipole selections per statistic to an ObservableTree."""
        defaults = {
            'mesh2_spectrum': {
                'ells': [0, 2],
                'kmin': 0.02,
                'kmax': 0.20,
                'rebin': 5,
            },
            'mesh3_spectrum': {
                'ells': [(0, 0, 0), (2, 0, 2)],
                'kmin': 0.02,
                'kmax_b0': 0.12,
                'kmax_b2': 0.08,
            },
        }

        user_cuts = cuts_kwargs or {}
        ps_cuts = defaults['mesh2_spectrum'] | user_cuts.get('mesh2_spectrum', {})
        bs_cuts = defaults['mesh3_spectrum'] | user_cuts.get('mesh3_spectrum', {})

        trimmed = observable_tree
        for key, at, _ in iter_stat_tracer_combos(stats):
            stat = key[0]
            branch = trimmed.get(**at)

            if stat == 'mesh2_spectrum':
                if ps_cuts.get('ells') is not None:
                    branch = branch.get(ells=list(ps_cuts['ells']))
                if ps_cuts.get('rebin') is not None and ps_cuts['rebin'] > 1:
                    branch = branch.select(k=slice(0, None, ps_cuts['rebin']))
                if ps_cuts.get('kmin') is not None and ps_cuts.get('kmax') is not None:
                    branch = branch.select(k=(ps_cuts['kmin'], ps_cuts['kmax']))

            elif stat == 'mesh3_spectrum':
                if bs_cuts.get('ells') is not None:
                    branch = branch.get(ells=list(bs_cuts['ells']))
                if bs_cuts.get('kmin') is not None and bs_cuts.get('kmax_b0') is not None:
                    branch = branch.select(k=(bs_cuts['kmin'], bs_cuts['kmax_b0']))
                if bs_cuts.get('rebin') is not None and bs_cuts['rebin'] > 1:
                    branch = branch.select(k=slice(0, None, bs_cuts['rebin']))
                if (2, 0, 2) in list(bs_cuts.get('ells', [])) and bs_cuts.get('kmin') is not None and bs_cuts.get('kmax_b2') is not None:
                    branch = branch.at(ells=(2, 0, 2)).select(k=(bs_cuts['kmin'], bs_cuts['kmax_b2']))

            trimmed = trimmed.at(**at).match(branch)

        return trimmed

    # ------------------------------------------------------------------
    # Step 1 — Data vector: average over 25 Abacus mocks
    # ------------------------------------------------------------------
    if 'abacus' not in data:
        raise NotImplementedError(f'data={data!r} is not supported')

    data_kwargs = dict(
        version=data,
        stats_dir=tools.desi_dir / 'mocks/cai/LSS/DA2/mocks/desipipe',
    )

    mock_observables = [
        obs
        for imock in range(25)
        if (obs := read_observables(stats, imock=imock, **data_kwargs)) is not None
    ]
    data_vector = types.mean(mock_observables)
    data_vector = _apply_stat_cuts(data_vector)

    # ------------------------------------------------------------------
    # Step 2 — Window matrix: read from mock 0
    # ------------------------------------------------------------------
    window_matrix = read_observables(
        stats,
        imock=0,
        stat_to_kind=lambda stat: f'window_{stat}',
        **data_kwargs,
    )

    # ------------------------------------------------------------------
    # Step 3 — Apply window-function rotation (optional)
    # ------------------------------------------------------------------
    # Each entry in applied_rotations is (at, rotation_obj) so we can
    # apply the same rotation to the covariance matrix later.
    applied_rotations = []

    if rotation:
        prior_matrices = []  # only populated when NOT using analytic marginalization

        for key, at, kw in iter_stat_tracer_combos(stats, imock=0, **data_kwargs):
            stat = key[0]
            if 'mesh2_spectrum' not in stat:
                continue

            # Load the pre-computed rotation for this statistic.
            rotation_fn = tools.get_stats_fn(kind='rotation_mesh2_spectrum', **kw)
            rotation_obj = types.read(rotation_fn)
            applied_rotations.append((at, rotation_obj))

            # Align data vector and window matrix to the rotation's observable grid,
            # then apply the rotation in place.
            data_vector = data_vector.match(rotation_obj.observable)
            window_matrix = _match_observable(window_matrix, data_vector)
            window_matrix, data_vector = rotation_obj.rotate(
                window=window_matrix,
                data=data_vector,
                at=at,
                prior_data=True,
            )

            # If not using analytic marginalization, collect the prior WindowMatrix
            # so it can be appended to the window matrix as extra columns.
            if not use_analytic_marg:
                prior_matrices.append(rotation_obj.prior(at=at))

        # Append marginalization priors to the window matrix as extra theory columns.
        if prior_matrices:
            window_matrix = _append_prior_columns(
                window_matrix, prior_matrices, applied_rotations
            )

    # ------------------------------------------------------------------
    # Step 4 — Covariance matrix: estimated from 1000 holi mocks
    # ------------------------------------------------------------------
    if 'holi' not in covariance:
        raise NotImplementedError(f'covariance={covariance!r} is not supported')

    cov_kwargs = dict(
        version=covariance,
        stats_dir=tools.desi_dir / 'mocks/cai/LSS/DA2/mocks/desipipe',
    )

    cov_observables = [
        obs
        for imock in range(1000)
        if (obs := read_observables(stats, imock=imock, **cov_kwargs)) is not None
    ]
    covariance_matrix = _match_observable(types.cov(cov_observables), data_vector)
    covariance_matrix.attrs['nobs'] = len(cov_observables)

    # Apply the same rotations to the covariance matrix.
    if rotation:
        for at, rotation_obj in applied_rotations:
            covariance_matrix = rotation_obj.rotate(
                covariance=covariance_matrix,
                at=at,
                prior_cov=use_analytic_marg,
            )

    # ------------------------------------------------------------------
    # Step 5 — Final alignment and build the likelihood
    # ------------------------------------------------------------------
    # Re-align window and covariance to the (possibly rotated) data vector.
    window_matrix = _match_observable(window_matrix, data_vector)
    covariance_matrix = _match_observable(covariance_matrix, data_vector)

    return types.GaussianLikelihood(
        observable=data_vector,
        window=window_matrix,
        covariance=covariance_matrix,
    )


def _append_prior_columns(window_matrix, prior_matrices, applied_rotations):
    """
    Append marginalization-prior columns to a WindowMatrix.

    Each prior is an additional WindowMatrix whose theory columns represent
    the k-templates to be marginalised over. They are stacked alongside the
    existing theory columns with a 'marg' suffix on the observables label.

    Parameters
    ----------
    window_matrix : types.WindowMatrix
        Base window matrix to extend.
    prior_matrices : list[types.WindowMatrix]
        One prior matrix per applied rotation.
    applied_rotations : list[tuple]
        List of (at, rotation_obj) matching prior_matrices in order.

    Returns
    -------
    types.WindowMatrix
        New window matrix with prior columns appended.
    """
    values = [window_matrix.value()]
    theories = [window_matrix.theory]

    for prior, (at, _) in zip(prior_matrices, applied_rotations):
        values.append(prior.value())

        # Build an ObservableTree for this prior's theory with a 'marg' suffix.
        marg_labels = dict(at)
        marg_labels['observables'] = marg_labels['observables'] + 'marg'
        theories.append(types.ObservableTree([prior.theory], **marg_labels))

    combined_value = np.concatenate(values, axis=-1)
    return window_matrix.clone(value=combined_value, theory=theories)
