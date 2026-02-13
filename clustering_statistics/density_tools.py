import os
import itertools

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
import jax
from jax import numpy as jnp

from . import tools


@jax.jit
def _compute_healpix_map_jax(ra, dec, weights=1., nside: int=256, nest: bool=False):
    import jax_healpy as hp

    hpmap = jnp.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    return hpmap.at[pix].add(weights)


def _compute_healpix_map_numpy(ra, dec, weights=1., nside: int=256, nest: bool=False):
    hpmap = np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    np.add.at(hpmap, pix, weights)
    return hpmap


def compute_angular_density(catalog, nside: int=256, nest: bool=False, backend: str='numpy'):
    """
    Compute an angular (HEALPix) density map from a catalog.

    The catalog may already contain 'RA' and 'DEC' fields; otherwise RA/DEC are
    derived from the 'POSITION' field via tools.cartesian_to_sky. When using the
    'jax' backend the computation happens on JAX arrays with optional sharding.

    Parameters
    ----------
    catalog : Catalog
        Catalog.
    nside : int, optional
        HEALPix nside resolution (default 256).
    nest : bool, optional
        Use NESTED pixel ordering if True (default False).
    backend : {'numpy', 'jax'}, optional
        Backend to use for HEALPix computation. 'numpy' uses NumPy+MPI, 'jax' uses JAX.

    Returns
    -------
    array-like
        HEALPix map containing the weighted counts per pixel.
    """
    if 'RA' not in catalog:
        catalog['RA'], catalog['DEC'] = tools.cartesian_to_sky(catalog['POSITION'])
    ra, dec, weights = catalog['RA'], catalog['DEC'], catalog.get('INDWEIGHT', catalog.ones())
    if backend == 'jax':
        from jaxpower.mesh import create_sharding_mesh, make_array_from_process_local_data
        with create_sharding_mesh() as sharding_mesh:
            ra, dec = [make_array_from_process_local_data(rd, pad='uniform', sharding_mesh=sharding_mesh) for rd in [ra, dec]]
            weights = make_array_from_process_local_data(weights, pad=0., sharding_mesh=sharding_mesh)
            hpmap = _compute_healpix_map_jax(ra, dec, weights=weights, nside=nside, nest=nest)
    else:
        # backend = 'numpy'
        hpmap = _compute_healpix_map_numpy(ra, dec, weights=weights, nside=nside, nest=nest)
        mpicomm = catalog.mpicomm
        hpmap = mpicomm.allreduce(hpmap)
    return hpmap


def compute_histogram1d(catalog, name='Z', edges=None, backend: str='numpy'):
    """
    Compute a weighted redshift histogram for a catalog.

    Parameters
    ----------
    catalog : Catalog
        Catalog.
    name : str
        Column name to bin along.
    edges : sequence or int or None, optional
        Bin edges for the histogram. If an integer is provided, that number of
        equal-width bins between min(z) and max(z) is constructed.
    backend : {'numpy', 'jax'}, optional
        Backend to use for histogramming. 'numpy' uses NumPy+MPI, 'jax' uses JAX.

    Returns
    -------
    hist, edges
    """
    z, weights = catalog[name], catalog.get('INDWEIGHT', catalog.ones())
    if isinstance(edges, int):
        import mpytools as mpy
        cmin, cmax = mpy.cmin(z), mpy.cmax(z)
        edges = np.linspace(cmin, cmax, edges + 1)
    if backend == 'jax':
        from jaxpower.mesh import create_sharding_mesh, make_array_from_process_local_data
        with create_sharding_mesh() as sharding_mesh:
            z = make_array_from_process_local_data(z, pad='uniform', sharding_mesh=sharding_mesh)
            weights = make_array_from_process_local_data(weights, pad=0., sharding_mesh=sharding_mesh)
            hist = jnp.histogram(z, bins=edges, weights=weights)
    else:
        hist = np.histogram(z, bins=edges, weights=weights)[0]
        mpicomm = catalog.mpicomm
        hist = mpicomm.allreduce(hist)
    return hist, edges


def plot_density_projections(get_catalog_fn=tools.get_catalog_fn, read_catalog=tools.read_clustering_catalog,
                             catalog=dict(), divide_randoms: bool | str=False, backend: str='numpy', edges=None,
                             nside=256, map_q=False, fn=None, **kwargs):
    """
    Plot angular density projections (HEALPix) and optional redshift distributions.

    This convenience function reads one or more catalogs determined by combinations
    of keyword grid `kwargs` and plots the averaged HEALPix maps and, if requested,
    the redshift histogram. Optionally divides data by randoms to produce overdensity.

    Parameters
    ----------
    get_catalog_fn : callable, optional
        Function returning a path or handle given kind='data'/'randoms' and keyword args.
    read_catalog : callable, optional
        Function that reads a catalog given a path.
    catalog : dict, optional
        Base keyword args passed to get_catalog_fn / read_catalog.
    divide_randoms : bool or 'same', optional
        If True divide data maps by randoms maps.
        Pass 'same' to reuse the same randoms.
    backend : {'numpy', 'jax'}, optional
        Backend to use for map/histogram computation.
    edges : dict of sequence, optional
        If provided, compute and plot histograms using these edges.
    nside : int, optional
        HEALPix nside resolution for angular maps.
    fn : str or None, optional
        If provided, save the figure to this filename.
    **kwargs :
        Keyword grid to iterate: keys map to parameter names of :func:`get_catalog_fn` and values to sequences.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the HEALPix projection (and z-histogram if requested).
    """
    nest = False
    randoms_hpmap = randoms_hist = None
    hpmaps = []
    edges = edges or {}
    hist_names = list(edges.keys())
    hists = {name: [] for name in hist_names}
    rank = 0
    names, values = zip(*kwargs.items())
    for values in itertools.product(*values):
        fn_kwargs = catalog | dict(get_catalog_fn=get_catalog_fn) | dict(zip(names, values))
        data = read_catalog(kind='data', **fn_kwargs)
        rank = data.mpicomm.rank
        data_hpmap = compute_angular_density(data, nside=nside, nest=nest, backend=backend)
        data_hist = {}
        for name in hist_names:
            data_hist[name] = compute_histogram1d(data, name=name, edges=edges[name], backend=backend)[0]
        if divide_randoms:
            if not (divide_randoms == 'same' and randoms_hpmap is not None):
                randoms = read_catalog(kind='randoms', **fn_kwargs)
                randoms_hpmap = compute_angular_density(randoms, nside=nside, nest=nest, backend=backend)
                randoms_hist = {}
                for name in hist_names:
                    randoms_hist[name] = compute_histogram1d(randoms, name=name, edges=edges[name], backend=backend)[0]
                del randoms
            data_hpmap = data_hpmap / randoms_hpmap * randoms_hpmap.sum() / data_hpmap.sum()
            for name in hist_names:
                data_hist[name] = data_hist[name] / randoms_hist[name] * randoms_hist[name].sum() / data_hist[name].sum()
        hpmaps.append(data_hpmap)
        for name in hists:
            hists[name].append(data_hist[name])

    # --- Layout decisions ---
    with_bottom = len(hist_names) > 0
    nrows = 1 + int(with_bottom)
    ncols = 1
    height_ratios = [4] + ([2] if with_bottom else [])
    fig = plt.figure(figsize=(10, 4 + 3 * int(with_bottom)))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios
    )
    from scipy.stats import chi2 as chi2_dist

    def _annotate_chi2(ax, chi2, ndof, *, loc="upper left"):
        """Write reduced chi2 and p-value on an axes."""
        if ndof <= 0 or not np.isfinite(chi2):
            return
        rchi2 = chi2 / ndof
        pval = chi2_dist.sf(chi2, ndof)  # upper-tail p-value
        text = rf"$\chi^2_\nu={rchi2:.2f}$" + "\n" + rf"$p={pval:.3g}$"
        # place in axes coordinates
        x = 0.02 if "left" in loc else 0.98
        y = 0.98 if "upper" in loc else 0.02
        ha = "left" if "left" in loc else "right"
        va = "top" if "upper" in loc else "bottom"
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))

    # --- Top row: Mollweide map ---
    ax_map = fig.add_subplot(gs[0, 0])
    plt.sca(ax_map)
    mean = np.mean(hpmaps, axis=0)
    mask = np.isfinite(mean)
    if divide_randoms:
        err = np.std(hpmaps, axis=0, ddof=0) / np.sqrt(len(hpmaps))
        residuals = (mean - 1.) / err
        chi2 = np.sum(residuals[mask]**2)
    if map_q:
        limits = np.quantile(residuals[mask], q=map_q)
        mean = residuals = np.clip(residuals, *limits)
    hp.mollview(mean, hold=True, cbar=True, nest=nest)
    if divide_randoms:
        _annotate_chi2(ax_map, chi2, mask.sum() - 1, loc="upper left")
    # --- Bottom row: 1D histograms ---
    if with_bottom:
        sub = gs[1, 0].subgridspec(1, len(hist_names), wspace=0.35)
        for iname, name in enumerate(hist_names):
            ax = fig.add_subplot(sub[0, iname])
            e = np.asarray(edges[name])
            x = 0.5 * (e[:-1] + e[1:])
            mean = np.mean(hists[name], axis=0)
            mask = np.isfinite(mean)
            err = np.std(hists[name], axis=0, ddof=0) / np.sqrt(len(hists[name]))
            # step-style histogram
            ax.set_xlabel(name)
            if divide_randoms:
                residuals = (mean - 1.) / err
                chi2 = np.sum(residuals[mask]**2)
                ax.plot(x, residuals, color='k')
                ax.grid(True)
                if iname == 0:
                    ax.set_ylabel(r"$(\mathrm{mean} - 1) / \sigma$")
                _annotate_chi2(ax, chi2, mask.sum() - 1, loc="upper left")
            else:
                ax.plot(x, mean, color='k')
                ax.fill_between(x, mean - std, mean + std, color='k', alpha=0.2)
                if iname == 0:
                    ax.set_ylabel('counts')

    if fn is not None:
        plt.tight_layout()
        tools.mkdir(os.path.dirname(fn))
        if rank == 0: plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close(plt.gcf())
    return fig