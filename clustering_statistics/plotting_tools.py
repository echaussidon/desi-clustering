from matplotlib import pyplot as plt

import lsstypes as types
from clustering_statistics import tools


def get_means_covs(kind, versions, tracer, zrange, region, stats_dir, rebin=1):
    means, covs = {}, {}
    for version in versions:
        kw = {'tracer': tracer}
        for name in ['version', 'weight', 'cut', 'auw', 'extra']:
            kw[name] = versions[version][name]
        if 'ELG' in kw['tracer']:
            if 'complete' in kw['version']:
                kw['tracer'] = 'ELG_LOP'
            elif 'data' in kw['version']:
                kw['tracer'] = 'ELG_LOPnotqso'
        if 'sugiyama-diagonal' in kind:
            kw['basis'] = 'sugiyama-diagonal'
            kw['auw'] = False
        fns = tools.get_stats_fn(kind=kind, stats_dir=stats_dir, zrange=zrange, region=region, **kw, imock='*')
        stats = list(map(types.read, fns))
        means[version] = types.mean(stats).select(k=slice(0, None, rebin))
        if len(stats) > 1:
            covs[version] = types.cov(stats).at.observable.match(means[version])
        else:
            covs[version] = None
    return means, covs


def plot_stats(kind, versions, tracer, zrange, region, stats_dir, ells=(0,2,4), rebin=1, reference=None, ylim=(-1.5, 1.5),
               figure=None, ax_col=0, linestyles=None, colors=None, scaling='kpk', save_fn=None):
    if reference is None:
        # use first item from versions as reference
        reference = next(iter(versions))
    if linestyles is None: linestyles = dict(zip(versions, ['-']*len(versions)))
    if colors is None: colors = dict(zip(versions, [f'C{i:d}' for i in range(len(versions))]))
    if figure is None:
        fig, lax = plt.subplots(len(ells) * 2, figsize=(6, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1] * len(ells)})
    else:
        fig, axes = figure
        if axes.ndim == 1:
            lax = axes
        else:
            lax = axes[:, ax_col]
    k_exp = 1 if scaling == 'kpk' else 0
    if 'mesh2_spectrum' in kind:
        means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, rebin=rebin)
        versions = list(means)
        lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^2$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^3$]')
                ax.set_yscale('log')
                ax.set_xscale('log')
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                ax.plot(pole.coords('k'), pole.coords('k')**k_exp * pole.value().real, color=colors[version], linestyle=linestyles[version], label=version)
            if ill == 0: ax.legend(frameon=False, ncol=1)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta P_{ell:d} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                # std = covs[reference].at.observable.get(ell).std().real
                std = covs[reference].at.observable.get(ell).std().real
                ax.plot(pole.coords('k'), (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version])
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    elif 'mesh3_spectrum_sugiyama-diagonal' in kind:
        means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, rebin=rebin)
        versions = list(means)
        lax[0].set_title(f'{tracer} in {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k^2 B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^6$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^4$]')
                ax.set_yscale('log')
                ax.set_xscale('log')

            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                x = pole.coords('k')[..., 0]
                ax.plot(x, (x**2)**k_exp * pole.value().real, color=colors[version], linestyle=linestyles[version], label=version)
                if ill == 0: ax.legend(frameon=False, ncol=2)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                std = covs[reference].at.observable.get(ell).std().real
                x = pole.coords('k')[..., 0]
                ax.plot(x, (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version])
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    if save_fn and figure is None:
        plt.tight_layout()
        fig.savefig(save_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.show()