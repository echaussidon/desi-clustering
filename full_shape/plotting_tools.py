
import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import numpy as np
from getdist import plots
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from desilike.samples import plotting, Chain

sys.path.append('../')

PLANCK_COSMOLOGY = {
    "Omega_m": 0.315191868,
    "H_0": 67.36,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "h": 0.6736,
    "A_s": 2.083e-9,
    "logA": 3.034,
    "n_s": 0.9649,
    "N_ur": 2.0328,
    "N_ncdm": 1.0,
    "omega_ncdm": 0.0006442,
    "w_0": -1,
    "w_a": 0.0
}

##### Color settings #####
COLOR_OVERALL = dict(BGS = 'yellowgreen',
                    LRG = 'red',
                    ELG = 'blue',
                    QSO = 'darkgreen')

COLOR_TRACERS = dict(BGS1='yellowgreen', 
                    LRG1='orange', LRG2='orangered', LRG3='firebrick',
                    ELG1='skyblue', ELG2= 'steelblue',
                    QSO1='purple')

COLOR_TRACER_GRADIENT = dict(BGS1='Greens', 
                    LRG1='Reds', LRG2='Reds', LRG3='Reds',
                    ELG1='Blues', ELG2= 'Blues',
                    QSO1='Purples')

##### Functions #####

def get_namespace(tracer, zrange):
    return {
        ('BGS_BRIGHT-21.35', (0.1, 0.4)): 'BGS1',
        ('BGS', (0.1, 0.4)): 'BGS1',
        ('LRG', (0.4, 0.6)): 'LRG1',
        ('LRG', (0.6, 0.8)): 'LRG2',
        ('LRG', (0.8, 1.1)): 'LRG3',
        ('ELG_LOPnotqso', (0.8, 1.1)): 'ELG1',
        ('ELG_LOPnotqso', (1.1, 1.6)): 'ELG2',
        ('ELG', (0.8, 1.1)): 'ELG1',
        ('ELG', (1.1, 1.6)): 'ELG2',
        ('QSO', (0.8, 2.1)): 'QSO1',
    }[(tracer, zrange)]

def get_mcmc_plot_args(task, params=None):
    """
    Return the plot setting for mcmc.
    """
    if 'SF' in task:
        if params==None: 
            params = ['qiso', 'qap', 'dm', 'df']
            params_label = [r'$\alpha_{\mathrm{iso}}$', r'$\alpha_{\mathrm{ap}}$', r'$dm$', r'$df$']
        true_values = dict(qiso=1, qap=1, dm=0, df=1)
        remove_burnin = 0.5
        slice_step = 5500
        fig_width = 10
        legend_fontsize = 16
        axes_labelsize = 16
        if 'all' in task: 
            params+= ['b1p', 'b2p', 'bsp', 'alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
            fig_width = 18
    elif 'FM' in task:
        if params==None: 
            params = ['h', 'omega_cdm', 'logA']
            params_label = [r'$h$', r'$\omega_{\mathrm{cdm}}$', r'$\ln(10^{10} A_s)$']
        true_values = {param: PLANCK_COSMOLOGY[param] for param in params}
        remove_burnin = 0.5
        slice_step = 5500
        fig_width = 8
        legend_fontsize = 16
        axes_labelsize = 16
    elif 'BAO' in task:
        if params==None: params = ['qiso', 'qap']
        true_values = dict(qiso=1, qap=1)
        remove_burnin = 0.5
        slice_step = 600
        fig_width = 5
        legend_fontsize = 14
        axes_labelsize  = 16
    mcmc_args = dict(params=params, params_label=params_label, true_values=true_values, remove_burnin=remove_burnin, slice_step=slice_step)
    getdist_args = dict(fig_width=fig_width, legend_fontsize=legend_fontsize, axes_labelsize=axes_labelsize)
    return (mcmc_args, getdist_args)


def plot_observable(self, ax_top=None, ax_bottom=None, **plot_kwargs):
    """
    Plot the observable into provided axes
    """
    show_legend = True
    corr_type = plot_kwargs.get('corr_type', 'pk')
    color = plot_kwargs.get('color', f'C0')
    label = plot_kwargs.get('label', None)
    fmt = plot_kwargs.get('fmt', 'o')
    # (tracer, i, plot_sysmodel) = (plot_kwargs[key] for key in ["tracer", "index", "sys_model"])
    data, theory, std = self.data, self.theory, self.std
    if corr_type == 'xi':
        # Plot the observable (top panel)
        for ill, ell in enumerate(self.ells):
            ax_top.errorbar(self.s[ill], self.s[ill]**2 * data[ill], yerr=self.s[ill]**2 * std[ill],
                            color=color, linestyle='none', marker='o')
            ax_top.plot(self.s[ill], self.s[ill]**2 * theory[ill], color = color, ls = linestyle, label = None)
        ax_top.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if show_legend:
            ax_top.legend()
        ax_top.grid(True)
        # Plot the residuals (bottom panel)
        for ill, ell in enumerate(self.ells):
            ax_bottom.plot(self.s[ill], (data[ill] - theory[ill]) / std[ill], color = color, ls = linestyle)
            ax_bottom.set_ylim(-4, 4)
            for offset in [-2., 2.]:
                ax_bottom.axhline(offset, color='k', linestyle='--')
            ax_bottom.set_ylabel(r'$\Delta \xi_{{{0:d}}} / \sigma_{{ \xi_{{{0:d}}} }}$'.format(ell))
        ax_bottom.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        ax_bottom.grid(True)
    if corr_type == 'pk':
        # Plot the observable (top panel)
        for ill, ell in enumerate(self.ells):
            if ill == 1: label=None
            ax_top.errorbar(self.k[ill], self.k[ill] * data[ill], yerr=self.k[ill] * std[ill],
                            color=color, fmt = fmt, label = label, markersize= 4)
            ax_top.plot(self.k[ill], self.k[ill] * theory[ill], color = color)
        if show_legend:
            ax_top.legend(loc=1)
        ax_top.set_ylabel(r'$k P_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{-1}$]', fontsize=12)
        # Plot the residuals (bottom panel)
        for ill, ell in enumerate(self.ells):
            ax_bottom[ill].plot(self.k[ill], (data[ill] - theory[ill]) / std[ill], color = color)
            ax_bottom[ill].set_ylim(-3.8, 3.8)
            for offset in [-2., 2.]:
                ax_bottom[ill].axhline(offset, color='k', linestyle='--')
            ax_bottom[ill].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell), fontsize=12)
        ax_bottom[1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]', fontsize=12)
        #settings
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_bottom[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def plot_observable_bao(self, ax_top=None, ax_bottom=None, **plot_kwargs):
    """
    Plot data and theory BAO correlation function peak.
    """
    lax = [ax_top, ax_bottom]
    if ax_bottom == None:
        lax = [ax_top]
    show_legend = False
    color = plot_kwargs.get('color', f'C0')
    linestyle = plot_kwargs.get('linestyle', '-')
    fmt = plot_kwargs.get('fmt', 'o')
    data, theory, std = self.data, self.theory, self.std
    nobao = self.theory_nobao
    for ill, ell in enumerate(self.ells):
            # lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill] - nobao[ill]), yerr=self.s[ill]**2 * std[ill], 
            #                 color=color, fmt = fmt, markersize= 6, label = 'std')
        lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill]), yerr=self.s[ill]**2 * std[ill], 
                        color=color, fmt = fmt, markersize= 6, label = 'std')
        lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill]), yerr=self.s[ill]**2 * std[ill], 
                        color=color, fmt = fmt, markersize= 6, markerfacecolor='none', label = 'dv')
        lax[ill].plot(self.s[ill], self.s[ill]**2 * (theory[ill]), 
                      color=color, linestyle = linestyle)
        lax[ill].set_ylabel(r'$s^{{2}} \Delta \xi_{{{:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        if ill == 0:
            lax[ill].legend(loc =3, ncol=2)
        # if 2 not in self.ells:
            # lax[ill].legend(loc =3, ncol=2)
    for ax in lax: ax.grid(True)
    lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')

def plot_mcmc_walkers(chain, params, nwalkers = 64, true_values = None):
    ndim            = len(params)
    chain_samples   = dict(zip(chain.basenames(), chain.data))
    samples         = np.array([chain_samples[p] for p in params])
    medians         = np.array(chain.median(params=params))
    # true_values     = set_true_values(params)
    fig, ax = plt.subplots(ndim, sharex=True, figsize=(16, 2 * ndim))
    for i in range(nwalkers):
        for j in range(ndim):
            ax[j].plot(samples[j, :, i], c = 'green', lw=0.3)
            ax[j].set_ylabel(params[j], fontsize=15)
            ax[j].grid(True)
            ax[j].axhline(medians[j], c='blue', lw=1.2)
            if true_values != None:
                ax[j].axhline(true_values[j], c='red', lw=1.2)

def convert_chain(chain):
    chain.set(chain['Omega_m'].clone(value=(chain['omega_cdm'] + chain['omega_b'] + PLANCK_COSMOLOGY['omega_ncdm'])/chain['h']**2, param={'basename': 'Omega_m', 'derived': True, 'latex': r'\Omega_m'}))
    chain.set(chain['H0'].clone(value=(chain['h']*100), param={'basename': 'H0', 'derived': True, 'latex': r'H_0'}))
    return 0

def read_bao_chain(filename, burnin=0.5, slice_step=1, apmode='qisoqap'):
    if isinstance(filename, list):
        chains = []
        for fn in filename:
            chains.append(Chain.load(fn))
        chain = chains[0].concatenate([chain.remove_burnin(burnin)[::slice_step] for chain in chains])
    else:
        chain = Chain.load(filename)
        chain = chain.remove_burnin(burnin)[::slice_step]
    if apmode == 'qparqper':
        qiso = (chain['qpar']**(1./3.) * chain['qper']**(2./3.)).clone(param=dict(basename='qiso', derived=True, latex=r'q_{\rm iso}'))
        qap = (chain['qpar'] / chain['qper']).clone(param=dict(basename='qap', derived=True, latex=r'q_{\rm AP}'))
        chain.set(qiso)
        chain.set(qap)
    if apmode == 'qisoqap':
        qpar = (chain['qiso'] * chain['qap']**(2/3)).clone(param=dict(basename='qpar', derived=True, latex=r'q_{\parallel}'))
        qper = (chain['qiso'] * chain['qap']**(-1/3)).clone(param=dict(basename='qper', derived=True, latex=r'q_{\perp}'))
        chain.set(qpar)
        chain.set(qper)
    alpha_iso = chain['qiso'].clone(param=dict(basename='alpha_iso', derived=True, latex=r'(D_{\mathrm{V}}/r_{d})/(D_{\mathrm{V}}/r_{d})^{\rm fid}'))
    chain.set(alpha_iso)
    if apmode in ['qisoqap', 'qparqper']:
        alpha_ap = chain['qap'].clone(param=dict(basename='alpha_ap', derived=True, latex=r'(D_{\mathrm{H}}/D_{\mathrm{M}})/(D_{\mathrm{H}}/D_{\mathrm{M}})^{\rm fid}'))
        alpha_par = chain['qpar'].clone(param=dict(basename='alpha_par', derived=True, latex=r'(D_{\mathrm{H}}/r_{d})/(D_{\mathrm{H}}/r_{d})^{\rm fid}'))
        alpha_per = chain['qper'].clone(param=dict(basename='alpha_per', derived=True, latex=r'(D_{\mathrm{M}}/r_{d})/(D_{\mathrm{M}}/r_{d})^{\rm fid}'))
        chain.set(alpha_ap)
        chain.set(alpha_par)
        chain.set(alpha_per)
    return chain

def plot_DM():
    return 0

def plot_mcmc_contour(chain, params, plot_args=None):
    g = plots.get_subplot_plotter()
    g.settings.fig_width_inch= 8
    g.settings.legend_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.axes_fontsize = 16
    g.settings.figure_legend_frame = False
    plotting.plot_triangle(chain, title_limit=1, filled = True, params = params,
                            #    legend_labels = labels, legend_loc= 'upper right',
                                contour_lws = 1.5,
                                # contour_ls = lss, contour_lws = lws, contour_colors = colors, 
                                # param_limits=param_limits, 
                                smoothed=True, show=False, g=g)
    # true_values     = set_true_values(params)
    # for i in range(len(true_values)):
    #     for j in range(i+1):
    #         g.subplots[i,j].axvline(true_values[j], c = 'k', ls = ':', lw = 1.2)
    #         if i != j:
    #             g.subplots[i,j].axhline(true_values[i], c = 'k', ls = ':', lw = 1.2)


