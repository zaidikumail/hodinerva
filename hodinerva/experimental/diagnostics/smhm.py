import os

import matplotlib.pyplot as plt
import numpy as np

from ..kernels.smhm import HALO_MASS_H0P7

plt.rc("font", family="serif", serif=["Times New Roman"])
colors_z = [
    "xkcd:lavender pink",
    "xkcd:cerulean",
    "xkcd:emerald green",
    "xkcd:avocado",
    "xkcd:peach",
    "xkcd:red pink",
]


def plot_med_smhm(med_cen_smhm_dir, run_label, zbins, cosmo, savedir):
    med_cen_mstar_med = np.load(
        f"{med_cen_smhm_dir}/{run_label}/{run_label}_med_cen_mstar_med.npy"
    )
    med_cen_mstar_low = np.load(
        f"{med_cen_smhm_dir}/{run_label}/{run_label}_med_cen_mstar_low.npy"
    )
    med_cen_mstar_upp = np.load(
        f"{med_cen_smhm_dir}/{run_label}/{run_label}_med_cen_mstar_upp.npy"
    )

    fs = 12
    figsize = (7.1, 4.0)
    margins = dict(hspace=0, wspace=0, left=0.11, right=0.89, bottom=0.1, top=0.98)

    # SHMR
    fig_SHMR, axs = plt.subplots(2, 3, figsize=figsize, gridspec_kw=margins)
    ax_SHMR = axs.ravel()

    # smhm
    fig_smhm, axs = plt.subplots(2, 3, figsize=figsize, gridspec_kw=margins)
    ax_smhm = axs.ravel()

    for zbin in range(1):
        if zbin == 0:
            mean_cenSHMR_med_z1 = med_cen_mstar_med[zbin] / HALO_MASS_H0P7
            med_cen_mstar_med_z1 = med_cen_mstar_med[zbin]
        else:
            ax_smhm[zbin].plot(
                HALO_MASS_H0P7, med_cen_mstar_med_z1, c=colors_z[0], alpha=1, lw=2
            )
            ax_SHMR[zbin].plot(
                HALO_MASS_H0P7, mean_cenSHMR_med_z1, c=colors_z[0], alpha=1, lw=2
            )

        ax_SHMR[zbin].plot(
            HALO_MASS_H0P7,
            med_cen_mstar_med[zbin] / HALO_MASS_H0P7,
            c=colors_z[zbin],
            alpha=1,
            lw=2,
            label=str(zbins[zbin][0]) + " < z < " + str(zbins[zbin][1]),
        )
        ax_SHMR[zbin].fill_between(
            HALO_MASS_H0P7,
            med_cen_mstar_low[zbin] / HALO_MASS_H0P7,
            med_cen_mstar_upp[zbin] / HALO_MASS_H0P7,
            color=colors_z[zbin],
            alpha=0.25,
        )

        ax_smhm[zbin].plot(
            HALO_MASS_H0P7,
            med_cen_mstar_med[zbin],
            c=colors_z[zbin],
            alpha=1,
            lw=2,
            label=str(zbins[zbin][0]) + " < z < " + str(zbins[zbin][1]),
        )
        ax_smhm[zbin].fill_between(
            HALO_MASS_H0P7,
            med_cen_mstar_low[zbin],
            med_cen_mstar_upp[zbin],
            color=colors_z[zbin],
            alpha=0.25,
        )

        ax_SHMR[zbin].legend()
        ax_smhm[zbin].legend()

    """SHMR"""
    # SFE
    yticks = np.logspace(-1, 2, 4)
    yticklabels = ["10$^{-1}$", "10$^{0}$", "10$^{1}$", "10$^{2}$"]
    sfe = (
        lambda y: (y / (cosmo.Ob0 / cosmo.Om0)) * 100,
        lambda y: y * (cosmo.Ob0 / cosmo.Om0) / 100,
    )
    for i in [2, 5]:
        ax_dual = ax_SHMR[i].secondary_yaxis("right", functions=sfe)
        ax_dual.set_yticks(yticks)
        ax_dual.set_yticklabels(yticklabels, fontsize=fs)
        ax_dual.tick_params(direction="in")
        if i == 5:
            ax_dual.set_ylabel("SFE x 100 [%]", fontsize=fs)

    for zbin in range(len(zbins)):
        ax_SHMR[zbin].set(
            xscale="log",
            yscale="log",
            xlim=(1e11, 1e15),
            ylim=(1e-4, 0.3),
            yticklabels=[],
            xticklabels=[],
        )
        ax_SHMR[zbin].tick_params(
            which="both", direction="in", top=True, right=True, labelsize=fs
        )

    # SHMR
    ax_SHMR[3].set_ylabel("M$_{\u2217}$ / M$_{h}$", fontsize=fs)
    ax_SHMR[4].set_xlabel("M$_{h}$ [M$_{\u2609}$]", fontsize=fs)

    # SHMR
    yticks = np.logspace(-4, -1, 4)
    yticklabels = ["", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$"]
    for i in [0, 3]:
        ax_SHMR[i].set_yticks(yticks)
        ax_SHMR[i].set_yticklabels(yticklabels, fontsize=fs)

    # SHMR
    xticks = np.logspace(12, 14, 2)
    xticklabels = ["10$^{12}$", "10$^{14}$"]
    for i in [3, 4, 5]:
        ax_SHMR[i].set_xticks(xticks)
        ax_SHMR[i].set_xticklabels(xticklabels, fontsize=fs)

    path = os.path.join(savedir, run_label)
    os.makedirs(path, exist_ok=True)

    fig_SHMR.savefig(path + "/med_smhm_ratio.png")

    ##################################################################################
    """smhm"""
    for zbin in range(0, len(zbins)):
        ax_smhm[zbin].set(
            xscale="log",
            yscale="log",
            xlim=(1e11, 1e15),
            ylim=(1e7, 1e13),
            yticklabels=[],
            xticklabels=[],
        )
        ax_smhm[zbin].tick_params(
            which="both", direction="in", top=True, right=True, labelsize=fs
        )

    # SHMR
    ax_smhm[3].set_ylabel("M$_{\u2217}$ [M$_{\u2609}$]", fontsize=fs)
    ax_smhm[4].set_xlabel("M$_{h}$ [M$_{\u2609}$]", fontsize=fs)

    # SHMR
    yticks = np.logspace(7, 13, 7)
    yticklabels = [
        "10$^{7}$",
        "10$^{8}$",
        "10$^{9}$",
        "10$^{10}$",
        "10$^{11}$",
        "10$^{12}$",
        "",
    ]
    for i in [0, 3]:
        ax_smhm[i].set_yticks(yticks)
        ax_smhm[i].set_yticklabels(yticklabels, fontsize=fs)

    # SHMR
    xticks = np.logspace(12, 14, 2)
    xticklabels = ["10$^{12}$", "10$^{14}$"]
    for i in [3, 4, 5]:
        ax_smhm[i].set_xticks(xticks)
        ax_smhm[i].set_xticklabels(xticklabels, fontsize=fs)

    fig_smhm.savefig(path + "/med_smhm.png")

    plt.show()
