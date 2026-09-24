import emcee
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate

from ..build_model import build_acf_model, get_model_smf
from ..defaults import DEFAULT_COSMOLOGY, HOD_INITIALIZE_PARAMS
from .plot_utils import scale_luminance

plt.rc("font", family="serif", serif=["Times New Roman"])
# colors = [
#     "xkcd:lavender pink",
#     "xkcd:cerulean",
#     "xkcd:aqua",
#     "xkcd:emerald green",
#     "xkcd:avocado",
#     "xkcd:peach",
#     "xkcd:red pink",
#     "xkcd:dark maroon",
#     "xkcd:mustard",
# ]

colors = [
    "xkcd:lavender pink",
    "xkcd:cerulean",
    "xkcd:emerald green",
    "xkcd:avocado",
    "xkcd:peach",
    "xkcd:red pink",
]


def plot_nz(zbins, mstar_thresh, nz_dir, save_dir, plt_show=True):
    fig, ax = plt.subplots(1, figsize=(7.1, 3.5), constrained_layout=True)

    alpha = 0.6
    labelsize = 10

    for zbin in range(0, len(zbins)):
        z_name = "z" + str(zbin + 1)
        for mthresh in range(0, 1):
            m_name = "m" + str(mthresh + 1)
            savename = "nz_" + z_name + "_" + m_name
            save_nz = nz_dir + "/" + savename + ".npy"

            n_of_z = np.load(save_nz)
            zgrid = n_of_z[:, 0]
            nz = n_of_z[:, 1]

            ax.plot(
                zgrid,
                nz,
                label=str(zbins[zbin][0]) + " < z < " + str(zbins[zbin][1]),
                c=colors[zbin],
                alpha=alpha,
            )
            ax.tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax.tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )

    ax.set_xlabel("redshift")
    ax.set_ylabel("N (z)")
    ax.set_xlim(-0.1, zbins[-1][1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=11)
    fig.savefig(
        save_dir + "/nz.png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    else:
        plt.close()


def plot_chains(backend_h5, title, save_dir, discard=1, thin=1, plt_show=True):
    sampler = emcee.backends.HDFBackend(backend_h5, read_only=True)
    chains = sampler.get_chain(discard=discard, thin=thin)

    fig, ax = plt.subplots(11, 1, figsize=(7.1, 8.0))
    fig.subplots_adjust(hspace=0, left=0.125, right=0.99, top=0.95, bottom=0.075)
    fig.suptitle(title, fontsize=12)

    alpha = 0.2
    lw = 0.7
    labelsize = 11
    fontsize = 14

    ax[0].plot(chains[:, :, 0], "k", alpha=alpha, lw=lw)
    ax[0].set_ylabel("M$_{\u002A,0}$", fontsize=fontsize)

    ax[1].plot(chains[:, :, 1], "k", alpha=alpha, lw=lw)
    ax[1].set_ylabel("M$_{1}$", fontsize=fontsize)

    ax[2].plot(chains[:, :, 2], "k", alpha=alpha, lw=lw)
    ax[2].set_ylabel("\u03b2", fontsize=fontsize)

    ax[3].plot(chains[:, :, 3], "k", alpha=alpha, lw=lw)
    ax[3].set_ylabel("\u03b4", fontsize=fontsize)

    ax[4].plot(chains[:, :, 4], "k", alpha=alpha, lw=lw)
    ax[4].set_ylabel("\u03b3", fontsize=fontsize)

    ax[5].plot(chains[:, :, 5], "k", alpha=alpha, lw=lw)
    ax[5].set_ylabel("\u03c3$_{logM_\u002A}$", fontsize=fontsize)

    ax[6].plot(chains[:, :, 6], "k", alpha=alpha, lw=lw)
    ax[6].set_ylabel("\u03b1$_{sat}$", fontsize=fontsize)

    ax[7].plot(chains[:, :, 7], "k", alpha=alpha, lw=lw)
    ax[7].set_ylabel("\u03b2$_{sat}$", fontsize=fontsize)

    ax[8].plot(chains[:, :, 8], "k", alpha=alpha, lw=lw)
    ax[8].set_ylabel("B$_{sat}$", fontsize=fontsize)

    ax[9].plot(chains[:, :, 9], "k", alpha=alpha, lw=lw)
    ax[9].set_ylabel("\u03b2$_{cut}$", fontsize=fontsize)

    ax[10].plot(chains[:, :, 10], "k", alpha=alpha, lw=lw)
    ax[10].set_ylabel("B$_{cut}$", fontsize=fontsize)

    ax[-1].set_xlabel("steps", fontsize=fontsize)

    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    for a in ax:
        a.ticklabel_format(style="plain", useOffset=False)
        a.tick_params(
            which="major",
            direction="in",
            top=False,
            bottom=False,
            left=True,
            right=True,
            length=4,
            width=1,
            labelsize=labelsize,
        )

    fig.savefig(
        save_dir + "/chains.png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    else:
        plt.close()


def _stellar_mass_colors(n, c1="tab:blue", c2="tab:orange"):
    cmap = mcolors.LinearSegmentedColormap.from_list("mass_grad", [c1, c2])
    return cmap(np.linspace(0, 1, n))


def redshift_luminosity_colors(zbins, mstar_thresh, band_colors):
    colors_z = []
    for zbin in range(0, len(zbins)):
        n_lum = len(mstar_thresh[zbin])
        scales = np.linspace(1.2, 0.4, n_lum)
        colors_z.append([scale_luminance(colors[zbin], s) for s in scales])
    return colors_z


def plot_tpcf(
    zbins,
    mstar_thresh,
    smf_dir,
    w_data_dir,
    Nz_dir,
    save_dir,
    savename,
    hod_post_ecsv,
    hod_model="Leauthaud11",
    plt_show=True,
):
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.5))
    fig.subplots_adjust(
        wspace=0, hspace=0, left=0.09, right=0.99, top=0.99, bottom=0.075
    )

    fontsize = 12
    labelsize = 12

    row = 0
    col = 0
    colors_z = redshift_luminosity_colors(zbins, mstar_thresh, colors)
    hod_post = ascii.read(hod_post_ecsv)

    smf_data_fits = smf_dir + "/smf_uds_cosmos_v0.1.fits"
    smf_data = Table.read(smf_data_fits)
    lmass = smf_data["Mbin"].data
    smf_z_names = ["2.0", "3.0", "4.0", "5.0", "6.0", "7.5", "10.0"]

    for zbin in range(len(zbins)):
        try:
            z_name = "z" + str(zbin + 1)
            smf_z_name = smf_z_names[zbin]
            z_min_label = str(np.round(zbins[zbin][0], 2))
            z_max_label = str(np.round(zbins[zbin][1], 2))

            ax[row][col].set_title(
                z_min_label + " < z < " + z_max_label, y=0.85, fontsize=fontsize + 2
            )

            inset_smf = ax[row][col].inset_axes([0.65, 0.6, 0.3, 0.25])

            phi_data = smf_data[smf_z_name].data
            lphi_data = np.log10(phi_data)

            err_hi = smf_data[smf_z_name + "_err_hi"].data
            err_lo = smf_data[smf_z_name + "_err_lo"].data
            lphi_err_hi = np.log10(phi_data + err_hi) - np.log10(phi_data)
            lphi_err_lo = np.log10(phi_data) - np.log10(phi_data - err_lo)

            inset_smf.errorbar(
                lmass,
                lphi_data,
                yerr=[lphi_err_lo, lphi_err_hi],
                c=colors_z[zbin][-1],
                ms=1,
                fmt="o",
                capsize=0,
                elinewidth=0.5,
            )
            inset_smf.set_ylim(-6, 0.5)
            inset_smf.set_xlim(6.2, 12)
            inset_smf.set_ylabel(
                "log$_{10}$ (\u03A6)", fontsize=fontsize / 2, labelpad=-1
            )
            inset_smf.set_xlabel(
                "log$_{10}$(M$_{*}$)", fontsize=fontsize / 2, labelpad=-1
            )
            inset_smf.tick_params(
                axis="both",
                which="both",
                direction="in",
                labelsize=fontsize / 2,
                length=fontsize / 4,
            )

            smhm_m0_0 = hod_post[hod_post["zbin"] == z_name]["smhm_m0_0"][0][0][1]
            smhm_m1_0 = hod_post[hod_post["zbin"] == z_name]["smhm_m1_0"][0][0][1]
            smhm_beta_0 = hod_post[hod_post["zbin"] == z_name]["smhm_beta_0"][0][0][1]
            smhm_delta_0 = hod_post[hod_post["zbin"] == z_name]["smhm_delta_0"][0][0][1]
            smhm_gamma_0 = hod_post[hod_post["zbin"] == z_name]["smhm_gamma_0"][0][0][1]
            sig_logmstar = hod_post[hod_post["zbin"] == z_name]["sig_logmstar"][0][0][1]
            alphasat = hod_post[hod_post["zbin"] == z_name]["alphasat"][0][0][1]
            betasat = hod_post[hod_post["zbin"] == z_name]["betasat"][0][0][1]
            bsat = hod_post[hod_post["zbin"] == z_name]["bsat"][0][0][1]
            betacut = hod_post[hod_post["zbin"] == z_name]["betacut"][0][0][1]
            bcut = hod_post[hod_post["zbin"] == z_name]["bcut"][0][0][1]

            hod_params = {
                **HOD_INITIALIZE_PARAMS,
                "sm_thresh": mstar_thresh[zbin][0],
                "smhm_m0_0": smhm_m0_0,
                "smhm_m1_0": smhm_m1_0,
                "smhm_beta_0": smhm_beta_0,
                "smhm_delta_0": smhm_delta_0,
                "smhm_gamma_0": smhm_gamma_0,
                "sig_logmstar": sig_logmstar,
                "alphasat": alphasat,
                "betasat": betasat,
                "bsat": bsat,
                "betacut": betacut,
                "bcut": bcut,
            }
            w_data_ascii = w_data_dir + "/w_" + z_name + "_m1.dat"
            w_data = ascii.read(w_data_ascii)
            sep = w_data["sep[deg]"].data
            Nz_npy = Nz_dir + "/nz_" + z_name + "_m1.npy"

            acf_model = build_acf_model(
                sep,
                Nz_npy,
                DEFAULT_COSMOLOGY,
                hod_model,
                hod_params,
                zbins[zbin][0],
                zbins[zbin][1],
            )

            lphi_model_h1p0 = get_model_smf(
                acf_model, smf_data_fits, DEFAULT_COSMOLOGY, smf_z_name
            )
            lphi_model_h0p7 = np.log10(
                (10**lphi_model_h1p0) * (DEFAULT_COSMOLOGY.h**3)
            )

            inset_smf.plot(
                lmass,
                lphi_model_h0p7,
                c=colors_z[zbin][-1],
                alpha=0.5,
            )

            for mthresh in range(0, len(mstar_thresh[zbin])):
                m_name = "m" + str(mthresh + 1)
                sample_name = z_name + "_" + m_name

                w_data_ascii = w_data_dir + "/w_" + sample_name + ".dat"
                w_data = ascii.read(w_data_ascii)
                sep = w_data["sep[deg]"].data
                corr = w_data["corr"].data
                std = w_data["std"].data

                ax[row][col].errorbar(
                    sep,
                    corr,
                    std,
                    fmt="o",
                    ms=5,
                    markerfacecolor="none",
                    c=colors_z[zbin][mthresh],
                    label="lgmstar > " + str(np.round(mstar_thresh[zbin][mthresh], 1)),
                    capsize=1,
                    markeredgewidth=0.5,
                    elinewidth=1,
                )

                Nz_npy = Nz_dir + "/nz_" + sample_name + ".npy"
                Nz = np.load(Nz_npy)
                nz = interpolate.interp1d(Nz[:, 0], Nz[:, 1], kind="cubic")
                hod_params["sm_thresh"] = mstar_thresh[zbin][mthresh]
                acf_model.update(p1=nz, hod_params=hod_params)

                ax[row][col].plot(
                    sep, acf_model.angular_corr_gal, color=colors_z[zbin][mthresh], lw=1
                )

            ax[row][col].set_yscale("log")
            ax[row][col].set_xscale("log")
            ax[row][col].set_ylim(5e-4, 2e1)
            ax[row][col].set_xlim(7e-4, 0.2)
            ax[row][col].legend(fontsize=5, loc="lower left")
            ax[row][col].minorticks_on()
            ax[row][col].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[row][col].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )

            if row == 0:
                ax[row][col].tick_params(labelbottom=False)
            if col != 0:
                ax[row][col].tick_params(labelleft=False)

        except Exception as e:
            print(
                f"Skipping zbin idx {zbin} ({z_name if 'z_name' in locals() else '?'}): {e}"
            )
            ax[row][col].set_visible(False)

        finally:
            if col == 2:
                col = 0
                row += 1
            else:
                col += 1

    fig.supxlabel(r"$\theta$ [deg]", fontsize=fontsize)
    fig.supylabel(r"$\omega$($\theta$)", fontsize=fontsize)
    fig.savefig(
        save_dir + "/" + savename + ".png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    else:
        plt.close()

    return acf_model


def plot_chi_sq(backend_h5, discard=0, thin=1):
    sampler = emcee.backends.HDFBackend(backend_h5, read_only=True)
    blobs = sampler.get_blobs(discard=discard, thin=thin, flat=True)

    # save hod parameter values at chi_sq min
    min_idx = np.argmin(blobs["chi_sq"])
    plt.plot(np.log10(blobs["chi_sq"]), "blue", alpha=0.3)
    plt.axvline(min_idx, ls="--", lw=0.5, c="k", alpha=0.5)
    plt.xlabel("iterations x walkers")
    plt.ylabel(r"$\log_{10}(\chi^{2})$")
    plt.show()
