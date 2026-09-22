import os

import corner
import emcee
import numpy as np
from astropy.io import ascii
from astropy.table import Column, Table


def make_randoms(mask, ra_edges, dec_edges, n_random, oversample=1.5):
    ra_min, ra_max = ra_edges[0], ra_edges[-1]
    sin_min, sin_max = np.sin(np.radians(dec_edges[0])), np.sin(
        np.radians(dec_edges[-1])
    )

    n_try = int(n_random * oversample)
    ra_r = np.random.uniform(ra_min, ra_max, n_try)
    dec_r = np.degrees(np.arcsin(np.random.uniform(sin_min, sin_max, n_try)))

    i = np.clip(np.searchsorted(ra_edges, ra_r) - 1, 0, mask.shape[0] - 1)
    j = np.clip(np.searchsorted(dec_edges, dec_r) - 1, 0, mask.shape[1] - 1)
    keep = mask[i, j]

    ra_r, dec_r = ra_r[keep], dec_r[keep]
    if len(ra_r) < n_random:
        raise ValueError(f"only got {len(ra_r)}/{n_random}, increase oversample")
    return ra_r[:n_random], dec_r[:n_random]


def get_pz_from_chi2(chi2, idx):
    ln_pz = chi2[idx] / (-2)
    pz = np.exp(ln_pz)
    return pz


def get_stacked_pz(chi2, idx_arr, zgrid, zmin, zmax):
    z_i = np.argmin(np.abs(zgrid - zmin))
    z_f = np.argmin(np.abs(zgrid - zmax))

    excluded_idx = []
    pz_stack = np.zeros(len(zgrid))
    for idx in idx_arr:
        pz = get_pz_from_chi2(chi2, idx)
        if np.trapezoid(pz, zgrid) == 0.0:
            excluded_idx.append(idx)
        else:
            pz_normalized = pz / np.trapezoid(pz, zgrid)
            pz_weight = np.trapezoid(pz_normalized[z_i:z_f], zgrid[z_i:z_f])
            pz_stack += pz_weight * pz_normalized

    pz_stack = pz_stack / np.trapezoid(pz_stack, zgrid)

    excluded_percentage = np.round((len(excluded_idx) / len(idx_arr)) * 100, 2)
    print(str(excluded_percentage) + "% of objects excluded")

    return pz_stack, excluded_idx


def get_zbin_mthresh_samples(cat, zbins, mstar_thresh, chi2, zgrid):
    zbin_cats = []
    zbin_pz = []
    for zbin in range(0, len(zbins)):
        zmin = zbins[zbin][0]
        zmax = zbins[zbin][1]

        z_mask = (cat["zbest"] > zmin) & (cat["zbest"] <= zmax)

        mth_cats = []
        mth_pz = []
        for mthresh in range(0, len(mstar_thresh[zbin])):
            sm_mask = cat["mstar50"] > mstar_thresh[zbin][mthresh]

            idx_arr = np.where(z_mask & sm_mask)[0]
            stacked_pz, excluded_idx = get_stacked_pz(chi2, idx_arr, zgrid, zmin, zmax)
            included_idx = np.setdiff1d(idx_arr, excluded_idx)

            mth_cats.append(cat[included_idx])
            mth_pz.append(stacked_pz)

        zbin_cats.append(mth_cats)
        zbin_pz.append(mth_pz)

    return zbin_cats, zbin_pz


def get_quantiles(backend_h5, discard=0, thin=1):
    sampler = emcee.backends.HDFBackend(backend_h5, read_only=True)
    flatchain = sampler.get_chain(discard=discard, thin=thin, flat=True)

    quantiles = []
    _, n_param = flatchain.shape
    for i in range(0, n_param):
        quantiles.append(corner.quantile(flatchain[:, i], [0.16, 0.5, 0.84]))
    return quantiles


def save_hod_post(chain_dir, savedir, savename, discard=0, thin=1):
    chains = os.listdir(chain_dir)
    n_samples = len(chains)

    hod_post = Table()

    hod_post.add_column(Column(name="zbin", length=n_samples, dtype="U10"))

    hod_post.add_column(Column(name="smhm_m0_0", length=n_samples, shape=(1, 3)))
    hod_post["smhm_m0_0"].info.format = "7.5f"
    hod_post.add_column(Column(name="smhm_m0_0_chi2min", length=n_samples))
    hod_post["smhm_m0_0_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="smhm_m1_0", length=n_samples, shape=(1, 3)))
    hod_post["smhm_m1_0"].info.format = "7.5f"
    hod_post.add_column(Column(name="smhm_m1_0_chi2min", length=n_samples))
    hod_post["smhm_m1_0_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="smhm_beta_0", length=n_samples, shape=(1, 3)))
    hod_post["smhm_beta_0"].info.format = "7.5f"
    hod_post.add_column(Column(name="smhm_beta_0_chi2min", length=n_samples))
    hod_post["smhm_beta_0_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="smhm_delta_0", length=n_samples, shape=(1, 3)))
    hod_post["smhm_delta_0"].info.format = "7.5f"
    hod_post.add_column(Column(name="smhm_delta_0_chi2min", length=n_samples))
    hod_post["smhm_delta_0_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="smhm_gamma_0", length=n_samples, shape=(1, 3)))
    hod_post["smhm_gamma_0"].info.format = "7.5f"
    hod_post.add_column(Column(name="smhm_gamma_0_chi2min", length=n_samples))
    hod_post["smhm_gamma_0_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="sig_logmstar", length=n_samples, shape=(1, 3)))
    hod_post["sig_logmstar"].info.format = "7.5f"
    hod_post.add_column(Column(name="sig_logmstar_chi2min", length=n_samples))
    hod_post["sig_logmstar_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="alphasat", length=n_samples, shape=(1, 3)))
    hod_post["alphasat"].info.format = "7.5f"
    hod_post.add_column(Column(name="alphasat_chi2min", length=n_samples))
    hod_post["alphasat_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="betasat", length=n_samples, shape=(1, 3)))
    hod_post["betasat"].info.format = "7.5f"
    hod_post.add_column(Column(name="betasat_chi2min", length=n_samples))
    hod_post["betasat_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="bsat", length=n_samples, shape=(1, 3)))
    hod_post["bsat"].info.format = "7.5f"
    hod_post.add_column(Column(name="bsat_chi2min", length=n_samples))
    hod_post["bsat_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="betacut", length=n_samples, shape=(1, 3)))
    hod_post["betacut"].info.format = "7.5f"
    hod_post.add_column(Column(name="betacut_chi2min", length=n_samples))
    hod_post["betacut_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="bcut", length=n_samples, shape=(1, 3)))
    hod_post["bcut"].info.format = "7.5f"
    hod_post.add_column(Column(name="bcut_chi2min", length=n_samples))
    hod_post["bcut_chi2min"].info.format = "7.5f"

    hod_post.add_column(Column(name="chi_sq", length=n_samples))
    hod_post["chi_sq"].info.format = "5.3f"

    hod_post.add_column(Column(name="chi_sq_red", length=n_samples))
    hod_post["chi_sq_red"].info.format = "5.3f"

    hod_post.add_column(Column(name="min_chi_sq", length=n_samples))
    hod_post["min_chi_sq"].info.format = "5.3f"

    hod_post.add_column(Column(name="min_chi_sq_red", length=n_samples))
    hod_post["min_chi_sq_red"].info.format = "5.3f"

    for j in range(0, n_samples):
        backend_h5 = chain_dir + "/" + chains[j]
        quantiles = get_quantiles(backend_h5, discard=discard, thin=thin)

        hod_post["zbin"][j] = chains[j].split(".")[0]
        hod_post["smhm_m0_0"][j] = quantiles[0]
        hod_post["smhm_m1_0"][j] = quantiles[1]
        hod_post["smhm_beta_0"][j] = quantiles[2]
        hod_post["smhm_delta_0"][j] = quantiles[3]
        hod_post["smhm_gamma_0"][j] = quantiles[4]
        hod_post["sig_logmstar"][j] = quantiles[5]
        hod_post["alphasat"][j] = quantiles[6]
        hod_post["betasat"][j] = quantiles[7]
        hod_post["bsat"][j] = quantiles[8]
        hod_post["betacut"][j] = quantiles[9]
        hod_post["bcut"][j] = quantiles[10]

    ascii.write(
        hod_post,
        savedir + "/" + savename + ".ecsv",
        format="ecsv",
        overwrite=True,
    )
    return hod_post
