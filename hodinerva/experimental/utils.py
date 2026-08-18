import numpy as np


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

        z_mask = (cat["z_ml"] > zmin) & (cat["z_ml"] <= zmax)

        mth_cats = []
        mth_pz = []
        for mthresh in range(0, len(mstar_thresh[zbin])):
            sm_mask = cat["mstar_50"] > mstar_thresh[zbin][mthresh]

            idx_arr = np.where(z_mask & sm_mask)[0]
            stacked_pz, excluded_idx = get_stacked_pz(chi2, idx_arr, zgrid, zmin, zmax)
            included_idx = np.setdiff1d(idx_arr, excluded_idx)

            mth_cats.append(cat[included_idx])
            mth_pz.append(stacked_pz)

        zbin_cats.append(mth_cats)
        zbin_pz.append(mth_pz)

    return zbin_cats, zbin_pz
