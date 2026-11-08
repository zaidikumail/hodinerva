import numpy as np
from numpy.linalg import pinv

from ..kernels.build_model import get_ic, get_model_smf


def chi_sq(
    model,
    w_data,
    w_rr_ic_sep,
    w_rr_ic_counts,
    smf_data,
    cosmo,
):
    w_chi_sq = get_w_chi_sq(w_data, model, w_rr_ic_sep, w_rr_ic_counts)
    print(f"w: Δχ² = {w_chi_sq:.2f}")

    smf_chi_sq = get_smf_chi_sq(model, smf_data, cosmo)
    print(f"SMF: Δχ² = {smf_chi_sq:.2f}")

    return w_chi_sq + smf_chi_sq


# correlation function term
def get_w_chi_sq(w_data, model, rr_ic_sep, rr_ic_counts):
    chi_sq = 0.0
    for Mth in range(len(w_data)):
        model.update(**{"hod_params": {"sm_thresh": w_data[Mth]["sm_thresh_h1p0"]}})

        ic = get_ic(model, rr_ic_sep, rr_ic_counts)

        # Add the contribution to the chi_sq of the current Mth sample
        corr_model = model.angular_corr_gal - ic
        corr_datum = w_data[Mth]["corr"]
        # C_inv = w_data[Mth]["C_inv"]

        # temporarily not fit corr<0 bins
        Cov = w_data[Mth]["Cov"]
        keep = corr_datum > 0
        corr_model = corr_model[keep]
        corr_datum = corr_datum[keep]
        C_inv = pinv(Cov[np.ix_(keep, keep)])

        corr_diff = corr_datum - corr_model
        d_chi = corr_diff.T @ C_inv @ corr_diff
        chi_sq += d_chi

    return chi_sq


def get_smf_chi_sq(model, smf_data, cosmo):
    lphi_model_h1p0 = get_model_smf(model, smf_data, cosmo)

    lphi_data_h1p0 = smf_data["lphi_h1p0"]
    lphi_err = smf_data["lphi_err"]

    chi_sq_per_bin = ((lphi_data_h1p0 - lphi_model_h1p0) / (lphi_err)) ** 2

    chi_sq = np.sum(chi_sq_per_bin)

    return chi_sq
