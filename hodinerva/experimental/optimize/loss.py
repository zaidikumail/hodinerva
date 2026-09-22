import numpy as np

from ..build_model import get_model_smf


def chi_sq(
    model,
    w_data,
    w_rr_ic_sep,
    w_rr_ic_counts,
    smf_data_fits,
    smf_data,
    smf_z_name,
    cosmo,
):
    w_chi_sq = get_w_chi_sq(w_data, model, w_rr_ic_sep, w_rr_ic_counts)
    print(f"w: Δχ² = {w_chi_sq:.2f}")

    smf_chi_sq = get_smf_chi_sq(model, smf_data_fits, smf_data, cosmo, smf_z_name)
    print(f"SMF: Δχ² = {smf_chi_sq:.2f}")

    return w_chi_sq + smf_chi_sq


# correlation function term
def get_w_chi_sq(w_data, model, rr_ic_sep, rr_ic_counts):
    chi_sq = 0.0
    for Mth in range(0, len(w_data)):
        model.update(**{"hod_params": {"sm_thresh": w_data[Mth]["sm_thresh_h1p0"]}})

        # update the thetas of the model to calculate the IC with the new hod parameters
        model.update(
            theta_min=rr_ic_sep.min() * np.pi / 180.0,
            theta_max=rr_ic_sep.max() * np.pi / 180.0,
            theta_num=len(rr_ic_sep),
        )
        IC = np.sum(model.angular_corr_gal * rr_ic_counts) / np.sum(rr_ic_counts)

        # revert back the thetas to calculate the chi-square
        model.update(
            theta_min=w_data[Mth]["sep"].min() * np.pi / 180.0,
            theta_max=w_data[Mth]["sep"].max() * np.pi / 180.0,
            theta_num=len(w_data[Mth]["sep"]),
        )

        # Add the contribution to the chi_sq of the current Mth sample
        corr_model = model.angular_corr_gal - IC
        corr_datum = w_data[Mth]["corr"]
        C_inv = w_data[Mth]["C_inv"]

        corr_diff = corr_datum - corr_model
        d_chi = corr_diff.T @ C_inv @ corr_diff
        chi_sq += d_chi

    return chi_sq


def get_smf_chi_sq(model, smf_data_fits, smf_data, cosmo, smf_z_name):
    lphi_model_h1p0 = get_model_smf(model, smf_data_fits, cosmo, smf_z_name)

    lphi_data_h1p0 = smf_data["lphi_h1p0"]
    lphi_err = smf_data["lphi_err_h1p0"]

    chi_sq_per_bin = ((lphi_data_h1p0 - lphi_model_h1p0) / (lphi_err)) ** 2
    chi_sq_per_bin = np.nan_to_num(chi_sq_per_bin, nan=0.0, posinf=0.0, neginf=0.0)

    chi_sq = np.sum(chi_sq_per_bin)

    return chi_sq
