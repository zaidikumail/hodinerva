import halomod.tools as tools
import numpy as np
from astropy.table import Table
from halomod.integrate_corr import AngularCF
from scipy import interpolate


def build_acf_model(sep, Nz_npy, cosmo, hod_model, hod_params, zmin, zmax):
    theta_min = sep.min() * np.pi / 180.0
    theta_max = sep.max() * np.pi / 180.0
    theta_num = len(sep)

    Nz = np.load(Nz_npy)
    nz = interpolate.interp1d(Nz[:, 0], Nz[:, 1], kind="cubic")

    model = AngularCF(
        hmf_model="Behroozi",
        bias_model="Tinker10",
        # sd_bias_model='TinkerSD05',
        transfer_model="EH",
        halo_concentration_model="Duffy08",
        halo_profile_model="NFW",
        mdef_model="SOVirial",
        cosmo_model=cosmo,
        hod_model=hod_model,
        hod_params=hod_params,
        logu_min=-3,
        logu_max=1.5,
        rnum=200,
        p1=nz,
        zmin=zmin,
        zmax=zmax,
        z=np.median([zmin, zmax]),
        theta_min=theta_min,
        theta_max=theta_max,
        theta_num=theta_num,
        theta_log=True,
        p_of_z=True,
    )

    return model.clone()


def get_ic(acf_model, rr_ic_sep, rr_ic_counts):
    acf_model_for_ic = acf_model.clone()

    acf_model_for_ic.update(
        theta_min=rr_ic_sep.min() * np.pi / 180.0,
        theta_max=rr_ic_sep.max() * np.pi / 180.0,
        theta_num=len(rr_ic_sep),
    )
    ic = np.sum(acf_model_for_ic.angular_corr_gal * rr_ic_counts) / np.sum(rr_ic_counts)

    return ic


def build_smf_data(smf_data_fits, cosmo, z_name, lmass_thresh_h0p7, h0p7=0.7):
    """
    Notes:
        Builds the SMF ready for fitting:
            - Drops bins below completeness limit at that redshift, lmass_thresh_h0p7
            - Drops any bins with phi <= 0
            - For bins where phi - err_lo <=0,
            the output, lphi_avg_err is just lphi_err_hi,
            otherwise an average of (lphi_err_hi, lphi_err_lo)
    """
    smf_data = Table.read(smf_data_fits)

    diff = np.diff(smf_data["Mbin"].data)
    assert np.all(diff == diff[0])
    d_lmass = diff[0]

    lmass_h0p7 = smf_data["Mbin"].data
    phi_h0p7 = smf_data[z_name].data
    err_h0p7_hi = smf_data[z_name + "_err_hi"].data
    err_h0p7_lo = smf_data[z_name + "_err_lo"].data

    lmass_h0p7_lo = lmass_h0p7 - (d_lmass / 2)
    lmass_h0p7_sel = (lmass_h0p7_lo >= lmass_thresh_h0p7) & (phi_h0p7 > 0)

    lmass_h0p7 = lmass_h0p7[lmass_h0p7_sel]
    lmass_h1p0 = lmass_h0p7 + np.log10(h0p7**2)

    phi_h0p7 = phi_h0p7[lmass_h0p7_sel]
    err_h0p7_hi = err_h0p7_hi[lmass_h0p7_sel]
    err_h0p7_lo = err_h0p7_lo[lmass_h0p7_sel]

    lphi_h0p7 = np.log10(phi_h0p7)
    lphi_h1p0 = lphi_h0p7 - np.log10(h0p7**3)

    lphi_err_hi = np.log10(phi_h0p7 + err_h0p7_hi) - np.log10(phi_h0p7)

    lo_neg = (phi_h0p7 - err_h0p7_lo) <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        lphi_err_lo = np.log10(phi_h0p7) - np.log10(phi_h0p7 - err_h0p7_lo)

    lphi_avg_err = np.where(lo_neg, lphi_err_hi, (lphi_err_hi + lphi_err_lo) / 2)

    smf_data = {
        "lmass_h1p0": lmass_h1p0,
        "lphi_h1p0": lphi_h1p0,
        "lphi_err": lphi_avg_err,
        "lmass_h0p7": lmass_h0p7,
        "lphi_h0p7": lphi_h0p7,
        "d_lmass": d_lmass,
    }
    return smf_data


def get_model_smf(model, smf_data, cosmo):
    d_lmass = smf_data["d_lmass"]

    phi_model_h1p0 = []
    for Mbin in range(0, len(smf_data["lmass_h1p0"])):
        model.update(
            **{
                "hod_params": {
                    "sm_thresh": smf_data["lmass_h1p0"][Mbin] - (d_lmass / 2)
                }
            }
        )
        total_occupation_lo = model._total_occupation

        model.update(
            **{
                "hod_params": {
                    "sm_thresh": smf_data["lmass_h1p0"][Mbin] + (d_lmass / 2)
                }
            }
        )
        total_occupation_hi = model._total_occupation

        total_occupation = total_occupation_lo - total_occupation_hi
        ngal = tools.spline_integral(model.m, model.dndm * total_occupation)
        phi_model_h1p0.append(ngal / d_lmass)

    lphi_model_h1p0 = np.log10(np.array(phi_model_h1p0))
    return lphi_model_h1p0
