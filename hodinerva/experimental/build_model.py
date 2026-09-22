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

    """Get Model"""
    # To choose theta, set it in degrees then convert to radians
    # theta_min = rr_ic_sep.min() * np.pi / 180.0
    # theta_max = rr_ic_sep.max() * np.pi / 180.0
    # theta_num = len(rr_ic_sep)

    model = AngularCF(
        hmf_model="Tinker10",
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

    # model.update(
    #     theta_min=sep.min() * np.pi / 180.0,
    #     theta_max=sep.max() * np.pi / 180.0,
    #     theta_num=len(sep),
    # )

    return model.clone()


# def build_smf_data_old(smf_data_ascii, cosmo):
#     smf_data = ascii.read(smf_data_ascii)
#     lmass_low = smf_data["lmasslow"].data
#     lmass = smf_data["lmass"].data
#     lmass_upp = smf_data["lmassupp"].data
#     lphi = smf_data["lphi"].data
#     lphi_h1p0 = np.log10((10**lphi) / (cosmo.h**3))

#     frac_errlow = smf_data["lphi_errlow"].data / lphi
#     lphi_errlow_h1p0 = frac_errlow * lphi_h1p0

#     frac_errupp = smf_data["lphi_errupp"].data / lphi
#     lphi_errupp_h1p0 = frac_errupp * lphi_h1p0

#     lphi_avg_err = (lphi_errlow_h1p0 + lphi_errupp_h1p0) / 2

#     smf_data = {
#         "lmass_low_h1p0": np.log10((10**lmass_low) * (cosmo.h**2)),
#         "lmass_h1p0": np.log10((10**lmass) * (cosmo.h**2)),
#         "lmass_upp_h1p0": np.log10((10**lmass_upp) * (cosmo.h**2)),
#         "lphi_h1p0": lphi_h1p0,
#         "lphi_err_h1p0": lphi_avg_err,
#     }
#     return smf_data


def build_smf_data(smf_data_fits, cosmo, z_name):
    smf_data = Table.read(smf_data_fits)

    lmass = smf_data["Mbin"].data
    lmass_h1p0 = np.log10((10**lmass) * (cosmo.h**2))

    phi = smf_data[z_name].data
    err_hi = smf_data[z_name + "_err_hi"].data
    err_lo = smf_data[z_name + "_err_lo"].data

    lphi_h1p0 = np.log10((phi) / (cosmo.h**3))

    lphi_err_hi = np.log10(phi + err_hi) - np.log10(phi)
    lphi_err_lo = np.log10(phi) - np.log10(phi - err_lo)

    lphi_avg_err = (lphi_err_hi + lphi_err_lo) / 2

    smf_data = {
        "lmass_h1p0": lmass_h1p0,
        "lphi_h1p0": lphi_h1p0,
        "lphi_err_h1p0": lphi_avg_err,
    }
    return smf_data


def get_model_smf(model, smf_data_fits, cosmo, z_name):
    smf_data = build_smf_data(smf_data_fits, cosmo, z_name)

    phi_model_h1p0 = []
    dlogMbin = np.diff(smf_data["lmass_h1p0"])[0] / 2
    for Mbin in range(0, len(smf_data["lmass_h1p0"])):
        model.update(
            **{"hod_params": {"sm_thresh": smf_data["lmass_h1p0"][Mbin] - dlogMbin}}
        )
        total_occupation_lo = model._total_occupation

        model.update(
            **{"hod_params": {"sm_thresh": smf_data["lmass_h1p0"][Mbin] + dlogMbin}}
        )
        total_occupation_hi = model._total_occupation

        total_occupation = total_occupation_lo - total_occupation_hi
        ngal = tools.spline_integral(model.m, model.dndm * total_occupation)
        phi_model_h1p0.append(ngal / dlogMbin)

    # lphi_model_h0p7 = np.log10(np.array(phi_model) * (cosmo.h**3))

    lphi_model_h1p0 = np.log10(np.array(phi_model_h1p0))
    return lphi_model_h1p0
