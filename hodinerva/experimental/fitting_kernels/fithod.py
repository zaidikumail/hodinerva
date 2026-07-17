import sys
import time
import warnings

import emcee
import halomod.tools as tools
import numpy as np
from astropy.io import ascii
from halomod.integrate_corr import AngularCF
from numpy.linalg import pinv
from schwimmbad import MPIPool
from scipy import interpolate


class Fit_HOD:
    def __init__(
        self, zmin, zmax, Nz_npy, RR_IC, sep_RR_IC, hod_model, hod_params, cosmo
    ):
        self.zmin = zmin
        self.zmax = zmax
        self.Nz_npy = Nz_npy
        self.RR_IC = RR_IC
        self.sep_RR_IC = sep_RR_IC
        self.hod_model = hod_model
        self.hod_params = hod_params
        self.cosmo = cosmo

    def setup_w_data(self, sm_thresh, w_data_ascii, cov_jknife_npy=None, mock_out=None):
        """Get correlation function data"""
        w_data = ascii.read(w_data_ascii)

        sep = w_data["sep[deg]"].data
        self.sep = sep

        corr = w_data["corr"].data
        std = w_data["std"].data
        N = w_data["N"][0]

        if isinstance(mock_out, str):
            # mock covariance
            Cov = np.load(mock_out, allow_pickle=True).item()["Cov"]
            C_inv = pinv(Cov)

        elif isinstance(cov_jknife_npy, str):
            # jackknife covariance
            Cov = np.load(cov_jknife_npy)
            C_inv = pinv(Cov)

        else:
            warnings.warn("No covariance provided! setting Cov=None, C_inv=None")
            Cov = None
            C_inv = None

        w_data = {
            "sm_thresh_h1p0": np.log10((10**sm_thresh) * (self.cosmo.h**2)),
            "sep": sep,
            "corr": corr,
            "std": std,
            "Cov": Cov,
            "C_inv": C_inv,
            "N": N,
        }

        return w_data

    def setup_smf_data(self, SMF_data_ascii):
        smf_data = ascii.read(SMF_data_ascii)
        lmass_low = smf_data["lmasslow"].data
        lmass = smf_data["lmass"].data
        lmass_upp = smf_data["lmassupp"].data
        lphi = smf_data["lphi"].data
        lphi_h1p0 = np.log10((10**lphi) / (self.cosmo.h**3))

        frac_errlow = smf_data["lphi_errlow"].data / lphi
        lphi_errlow_h1p0 = frac_errlow * lphi_h1p0

        frac_errupp = smf_data["lphi_errupp"].data / lphi
        lphi_errupp_h1p0 = frac_errupp * lphi_h1p0

        lphi_avg_err = (lphi_errlow_h1p0 + lphi_errupp_h1p0) / 2

        self.smf_data = {
            "lmass_low_h1p0": np.log10((10**lmass_low) * (self.cosmo.h**2)),
            "lmass_h1p0": np.log10((10**lmass) * (self.cosmo.h**2)),
            "lmass_upp_h1p0": np.log10((10**lmass_upp) * (self.cosmo.h**2)),
            "lphi_h1p0": lphi_h1p0,
            "lphi_err_h1p0": lphi_avg_err,
        }

    def setup_model(self, w_data_ascii):
        # get data sep array
        w_data = ascii.read(w_data_ascii)
        sep = w_data["sep[deg]"].data

        # Nz = np.load(self.Nz_npy, allow_pickle=True)
        # nz = interp1d(Nz[0], Nz[1])

        Nz = np.load(self.Nz_npy)
        nz = interpolate.interp1d(Nz[:, 0], Nz[:, 1], kind="cubic")

        """Get Model"""
        # To choose theta, set it in degrees then convert to radians
        theta_min = self.sep_RR_IC.min() * np.pi / 180.0
        theta_max = self.sep_RR_IC.max() * np.pi / 180.0
        theta_num = len(self.sep_RR_IC)

        model = AngularCF(
            hmf_model="Courtin",
            bias_model="Tinker10",
            # sd_bias_model='TinkerSD05',
            transfer_model="EH",
            halo_concentration_model="Duffy08",
            halo_profile_model="NFW",
            mdef_model="FOF",
            cosmo_model=self.cosmo,
            hod_model=self.hod_model,
            hod_params=self.hod_params,
            logu_min=-3,
            logu_max=1.5,
            rnum=200,
            p1=nz,
            zmin=self.zmin,
            zmax=self.zmax,
            z=self.zmin + ((self.zmax - self.zmin) / 2),
            theta_min=theta_min,
            theta_max=theta_max,
            theta_num=theta_num,
            theta_log=True,
            p_of_z=True,
        )

        model.update(
            theta_min=sep.min() * np.pi / 180.0,
            theta_max=sep.max() * np.pi / 180.0,
            theta_num=len(sep),
        )

        self.model = model.clone()

    def chi_square(self):
        chi_sq = 0

        # correlation function term
        for Mth in range(0, len(self.w_data)):
            if self.hod_model == "Leauthaud11":
                self.model.update(
                    **{"hod_params": {"sm_thresh": self.w_data[Mth]["sm_thresh_h1p0"]}}
                )
            else:
                print("Zheng05 model, no updating stellar mass thresholds")
            # update the thetas of the model to calculate the IC with the new hod parameters
            self.model.update(
                theta_min=self.sep_RR_IC.min() * np.pi / 180.0,
                theta_max=self.sep_RR_IC.max() * np.pi / 180.0,
                theta_num=len(self.sep_RR_IC),
            )
            IC = np.sum(self.model.angular_corr_gal * self.RR_IC) / np.sum(self.RR_IC)

            # revert back the thetas to calculate the chi-square
            self.model.update(
                theta_min=self.w_data[Mth]["sep"].min() * np.pi / 180.0,
                theta_max=self.w_data[Mth]["sep"].max() * np.pi / 180.0,
                theta_num=len(self.w_data[Mth]["sep"]),
            )

            # Add the contribution to the chi_sq of the current Mth sample
            corr_model = self.model.angular_corr_gal - IC
            corr_datum = self.w_data[Mth]["corr"]
            C_inv = self.w_data[Mth]["C_inv"]
            std = self.w_data[Mth]["std"]

            # corr func term
            corr_diff = corr_datum - corr_model

            # chi_sq += np.sum(((corr_datum-corr_model)/std)**2)
            d_chi = corr_diff.T @ C_inv @ corr_diff
            d_chi2 = np.dot(corr_diff, np.dot(C_inv, corr_diff))
            d_chi3 = np.matmul(np.matmul(corr_diff, C_inv), corr_diff)
            print(f"w_m{Mth+1:02d}: Δχ² = {d_chi:.2f}, {d_chi2:.2f}, {d_chi3:.2f}")
            chi_sq += d_chi

        # SMF term
        delta_logMbin = (
            self.smf_data["lmass_upp_h1p0"][0] - self.smf_data["lmass_low_h1p0"][0]
        )
        for Mbin in range(0, len(self.smf_data["lmass_h1p0"])):
            self.model.update(
                **{"hod_params": {"sm_thresh": self.smf_data["lmass_low_h1p0"][Mbin]}}
            )
            total_occupation_low = self.model._total_occupation

            self.model.update(
                **{"hod_params": {"sm_thresh": self.smf_data["lmass_upp_h1p0"][Mbin]}}
            )
            total_occupation_upp = self.model._total_occupation

            total_occupation = total_occupation_low - total_occupation_upp
            ngal = tools.spline_integral(
                self.model.m, self.model.dndm * total_occupation
            )
            phi_model = ngal / delta_logMbin

            phi_data = 10 ** self.smf_data["lphi_h1p0"][Mbin]
            phi_err = 10 ** self.smf_data["lphi_err_h1p0"][Mbin]

            chi_sq += (
                (np.log10(phi_data) - np.log10(phi_model)) / (np.log10(phi_err))
            ) ** 2

        return chi_sq

    def flat_to_nested_dict(self, dct: dict) -> dict:
        """Convert a dct of key: value pairs into a nested dict.

        Keys that have dots in them indicate nested structure.
        """

        def key_to_dct(key, val, dct):
            if "." in key:
                key, parts = key.split(".", maxsplit=1)

                if key not in dct:
                    dct[key] = {}

                key_to_dct(parts, val, dct[key])
            else:
                dct[key] = val

        out = {}
        for k, v in dct.items():
            key_to_dct(k, v, out)

        return out

    def log_prob(self, param_values):
        # Pack parameters into a dict
        params = dict(zip(self.param_names, param_values))

        # Allow for simple bounded flat priors.
        bounds = self.hod_bounds or {}
        for key, val in params.items():
            bound = bounds.get(key, (-np.inf, np.inf))
            if not bound[0] < val < bound[1]:
                return (
                    (-np.inf,)
                    + (None,) * len(self.hod_derived) * len(self.w_data)
                    + (None,)
                )

        # Update the base model with all the parameters that are being constrained.
        params = self.flat_to_nested_dict(params)
        self.model.update(**params)

        chi_sq = self.chi_square()
        ll = chi_sq / (-2)

        if not np.isfinite(ll):
            return (
                (-np.inf,)
                + (None,) * len(self.hod_derived) * len(self.w_data)
                + (None,)
            )

        derived = ()
        #    derived += tuple(getattr(model, d) for d in hod_derived)
        for Mth in range(0, len(self.w_data)):
            derived += tuple(getattr(self.model, d) for d in self.hod_derived)

        out = (ll,) + derived + (chi_sq,)
        return out

    def run_mcmc(self, nwalkers=20, ndim=5, nsteps=10, restart=0):
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            sampler = emcee.EnsembleSampler(
                nwalkers=nwalkers,
                ndim=ndim,
                log_prob_fn=self.log_prob,
                pool=pool,
                blobs_dtype=self.blobs_dtype,
                backend=self.backend,
            )

            start = time.time()

            if restart == 1:
                sampler.run_mcmc(None, nsteps=nsteps, progress=True)
            else:
                initialpos_normal = self.initialpos + 1e-4 * np.random.normal(
                    size=(sampler.nwalkers, sampler.ndim)
                )
                sampler.run_mcmc(initialpos_normal, nsteps=nsteps, progress=True)

            end = time.time()
            multi_time = end - start
            print("MPI took {0:.1f} seconds".format(multi_time))

        return sampler
