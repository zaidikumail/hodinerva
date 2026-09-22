import sys
import time

import emcee
import halomod.tools as tools
import numpy as np
from astropy.io import ascii
from halomod.integrate_corr import AngularCF
from numpy.linalg import pinv
from schwimmbad import MPIPool


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

    def setup_w_data(self, sm_thresh, w_data_ascii, cov_npy):
        """Get correlation function data"""
        w_data = ascii.read(w_data_ascii)

        sep = w_data["sep[deg]"].data
        self.sep = sep

        corr = w_data["corr"].data
        std = w_data["std"].data
        #        RR = w_data['RR'].data
        N = w_data["N"][0]

        C_inv = pinv(np.load(cov_npy))

        w_data = {
            "sm_thresh_h1p0": np.log10((10**sm_thresh) * (self.cosmo.h**2)),
            "sep": sep,
            "corr": corr,
            "std": std,
            #                  'RR': RR,
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
            hmf_model="Tinker10",
            bias_model="Tinker10",
            # sd_bias_model='TinkerSD05',
            transfer_model="EH",
            halo_concentration_model="Duffy08",
            halo_profile_model="NFW",
            mdef_model="SOVirial",
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

        fit_hod_z = self.zlist
        # Jointly fitting correlation functions and SMFs at all redshifts
        for z in range(0, len(fit_hod_z)):
            # correlation function term (summing over all stellar mass thresholds)
            for Mth in range(0, len(fit_hod_z[z].w_data)):
                fit_hod_z[z].model.update(
                    **{
                        "hod_params": {
                            "sm_thresh": fit_hod_z[z].w_data[Mth]["sm_thresh_h1p0"]
                        }
                    }
                )

                # update the thetas of the model to calculate the IC with the new hod parameters
                fit_hod_z[z].model.update(
                    theta_min=fit_hod_z[z].sep_RR_IC.min() * np.pi / 180.0,
                    theta_max=fit_hod_z[z].sep_RR_IC.max() * np.pi / 180.0,
                    theta_num=len(fit_hod_z[z].sep_RR_IC),
                )

                IC = np.sum(
                    fit_hod_z[z].model.angular_corr_gal * fit_hod_z[z].RR_IC
                ) / np.sum(fit_hod_z[z].RR_IC)
                # revert back the thetas to calculate the chi-square
                fit_hod_z[z].model.update(
                    theta_min=fit_hod_z[z].w_data[Mth]["sep"].min() * np.pi / 180.0,
                    theta_max=fit_hod_z[z].w_data[Mth]["sep"].max() * np.pi / 180.0,
                    theta_num=len(fit_hod_z[z].w_data[Mth]["sep"]),
                )

                # Add the contribution to the chi_sq of the current Mth sample
                corr_model = fit_hod_z[z].model.angular_corr_gal - IC
                corr_datum = fit_hod_z[z].w_data[Mth]["corr"]
                C_inv = fit_hod_z[z].w_data[Mth]["C_inv"]

                # print('corr_datum: ')
                # print(corr_datum)
                # print('corr_model')
                # print(corr_model)
                # print(np.sum(C_inv))

                # corr func term
                corr_diff = corr_datum - corr_model
                chi_sq += np.matmul(np.matmul(corr_diff, C_inv), corr_diff)

            # SMF term (summing over all stellar mass bins)
            delta_logMbin = (
                fit_hod_z[z].smf_data["lmass_upp_h1p0"][0]
                - fit_hod_z[z].smf_data["lmass_low_h1p0"][0]
            )
            for Mbin in range(0, len(fit_hod_z[z].smf_data["lmass_h1p0"])):
                fit_hod_z[z].model.update(
                    **{
                        "hod_params": {
                            "sm_thresh": fit_hod_z[z].smf_data["lmass_low_h1p0"][Mbin]
                        }
                    }
                )
                total_occupation_low = fit_hod_z[z].model._total_occupation

                fit_hod_z[z].model.update(
                    **{
                        "hod_params": {
                            "sm_thresh": fit_hod_z[z].smf_data["lmass_upp_h1p0"][Mbin]
                        }
                    }
                )
                total_occupation_upp = fit_hod_z[z].model._total_occupation

                total_occupation = total_occupation_low - total_occupation_upp
                ngal = tools.spline_integral(
                    fit_hod_z[z].model.m, fit_hod_z[z].model.dndm * total_occupation
                )
                phi_model = ngal / delta_logMbin

                phi_data = 10 ** fit_hod_z[z].smf_data["lphi_h1p0"][Mbin]
                phi_err = 10 ** fit_hod_z[z].smf_data["lphi_err_h1p0"][Mbin]

                # print('log phi_data:')
                # print(np.log10(phi_data))
                # print(np.log10(phi_err))
                # print('log phi_model:')
                print(
                    ((np.log10(phi_data) - np.log10(phi_model)) / (np.log10(phi_err)))
                    ** 2
                )

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
        # Allow for simple bounded flat priors.
        bounds = self.hod_bounds or {}
        for p in range(0, len(self.param_names)):
            key = self.param_names[p]
            val = param_values[p]
            bound = bounds.get(key, (-np.inf, np.inf))
            if not bound[0] < val < bound[1]:
                return (-np.inf,) + (None,)

        # continuity prior
        lp = 0

        # M0 prior
        for p in range(0, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.M0_prior_sigma
            x = val - param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x) / sigma) ** 2
            )

        # M1 prior
        for p in range(1, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.M1_prior_sigma
            x = val - param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x) / sigma) ** 2
            )

        # beta prior
        for p in range(2, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.beta_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # delta prior
        for p in range(3, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.delta_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # gamma prior
        for p in range(4, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.gamma_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # sig_logmstar prior
        for p in range(5, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.sig_logmstar_prior_sigma
            x = val - param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x) / sigma) ** 2
            )

        # alphasat prior
        for p in range(6, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.alphasat_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # betasat prior
        for p in range(7, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.betasat_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # Bsat prior
        for p in range(8, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.Bsat_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # betacut prior
        for p in range(9, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.betacut_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # Bcut prior
        for p in range(10, len(self.param_names) - 11, 11):
            key = self.param_names[p]
            val = param_values[p]

            sigma = self.Bcut_prior_sigma
            x = val / param_values[p + 11]
            lp += np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (
                ((x - 1) / sigma) ** 2
            )

        # Update the base model with all the parameters that are being constrained.
        fit_hod_z = self.zlist
        z_params_idx = 0
        for z in range(0, len(fit_hod_z)):
            # print(self.param_names[z_params_idx:z_params_idx+11])

            z_params = dict(
                zip(
                    self.param_names[z_params_idx : z_params_idx + 11],
                    param_values[z_params_idx : z_params_idx + 11],
                )
            )
            z_params = self.flat_to_nested_dict(z_params)

            fit_hod_z[z].model.update(**z_params)

            z_params_idx += 11

        chi_sq = self.chi_square()
        # print(chi_sq)
        ll = chi_sq / (-2)
        ll += lp

        if not np.isfinite(ll):
            return (-np.inf,) + (None,)

        out = (ll,) + (chi_sq,)
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
                sampler.run_mcmc(None, nsteps=nsteps, progress=True, thin_by=1)
            else:
                initialpos_normal = self.initialpos + 1e-4 * np.random.normal(
                    size=(sampler.nwalkers, sampler.ndim)
                )
                sampler.run_mcmc(
                    initialpos_normal, nsteps=nsteps, progress=True, thin_by=1
                )

            end = time.time()
            multi_time = end - start
            print("MPI took {0:.1f} seconds".format(multi_time))

        return sampler
