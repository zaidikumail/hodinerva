import sys
import time
import warnings
from pathlib import Path

import emcee
import numpy as np
from astropy.io import ascii
from numpy.linalg import pinv
from schwimmbad import MPIPool

from ..build_model import build_acf_model, build_smf_data
from .loss import chi_sq


class Fithod:
    def __init__(
        self,
        zmin,
        zmax,
        Nz_npy,
        rr_ic_counts,
        rr_ic_sep,
        hod_model,
        hod_params,
        cosmo,
        smf_data_fits,
        smf_z_name,
    ):
        self.zmin = zmin
        self.zmax = zmax
        self.Nz_npy = Nz_npy
        self.rr_ic_counts = rr_ic_counts
        self.rr_ic_sep = rr_ic_sep
        self.hod_model = hod_model
        self.hod_params = hod_params
        self.cosmo = cosmo
        self.smf_data_fits = smf_data_fits
        self.smf_z_name = smf_z_name

    def setup_w_data(self, sm_thresh, w_data_ascii, cov_jknife_npy=None, mock_out=None):
        """Get correlation function data"""
        w_data = ascii.read(w_data_ascii)

        sep = w_data["sep[deg]"].data
        self.sep = sep

        corr = w_data["corr"].data
        std = w_data["std"].data
        N = w_data["N"][0]

        if isinstance(mock_out, Path):
            # mock covariance
            Cov = np.load(mock_out, allow_pickle=True).item()["Cov"]
            C_inv = pinv(Cov)

        elif isinstance(cov_jknife_npy, Path):
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

    def setup_smf_data(self):
        self.smf_data = build_smf_data(self.smf_data_fits, self.cosmo, self.smf_z_name)

    def setup_model(self, sep):
        self.model = build_acf_model(
            sep,
            self.Nz_npy,
            self.cosmo,
            self.hod_model,
            self.hod_params,
            self.zmin,
            self.zmax,
        )

    def chi_square(self):
        return chi_sq(
            self.model,
            self.w_data,
            self.rr_ic_sep,
            self.rr_ic_counts,
            self.smf_data_fits,
            self.smf_data,
            self.smf_z_name,
            self.cosmo,
        )

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

    def run_mcmc(self, nwalkers=20, ndim=5, nsteps=10, restart=False):
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

            if restart:
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
