import argparse
import os
from pathlib import Path

import emcee
import numpy as np
import yaml
from astropy.io import ascii
from pycorr import TwoPointCorrelationFunction, setup_logging

from hodinerva.experimental.defaults import DEFAULT_COSMOLOGY, HOD_INITIALIZE_PARAMS
from hodinerva.experimental.fitting_kernels.fithod import Fithod

os.environ["OMP_NUM_THREADS"] = "1"
setup_logging()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    zbins = np.load(Path(cfg["base_path"]) / cfg["zbins"])
    mstar_thresh = np.load(
        Path(cfg["base_path"]) / cfg["mstar_thresh"],
        allow_pickle=True,
    )
    hod_post_initialpos = ascii.read(
        Path(cfg["base_path"]) / cfg["hod_post_initialpos"]
    )

    zbin = cfg["zbin"]
    samples = []
    for m in range(0, len(mstar_thresh[zbin])):
        samples.append("z" + str(zbin + 1) + "_m" + str(m + 1))

    ####################################################################################

    rr_ic = TwoPointCorrelationFunction.load(Path(cfg["base_path"]) / cfg["rr_ic"])
    rr_ic_sep = rr_ic.sep
    rr_ic_counts = rr_ic.R1R2.normalized_wcounts()

    w_data_ascii = Path(cfg["base_path"]) / cfg["w_data_dir"] / f"w_{samples[0]}.dat"
    print(w_data_ascii)

    if cfg["mock_errors"]:
        mock_out = Path(cfg["base_path"]) / cfg["mock_out_npy"]
        cov_jknife_npy = None
    else:
        cov_jknife_npy = (
            Path(cfg["base_path"]) / cfg["cov_jknife_dir"] / f"cov_{samples[0]}.npy"
        )
        mock_out = None

    nz_file = Path(cfg["base_path"]) / cfg["nz"]
    print(nz_file)

    # Initialize the Fithod class
    fit_hod = Fithod(
        zbins[zbin - 1][0],
        zbins[zbin - 1][1],
        nz_file,
        rr_ic_counts,
        rr_ic_sep,
        cfg["hod_model"],
        HOD_INITIALIZE_PARAMS,
        DEFAULT_COSMOLOGY,
    )

    # w data
    w_datum = fit_hod.setup_w_data(
        sm_thresh=mstar_thresh[zbin - 1][0],
        w_data_ascii=w_data_ascii,
        cov_jknife_npy=cov_jknife_npy,
        mock_out=mock_out,
    )
    w_data = [w_datum]
    for s in range(1, len(samples)):
        w_data_ascii = (
            Path(cfg["base_path"]) / cfg["w_data_dir"] / f"w_{samples[s]}.dat"
        )

        if cfg["mock_errors"]:
            mock_out = Path(cfg["base_path"]) / cfg["mock_out_npy"]
            cov_jknife_npy = None
        else:
            cov_jknife_npy = (
                Path(cfg["base_path"]) / cfg["cov_jknife_dir"] / f"cov_{samples[s]}.npy"
            )
            mock_out = None

        w_datum = fit_hod.setup_w_data(
            sm_thresh=mstar_thresh[zbin][s],
            w_data_ascii=w_data_ascii,
            cov_jknife_npy=cov_jknife_npy,
            mock_out=mock_out,
        )
        w_data.append(w_datum)
    fit_hod.w_data = w_data

    # smf data
    smf_data_fits = Path(cfg["base_path"]) / cfg["smf_data"]
    z_name = cfg["z_name"]
    fit_hod.setup_smf_data(smf_data_fits=smf_data_fits, z_name=z_name)

    # model
    w_data = ascii.read(w_data_ascii)
    sep = w_data["sep[deg]"].data
    fit_hod.setup_model(sep=sep)
    ####################################################################################

    blobs_dtype = []
    blobs_dtype.append(("chi_sq", float))
    fit_hod.blobs_dtype = blobs_dtype

    ####################################################################################

    fit_hod.param_names = [
        "hod_params.smhm_m0_0",
        "hod_params.smhm_m1_0",
        "hod_params.smhm_beta_0",
        "hod_params.smhm_delta_0",
        "hod_params.smhm_gamma_0",
        "hod_params.sig_logmstar",
        "hod_params.alphasat",
        "hod_params.betasat",
        "hod_params.bsat",
        "hod_params.betacut",
        "hod_params.bcut",
    ]

    fit_hod.hod_bounds = {
        "hod_params.smhm_m0_0": (10, 12),
        "hod_params.smhm_m1_0": (11.8, 14),
        "hod_params.smhm_beta_0": (0.1, 0.9),  # 0.9
        "hod_params.smhm_delta_0": (0.1, 4),
        "hod_params.smhm_gamma_0": (0, 8),  # (0.1, 4),
        "hod_params.sig_logmstar": (0.001, 0.5),
        "hod_params.alphasat": (0.4, 2),
        "hod_params.betasat": (0.01, 4),
        "hod_params.bsat": (1, 17),
        "hod_params.betacut": (-1, 8),
        "hod_params.bcut": (0, 14),
    }

    fit_hod.hod_derived = []

    fit_hod.initialpos = [
        hod_post_initialpos["smhm_m0_0"][zbin][0][1],  # M0
        hod_post_initialpos["smhm_m1_0"][zbin][0][1],  # M1
        hod_post_initialpos["smhm_beta_0"][zbin][0][1],  # beta
        hod_post_initialpos["smhm_delta_0"][zbin][0][1],  # delta
        hod_post_initialpos["smhm_gamma_0"][zbin][0][1],  # gamma
        hod_post_initialpos["sig_logmstar"][zbin][0][1],  # sig_logmstar
        hod_post_initialpos["alphasat"][zbin][0][1],  # alphasat
        hod_post_initialpos["betasat"][zbin][0][1],  # betasat
        hod_post_initialpos["bsat"][zbin][0][1],  # bsat
        hod_post_initialpos["betacut"][zbin][0][1],  # betacut
        hod_post_initialpos["bcut"][zbin][0][1],  # bcut
    ]

    # fit_hod.initialpos = np.array(
    #     [10.72, 12.35, 0.43, 0.56, 1.54, 0.206, 1.0, 0.859, 10.62, -0.13, 1.47]
    # )

    ####################################################################################
    backend_h5 = Path(cfg["base_path"]) / cfg["backend_h5"]

    fit_hod.backend = emcee.backends.HDFBackend(backend_h5)
    sampler = fit_hod.run_mcmc(
        nwalkers=cfg["n_walkers"],
        ndim=cfg["n_dim"],
        nsteps=cfg["n_steps"],
        restart=cfg["restart"],
    )
