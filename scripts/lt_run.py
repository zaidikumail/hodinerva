import os
import sys

import emcee
import numpy as np
from astropy.cosmology import Planck13
from astropy.io import ascii
from pycorr import TwoPointCorrelationFunction, setup_logging

from hodinerva.experimental.fithod import Fithod

os.environ["OMP_NUM_THREADS"] = "1"
setup_logging()

####################################
field = "astrodeep"
nsteps = 14000
restart = 0
run = "run6/"

mock_errors = "yes"  #'yes' or 'no'
mock_hod_runid = "5"  # runid based on which mock angular covariance was calculated
N_samples = 1000  # Number of mocks used to get covariance
###################################

hod_model = "Leauthaud11"
hod_params = {"redshift": 0, "sm_thresh": 9, "central": False}
nwalkers = 60
ndim = 11

hod_post_initialpos = ascii.read(
    "/cluster/tufts/marchesini_lab/mzaidi01/Clustering/halomod/lt_hod_post/hod_post_126+.ecsv"
)


######################################################################

# cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0474, Tcmb0=2.7255)
cosmo = Planck13

coverage = ascii.read(
    "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/DB_fits/ASTRODEEP_Coverage.dat"
)

uds_area_deg2 = (
    coverage[coverage["Field"] == "PRIMER-UDS"]["Area[arcmin2]"].data[0] / 3600
)
cosmos_area_deg2 = (
    coverage[coverage["Field"] == "PRIMER-COSMOS"]["Area[arcmin2]"].data[0] / 3600
)
total_area_deg2 = np.sum(coverage["Area[arcmin2]"]) / 3600


uds_area = coverage[coverage["Field"] == "PRIMER-UDS"]["Area[arcmin2]"].data[0]
ceers_area = coverage[coverage["Field"] == "CEERS"]["Area[arcmin2]"].data[0]
cosmos_area = coverage[coverage["Field"] == "PRIMER-COSMOS"]["Area[arcmin2]"].data[0]
gn_area = coverage[coverage["Field"] == "JADES-GN"]["Area[arcmin2]"].data[0]
gs_area = coverage[coverage["Field"] == "JADES-GS"]["Area[arcmin2]"].data[0]
ngdeep_area = coverage[coverage["Field"] == "NGDEEP"]["Area[arcmin2]"].data[0]
total_area = np.sum(coverage["Area[arcmin2]"])

####################################################################################################

zbin = int(sys.argv[1][1])
print(zbin)

zbins = np.load("/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/zbins.npy")
Mthresh = np.load(
    "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/Mthresh_DB.npy",
    allow_pickle=True,
)

samples = []
for i in range(0, len(Mthresh[zbin - 1]) - 1):
    samples.append(sys.argv[1] + "_m" + str(i + 1))

####################################################################################################

if field == "astrodeep":
    uds_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/PRIMER-UDS_DB/tpcf_npy/RR_IC_tpcf.npy"
    )
    ceers_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/CEERS_DB/tpcf_npy/RR_IC_tpcf.npy"
    )
    cosmos_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/PRIMER-COSMOS_DB/tpcf_npy/RR_IC_tpcf.npy"
    )
    gn_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/JADES-GN_DB/tpcf_npy/RR_IC_tpcf.npy"
    )
    gs_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/JADES-GS_DB/tpcf_npy/RR_IC_tpcf.npy"
    )
    ngdeep_RR_IC_tpcf = TwoPointCorrelationFunction.load(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/NGDEEP_DB/tpcf_npy/RR_IC_tpcf.npy"
    )

    sep_RR_IC = uds_RR_IC_tpcf.sep
    RR_IC = (
        ((uds_area / total_area) * uds_RR_IC_tpcf.R1R2.normalized_wcounts())
        + ((ceers_area / total_area) * ceers_RR_IC_tpcf.R1R2.normalized_wcounts())
        + ((cosmos_area / total_area) * cosmos_RR_IC_tpcf.R1R2.normalized_wcounts())
        + ((gn_area / total_area) * gn_RR_IC_tpcf.R1R2.normalized_wcounts())
        + ((gs_area / total_area) * gs_RR_IC_tpcf.R1R2.normalized_wcounts())
        + ((ngdeep_area / total_area) * ngdeep_RR_IC_tpcf.R1R2.normalized_wcounts())
    )

    w_data_ascii = (
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/ASTRODEEP_DB/w_data/"
        + samples[0]
        + ".dat"
    )

    if mock_errors == "yes":
        mock_out = (
            "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_outs/mock_out_ASTRODEEP_hod"
            + mock_hod_runid
            + "_"
            + samples[0]
            + "_N"
            + str(N_samples)
            + ".npy"
        )
        cov_jknife_npy = None
    else:
        cov_jknife_npy = (
            "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/ASTRODEEP_DB/cov_data/"
            + samples[0]
            + ".npy"
        )
        mock_out = None

    print(w_data_ascii)
    Nz_npy = (
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/ASTRODEEP_DB/Nz/N_"
        + samples[0][:2]
        + ".npy"
    )
    print(Nz_npy)
    SMF_data_ascii = (
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/SMFs/ASTRODEEP_SMF_"
        + samples[0][:2]
        + ".dat"
    )

    # Initialize the Fithod class
    fit_hod = Fithod(
        zbins[zbin - 1][0],
        zbins[zbin - 1][1],
        Nz_npy,
        RR_IC,
        sep_RR_IC,
        hod_model,
        hod_params,
        cosmo,
    )

    # w_data
    w_datum = fit_hod.setup_w_data(
        sm_thresh=Mthresh[zbin - 1][0],
        w_data_ascii=w_data_ascii,
        cov_jknife_npy=cov_jknife_npy,
        mock_out=mock_out,
    )
    w_data = [w_datum]
    for s in range(1, len(samples)):
        w_data_ascii = (
            "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/ASTRODEEP_DB/w_data/"
            + samples[s]
            + ".dat"
        )
        print(w_data_ascii)
        if mock_errors == "yes":
            mock_out = (
                "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_outs/mock_out_ASTRODEEP_hod"
                + mock_hod_runid
                + "_"
                + samples[s]
                + "_N"
                + str(N_samples)
                + ".npy"
            )
            cov_jknife_npy = None
        else:
            cov_jknife_npy = (
                "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/ASTRODEEP_DB/cov_data/"
                + samples[s]
                + ".npy"
            )
            mock_out = None

        w_datum = fit_hod.setup_w_data(
            sm_thresh=Mthresh[zbin - 1][s],
            w_data_ascii=w_data_ascii,
            cov_jknife_npy=cov_jknife_npy,
            mock_out=mock_out,
        )
        w_data.append(w_datum)
    fit_hod.w_data = w_data

    # SMF data
    fit_hod.setup_smf_data(SMF_data_ascii=SMF_data_ascii)

    # model
    fit_hod.setup_model(w_data_ascii=w_data_ascii)
####################################################################################################

blobs_dtype = []
blobs_dtype.append(("chi_sq", float))
fit_hod.blobs_dtype = blobs_dtype

#####################################################################################################

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
    hod_post_initialpos["smhm_m0_0"][zbin - 1][0][1],  # M0
    hod_post_initialpos["smhm_m1_0"][zbin - 1][0][1],  # M1
    hod_post_initialpos["smhm_beta_0"][zbin - 1][0][1],  # beta
    hod_post_initialpos["smhm_delta_0"][zbin - 1][0][1],  # delta
    hod_post_initialpos["smhm_gamma_0"][zbin - 1][0][1],  # gamma
    hod_post_initialpos["sig_logmstar"][zbin - 1][0][1],  # sig_logmstar
    hod_post_initialpos["alphasat"][zbin - 1][0][1],  # alphasat
    hod_post_initialpos["betasat"][zbin - 1][0][1],  # betasat
    hod_post_initialpos["bsat"][zbin - 1][0][1],  # bsat
    hod_post_initialpos["betacut"][zbin - 1][0][1],  # betacut
    hod_post_initialpos["bcut"][zbin - 1][0][1],  # bcut
]


# fit_hod.initialpos = np.array([10.72, 12.35, 0.43, 0.56, 1.54, 0.206, 1.0, 0.859, 10.62,  -0.13, 1.47 ])

####################################################################################################
backendfile = (
    "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/halomod/lt_chains/"
    + run
    + sys.argv[1]
    + "_backend.h5"
)

fit_hod.backend = emcee.backends.HDFBackend(backendfile)
sampler = fit_hod.run_mcmc(nwalkers=nwalkers, ndim=ndim, nsteps=nsteps, restart=restart)
