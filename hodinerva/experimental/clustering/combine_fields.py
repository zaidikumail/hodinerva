import numpy as np
from pycorr import TwoPointCorrelationFunction

from .tpcf import save_cov_and_w_theta


def get_field_weighted_corr(sep, DDs, DRs, RRs, weights):
    n_sep = len(sep)

    DD = np.zeros(n_sep)
    DR = np.zeros(n_sep)
    RR = np.zeros(n_sep)

    for field in range(0, len(weights)):
        DD += weights[field] * DDs[field]
        DR += weights[field] * DRs[field]
        RR += weights[field] * RRs[field]

    corr = (DD - (2 * (DR)) + RR) / RR

    return (
        sep,
        DD,
        DR,
        RR,
        corr,
    )


def get_combined_corr_data(field_bns, sample, weights):
    tpcf = TwoPointCorrelationFunction.load(field_bns[0] + "/tpcf/" + sample)
    sep = tpcf.sep
    n_sep = len(sep)

    DD = np.zeros(n_sep)
    DR = np.zeros(n_sep)
    RR = np.zeros(n_sep)

    DDs = []
    DRs = []
    RRs = []
    N = 0
    for field in range(len(field_bns)):
        tpcf = TwoPointCorrelationFunction.load(field_bns[field] + "/tpcf/" + sample)

        DDs.append(tpcf.D1D2.normalized_wcounts())
        DRs.append(tpcf.D1R2.normalized_wcounts())
        RRs.append(tpcf.R1R2.normalized_wcounts())

        N += tpcf.D1D2.size1

    (
        sep,
        DD,
        DR,
        RR,
        corr,
    ) = get_field_weighted_corr(sep, DDs, DRs, RRs, weights)

    return (
        sep,
        DD,
        DR,
        RR,
        corr,
        N,
    )


def get_combined_cov(field_bns, sample, weights):
    tpcf = TwoPointCorrelationFunction.load(field_bns[0] + "/tpcf/" + sample)
    n_sep = len(tpcf.sep)
    cov = np.zeros((n_sep, n_sep))
    w_tot = sum(weights)
    for field in range(len(field_bns)):
        tpcf = TwoPointCorrelationFunction.load(field_bns[field] + "/tpcf/" + sample)
        cov += (weights[field] ** 2) * tpcf.cov()
    cov /= w_tot**2
    return cov


def get_combined_nz(field_bns, sample, weights):
    Nz = np.load(field_bns[0] + "/nz/nz_" + sample)
    zgrid = Nz[:, 0]
    nz = Nz[:, 1] * weights[0]

    for s in range(1, len(weights)):
        Nz = np.load(field_bns[s] + "/nz/nz_" + sample)

        nz += Nz[:, 1] * weights[s]

    nz = np.vstack((zgrid, nz)).T

    return nz


def combine_tpcf_fields(field_bns, samples, weights, savedir):
    for sample in samples:
        (
            sep,
            DD,
            DR,
            RR,
            corr,
            N,
        ) = get_combined_corr_data(field_bns, sample, weights)
        cov = get_combined_cov(field_bns, sample, weights)
        nz = get_combined_nz(field_bns, sample, weights)

        save_cov = savedir + "/cov/cov_" + sample
        save_w_theta = savedir + "/w_theta/w_" + sample.split(".")[0] + ".dat"
        save_nz = savedir + "/nz/nz_" + sample

        save_cov_and_w_theta(sep, corr, cov, DD, DR, RR, N, save_cov, save_w_theta)
        np.save(save_nz, nz)
