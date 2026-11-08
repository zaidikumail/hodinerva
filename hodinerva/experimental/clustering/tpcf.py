import numpy as np
from astropy.io import ascii
from astropy.table import Column, Table
from pycorr import (
    KMeansSubsampler,
    TwoPointCorrelationFunction,
    mpi,
    setup_logging,
)

# To activate logging
setup_logging()


def create_subsampler(data_radec, nsamples):
    mpicomm = mpi.COMM_WORLD
    mpiroot = 0

    if mpicomm.rank == mpiroot:
        data_radec = list(data_radec)
    else:
        data_radec = None

    subsampler = KMeansSubsampler(
        mode="angular",
        positions=data_radec,
        nsamples=nsamples,
        nside=5000,
        random_state=42,
        position_type="rd",
        mpicomm=mpicomm,
        mpiroot=mpiroot,
    )
    data_samples = subsampler.label(data_radec)
    subsampler.log_info(
        "Labels from {:d} to {:d}.".format(data_samples.min(), data_samples.max())
    )
    return subsampler, data_samples, mpicomm, mpiroot


def evaluate_tpcf(
    data_radec,
    ran_radec,
    subsampler,
    bin_edges,
    mpicomm,
    mpiroot,
    savedir,
    savename,
    R1R2=None,
):
    if mpicomm.rank == mpiroot:
        data_radec = list(data_radec)
        ran_radec = list(ran_radec)
    else:
        data_radec, ran_radec = None, None

    if mpicomm.rank == mpiroot:
        data_samples = subsampler.label(data_radec)
        subsampler.log_info(
            "Labels from {:d} to {:d}.".format(data_samples.min(), data_samples.max())
        )

        randoms_samples = subsampler.label(ran_radec)
        subsampler.log_info(
            "Labels from {:d} to {:d}.".format(
                randoms_samples.min(), randoms_samples.max()
            )
        )
    else:
        data_samples, randoms_samples = None, None

    result = TwoPointCorrelationFunction(
        "theta",
        bin_edges,
        data_positions1=data_radec,
        randoms_positions1=ran_radec,
        data_samples1=data_samples,
        randoms_samples1=randoms_samples,
        R1R2=R1R2,
        engine="corrfunc",
        compute_sepsavg=False,
        estimator="landyszalay",
        position_type="rd",
        nthreads=5,
        mpicomm=mpicomm,
        mpiroot=mpiroot,
        nprocs_per_real=2,
    )

    save_tpcf = savedir + "/tpcf/" + savename + ".npy"
    result.save(save_tpcf)

    save_cov = savedir + "/cov/cov_" + savename + ".npy"
    save_w_theta = savedir + "/w_theta/w_" + savename + ".dat"

    save_cov_and_w_theta(
        result.sep,
        result.corr,
        result.cov(),
        result.D1D2.normalized_wcounts(),
        result.D1R2.normalized_wcounts(),
        result.R1R2.normalized_wcounts(),
        result.D1D2.size1,
        save_cov,
        save_w_theta,
    )

    return result


def save_cov_and_w_theta(sep, corr, cov, DD, DR, RR, N, save_cov, save_w_theta):
    np.save(save_cov, cov)
    n_scales = len(sep)
    std = np.sqrt(np.diagonal(cov))

    w_theta = Table()
    w_theta.add_column(Column(name="sep[deg]", length=n_scales))
    w_theta["sep[deg]"] = sep
    w_theta["sep[deg]"].info.format = "11.10f"

    w_theta.add_column(Column(name="DD", length=n_scales))
    w_theta["DD"] = DD
    w_theta["DD"].info.format = "11.10f"

    w_theta.add_column(Column(name="DR", length=n_scales))
    w_theta["DR"] = DR
    w_theta["DR"].info.format = "11.10f"

    w_theta.add_column(Column(name="RR", length=n_scales))
    w_theta["RR"] = RR
    w_theta["RR"].info.format = "11.10f"

    w_theta.add_column(Column(name="corr", length=n_scales))
    w_theta["corr"] = corr
    w_theta["corr"].info.format = "11.10f"

    w_theta.add_column(Column(name="std", length=n_scales))
    w_theta["std"] = std
    w_theta["std"].info.format = "11.10f"

    w_theta.add_column(Column(name="N", length=n_scales, dtype="int"))
    w_theta["N"] = N

    ascii.write(
        w_theta,
        save_w_theta,
        format="fixed_width",
        delimiter=" ",
        overwrite=True,
    )
