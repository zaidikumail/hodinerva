from pycorr import (
    KMeansSubsampler,
    TwoPointCorrelationFunction,
    mpi,
    setup_logging,
)

# To activate logging
setup_logging()


def create_subsampler(cat, nsamples):
    mpicomm = mpi.COMM_WORLD
    mpiroot = 0

    if mpicomm.rank == mpiroot:
        data_radec = cat["RA"].data, cat["DEC"].data
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


def evaluate_tpcf(
    cat,
    subsampler,
    bin_edges,
    rand_RA,
    rand_DEC,
    mpicomm,
    mpiroot,
    savedir,
    savename,
    R1R2=None,
    cat2=[],
):
    if mpicomm.rank == mpiroot:
        data_radec = cat["RA"].data, cat["DEC"].data
        data_radec = list(data_radec)

        if len(cat2) != 0:
            data_radec_2 = cat2["RA"].data, cat2["DEC"].data
            data_radec_2 = list(data_radec_2)

        rand_radec = rand_RA, rand_DEC
        rand_radec = list(rand_radec)
    else:
        data_radec, rand_radec = None, None

        if len(cat2) != 0:
            data_radec_2 = None

    if mpicomm.rank == mpiroot:
        data_samples = subsampler.label(data_radec)
        subsampler.log_info(
            "Labels from {:d} to {:d}.".format(data_samples.min(), data_samples.max())
        )

        if len(cat2) != 0:
            data_samples_2 = subsampler.label(data_radec_2)
            subsampler.log_info(
                "Labels from {:d} to {:d}.".format(
                    data_samples_2.min(), data_samples_2.max()
                )
            )

        randoms_samples = subsampler.label(rand_radec)
        subsampler.log_info(
            "Labels from {:d} to {:d}.".format(
                randoms_samples.min(), randoms_samples.max()
            )
        )
    else:
        data_samples, randoms_samples = None, None

        if len(cat2) != 0:
            data_samples_2 = None

    if R1R2 is None:
        if len(cat2) == 0:
            result = TwoPointCorrelationFunction(
                "theta",
                bin_edges,
                data_positions1=data_radec,
                randoms_positions1=rand_radec,
                data_samples1=data_samples,
                randoms_samples1=randoms_samples,
                engine="corrfunc",
                compute_sepsavg=False,
                estimator="landyszalay",
                position_type="rd",
                nthreads=64,
                mpicomm=mpicomm,
                mpiroot=mpiroot,
                nprocs_per_real=2,
            )
        else:
            result = TwoPointCorrelationFunction(
                "theta",
                bin_edges,
                data_positions1=data_radec,
                data_positions2=data_radec_2,
                randoms_positions1=rand_radec,
                randoms_positions2=rand_radec,
                data_samples1=data_samples,
                data_samples2=data_samples_2,
                randoms_samples1=randoms_samples,
                randoms_samples2=randoms_samples,
                engine="corrfunc",
                compute_sepsavg=False,
                estimator="landyszalay",
                position_type="rd",
                nthreads=64,
                mpicomm=mpicomm,
                mpiroot=mpiroot,
                nprocs_per_real=2,
            )
    else:
        if len(cat2) == 0:
            result = TwoPointCorrelationFunction(
                "theta",
                bin_edges,
                data_positions1=data_radec,
                randoms_positions1=rand_radec,
                data_samples1=data_samples,
                randoms_samples1=randoms_samples,
                R1R2=R1R2,
                engine="corrfunc",
                compute_sepsavg=False,
                estimator="landyszalay",
                position_type="rd",
                nthreads=64,
                mpicomm=mpicomm,
                mpiroot=mpiroot,
                nprocs_per_real=2,
            )

        else:
            result = TwoPointCorrelationFunction(
                "theta",
                bin_edges,
                data_positions1=data_radec,
                data_positions2=data_radec_2,
                randoms_positions1=rand_radec,
                randoms_positions2=rand_radec,
                data_samples1=data_samples,
                data_samples2=data_samples_2,
                randoms_samples1=randoms_samples,
                randoms_samples2=randoms_samples,
                R1R2=R1R2,
                engine="corrfunc",
                compute_sepsavg=False,
                estimator="landyszalay",
                position_type="rd",
                nthreads=64,
                mpicomm=mpicomm,
                mpiroot=mpiroot,
                nprocs_per_real=2,
            )

    result.save(savedir + "/" + savename + ".npy")

    return result
