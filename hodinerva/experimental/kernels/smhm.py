import os

import emcee
import numpy as np
from halomod.hod import Leauthaud11

h0p7 = 0.7
HALO_MASS_H1P0 = np.logspace(9, 16, 1000)  # Msol/h units
HALO_MASS_H0P7 = HALO_MASS_H1P0 / h0p7


def med_cen_smhm(chain_dir, mstar_thresh, run_label, savedir):
    chains = sorted(os.listdir(chain_dir))
    n_chains = len(chains)
    sampler = emcee.backends.HDFBackend(chain_dir + "/" + chains[0], read_only=True)

    chain_z1 = sampler.get_chain(flat=False)
    n_walkers = np.shape(chain_z1)[1]

    med_cen_mstar = np.zeros((n_chains, n_walkers, len(HALO_MASS_H1P0)))
    med_cen_mstar_med = np.zeros((n_chains, len(HALO_MASS_H1P0)))
    med_cen_mstar_low = np.zeros((n_chains, len(HALO_MASS_H1P0)))
    med_cen_mstar_upp = np.zeros((n_chains, len(HALO_MASS_H1P0)))

    log_Mhpeak = np.zeros((n_chains, n_walkers))
    log_Mhpeak_med = np.zeros(n_chains)
    log_Mhpeak_low = np.zeros(n_chains)
    log_Mhpeak_upp = np.zeros(n_chains)

    for zbin in range(n_chains):
        sampler = emcee.backends.HDFBackend(
            chain_dir + "/" + chains[zbin], read_only=True
        )
        chain_z = sampler.get_chain(flat=False)

        # sm_thresh_h1p0 = mstar_thresh[zbin][0] + np.log10(h0p7**2)

        for walker in range(n_walkers):
            LT = Leauthaud11(
                redshift=0.0,
                smhm_m0_0=chain_z[-1, walker, 0],
                smhm_m1_0=chain_z[-1, walker, 1],
                smhm_beta_0=chain_z[-1, walker, 2],
                smhm_delta_0=chain_z[-1, walker, 3],
                smhm_gamma_0=chain_z[-1, walker, 4],
                sig_logmstar=chain_z[-1, walker, 5],
                alphasat=chain_z[-1, walker, 6],
                betasat=chain_z[-1, walker, 7],
                bsat=chain_z[-1, walker, 8],
                betacut=chain_z[-1, walker, 9],
                bcut=chain_z[-1, walker, 10],
            )

            log_mstar_h1p0 = LT.mean_log_stellar_mass(HALO_MASS_H1P0)
            mstar_h0p7 = (10**log_mstar_h1p0) / (h0p7**2)

            med_cen_mstar[zbin][walker] = mstar_h0p7
            log_Mhpeak[zbin][walker] = np.log10(
                HALO_MASS_H0P7[np.argmax(mstar_h0p7 / HALO_MASS_H0P7)]
            )

        med_cen_mstar_med[zbin] = np.median(med_cen_mstar[zbin], axis=0)
        med_cen_mstar_low[zbin] = np.percentile(med_cen_mstar[zbin], 16, axis=0)
        med_cen_mstar_upp[zbin] = np.percentile(med_cen_mstar[zbin], 84, axis=0)

        log_Mhpeak_med[zbin] = np.median(log_Mhpeak[zbin], axis=0)
        log_Mhpeak_low[zbin] = np.percentile(log_Mhpeak[zbin], 16, axis=0)
        log_Mhpeak_upp[zbin] = np.percentile(log_Mhpeak[zbin], 84, axis=0)

    path = os.path.join(savedir, run_label)
    os.makedirs(path, exist_ok=True)

    np.save(
        path + "/" + run_label + "_med_cen_mstar_med",
        med_cen_mstar_med,
    )
    np.save(
        path + "/" + run_label + "_med_cen_mstar_low",
        med_cen_mstar_low,
    )
    np.save(
        path + "/" + run_label + "_med_cen_mstar_upp",
        med_cen_mstar_upp,
    )
    np.save(
        path + "/" + run_label + "_log_Mhpeak_med",
        log_Mhpeak_med,
    )
    np.save(
        path + "/" + run_label + "_log_Mhpeak_low",
        log_Mhpeak_low,
    )
    np.save(
        path + "/" + run_label + "_log_Mhpeak_upp",
        log_Mhpeak_upp,
    )
