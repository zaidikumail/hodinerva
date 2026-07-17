import os
import sys

import numpy as np
from astropy.cosmology import Planck13
from astropy.io import ascii
from halotools.mock_observables import (
    return_xyz_formatted_array,
    tpcf,
    tpcf_one_two_halo_decomp,
)
from halotools.sim_manager import CachedHaloCatalog
from mock_angular_cov import mock_angular_cov

sys.path.append(
    os.path.abspath(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/htools_mock"
    )
)  # add parent directory to search path

# cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115, Ob0=0.048206)
cosmo = Planck13

hod_runid = "5"
N_samples = 1000
hod_post = ascii.read("../halomod/lt_hod_post/hod_post_" + hod_runid + ".ecsv")

coverage = ascii.read(
    "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/DB_fits/ASTRODEEP_Coverage.dat"
)
zbins = np.load("../zbins.npy")
Mthresh = np.load("../Mthresh_DB.npy", allow_pickle=True)
smdpl_z = np.load("smdpl_z.npy")

z_idx = int(sys.argv[1])
print(smdpl_z[z_idx])

redshift = smdpl_z[z_idx]
mthresh = Mthresh[z_idx]
zlabel = "z" + str(z_idx + 1)

simname = "SMDPL"
halo_finder = "rockstar"
version_name = "rockstar_v1.0_mvir_no_cuts"
Lbox = 400  # Mpc/h
halocat = CachedHaloCatalog(
    simname=simname,
    halo_finder=halo_finder,
    version_name=version_name,
    redshift=redshift,
)

fields = ["PRIMER-UDS", "CEERS", "PRIMER-COSMOS", "JADES-GS", "NGDEEP"]


theta_bin_edges = np.logspace(-3, -0.9, 10)

for mth in range(0, len(mthresh)):
    sm_thresh = mthresh[mth]
    mlabel = "m" + str(mth + 1)

    mock_out = {}
    mock_out["sim"] = simname + "_" + version_name + "_" + str(Lbox)
    mock_out["subsample"] = zlabel + "_" + mlabel
    mock_out["theta_bin_edges"] = theta_bin_edges
    mock_out["N_samples"] = N_samples
    mock_out["fields"] = fields

    mock = mock_angular_cov(
        redshift=redshift,
        sm_thresh=sm_thresh,
        hod_post=hod_post,
        hod_runid=hod_runid,
        zlabel=zlabel,
        mlabel=mlabel,
        Lbox=Lbox,
        N_samples=N_samples,
        theta_bin_edges=theta_bin_edges,
        cosmo=cosmo,
    )

    hod = mock.populate_hod()
    halo_mass = np.logspace(7, 16, 100)
    mean_ncen = hod.mean_occupation_centrals(prim_haloprop=halo_mass)
    mean_nsat = hod.mean_occupation_satellites(
        prim_haloprop=halo_mass, modulate_with_cenocc=True
    )
    mean_ntot = mean_ncen + mean_nsat

    mock_out["halo_mass_h1p0"] = halo_mass
    mock_out["mean_ncen"] = mean_ncen
    mock_out["mean_nsat"] = mean_nsat

    """populate mock"""
    hod.populate_mock(
        halocat, Num_ptcl_requirement=50
    )  # default: Num_ptcl_requirement=300
    gals = hod.mock.galaxy_table
    log_ng = np.log10(len(gals) / (Lbox**3))
    print("log10(ng [[h3/Mpc3]]): " + str(log_ng))
    print(zbins[z_idx])
    mock.ra, mock.dec = mock.convert_xyz_to_radec(
        x=gals["x"], y=gals["y"], z=gals["z"], zbin=zbins[z_idx]
    )
    mock_out["ra"] = mock.ra
    mock_out["dec"] = mock.dec

    """generate random catalog"""
    num_points = len(gals["x"]) * 10
    ran_x = np.random.uniform(np.min(gals["x"]), np.max(gals["x"]), num_points)
    ran_y = np.random.uniform(np.min(gals["y"]), np.max(gals["y"]), num_points)
    ran_z = np.random.uniform(np.min(gals["z"]), np.max(gals["z"]), num_points)

    mock.ran_ra, mock.ran_dec = mock.convert_xyz_to_radec(
        x=ran_x, y=ran_y, z=ran_z, zbin=zbins[z_idx]
    )
    mock_out["ran_ra"] = mock.ran_ra
    mock_out["ran_dec"] = mock.ran_dec

    """compare xi between halomod and halotools"""
    pos = return_xyz_formatted_array(gals["x"], gals["y"], gals["z"])
    rbins = np.logspace(-1, 1.6, 16)
    rbin_centers = (rbins[1:] + rbins[:-1]) / 2.0

    xi_all = tpcf(
        pos, rbins, period=hod.mock.Lbox, estimator="Landy-Szalay", num_threads="max"
    )

    halo_hostid = gals["halo_hostid"]
    xi_1h, xi_2h = tpcf_one_two_halo_decomp(
        pos,
        halo_hostid,
        rbins,
        period=hod.mock.Lbox,
        estimator="Landy-Szalay",
        num_threads="max",
    )
    mock_out["rbin_edges"] = rbins
    mock_out["xi_all"] = xi_all
    mock_out["xi_1h"] = xi_1h
    mock_out["xi_2h"] = xi_2h

    # plt.plot(rbin_centers, xi_all, c='k', label='halotools')
    # plt.plot(rbin_centers, xi_1h, c='k', ls='-.')
    # plt.plot(rbin_centers, xi_2h, c='k', ls=':')

    # xi_halomod = ascii.read('/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/halomod/lt_hod_post/astrodeep_xi_z1_m'+str(mth+1)+'_3.dat')
    # plt.plot(xi_halomod['r[Mpc/h]'], xi_halomod['xi'], c='orange', label='halomod')
    # plt.plot(xi_halomod['r[Mpc/h]'], xi_halomod['xi_1h'], c='orange', ls='-.')
    # plt.plot(xi_halomod['r[Mpc/h]'], xi_halomod['xi_2h'], c='orange', ls=':')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(1e-4, 1e5)
    # plt.xlabel('r [Mpc/h]')
    # plt.ylabel('$\u03be$ (r)')
    # plt.title('log M$_{*}$ [M$_{\odot}$] > '+str(mthresh[mth]))
    # plt.legend()
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    # plt.savefig('./figures/tools_v_mod_z1_m'+str(mth+1)+'.pdf')
    # plt.show()

    """Plot the full mock ra dec ran_ra ran_dec"""
    # plt.plot(mock.ra, mock.dec,'.',color='red', markersize=0.01, rasterized=True)
    # plt.ylabel(r'$\delta$  $[{\rm degrees}]$', fontsize=12)
    # plt.xlabel(r'$\alpha$  $[{\rm degrees}]$', fontsize=12)
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.title('Mock catalog in angular coordinates', fontsize=12)
    # plt.show()

    # plt.plot(mock.ran_ra, mock.ran_dec,'.',color='red', markersize=0.01, rasterized=True)
    # plt.ylabel(r'$\delta$  $[{\rm degrees}]$', fontsize=12)
    # plt.xlabel(r'$\alpha$  $[{\rm degrees}]$', fontsize=12)
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.title('Random Mock catalog in angular coordinates', fontsize=12)
    # plt.show()

    """measure w(theta) field-by-field"""
    DDs = []
    DRs = []
    RRs = []
    for F in range(0, len(fields)):
        print(fields[F])
        area = coverage[coverage["Field"] == fields[F]]["Area[arcmin2]"].data[0] / 3600
        print("Area [deg2]: " + str(area))

        # capture this field's footprint and store RA/DEC etc. in mock. so that mock.evaluate_tpcf can be run
        mock.capture_footprint(field=fields[F])

        DDs_field, DRs_field, RRs_field, Ngals = mock.evaluate_field_tpcf(
            field=fields[F]
        )

        DDs.append(DDs_field)
        DRs.append(DRs_field)
        RRs.append(RRs_field)

        # Vol = (area/mock_tot_area_deg2)*(Lbox**3)

    #     for Ngal in Ngals:
    #         log_ng = np.log10(Ngal/Vol)
    #         print('log10(ng [[h3/Mpc3]]): '+str(log_ng))

    """get combined corrs and cov"""
    mock.get_combined_corr(fields=fields, coverage=coverage, DDs=DDs, DRs=DRs, RRs=RRs)

    mock_out["corrs_combined"] = mock.corrs_combined
    mock_out["Cov"] = mock.Cov

    np.save(
        "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_outs/mock_out_ASTRODEEP_hod"
        + hod_runid
        + "_"
        + zlabel
        + "_"
        + mlabel
        + "_N"
        + str(N_samples),
        mock_out,
    )

    # np.save('/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_outs/corrs_combined_ASTRODEEP_hod'+hod_runid+'_'+zlabel+'_'+mlabel+'_N'+str(N_samples), mock.corrs_combined)
    # np.save('/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_outs/Cov_ASTRODEEP_hod'+hod_runid+'_'+zlabel+'_'+mlabel+'_N'+str(N_samples), mock.Cov)

    """plot covariance"""
    # fig = plt.figure(figsize=(4, 3.25))
    # ax = plt.gca()
    # c = ax.pcolor(mock.theta_bin_centers, mock.theta_bin_centers, mock.Cov, cmap=plt.get_cmap('jet'))
    # fig.colorbar(c, ax=ax)
    # ax.set_title('Covariance')
    # ax.set_xlabel(r'$\theta$ [deg]')
    # ax.set_ylabel(r'$\theta$ [deg]')
    # fig.tight_layout()
    # plt.show()

    # mock.show_combined_corr()
