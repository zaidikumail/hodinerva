import matplotlib.patches as patches
import numpy as np
import proplot as pplt
from halotools.empirical_models import PrebuiltHodModelFactory
from matplotlib.path import Path
from proplot.gridspec import GridSpec as GSpec
from pycorr import TwoPointCorrelationFunction

# os.environ["OMP_NUM_THREADS"] = "1"
# %matplotlib inline


class mock_angular_cov:
    def __init__(
        self,
        redshift,
        sm_thresh,
        hod_post,
        hod_runid,
        zlabel,
        mlabel,
        Lbox,
        N_samples,
        theta_bin_edges,
        cosmo,
    ):
        self.redshift = redshift
        self.sm_thresh = sm_thresh
        self.hod_post = hod_post
        self.hod_runid = hod_runid
        self.zlabel = zlabel
        self.mlabel = mlabel
        self.Lbox = Lbox  # [Mpc/h]
        self.N_samples = N_samples
        self.theta_bin_edges = theta_bin_edges
        self.theta_bin_centers = 0.5 * (theta_bin_edges[1:] + theta_bin_edges[:-1])
        self.cosmo = cosmo

    def populate_hod(self):
        sm_thresh = self.sm_thresh + (2 * np.log10(self.cosmo.h))
        hod = PrebuiltHodModelFactory(
            "leauthaud11", threshold=sm_thresh, redshift=self.redshift
        )

        hod.param_dict["smhm_m0_0"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["smhm_m0_0"][0][0][1]
        hod.param_dict["smhm_m0_a"] = 0

        hod.param_dict["smhm_m1_0"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["smhm_m1_0"][0][0][1]
        hod.param_dict["smhm_m1_a"] = 0

        hod.param_dict["smhm_beta_0"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["smhm_beta_0"][0][0][1]
        hod.param_dict["smhm_beta_a"] = 0

        hod.param_dict["smhm_delta_0"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["smhm_delta_0"][0][0][1]
        hod.param_dict["smhm_delta_a"] = 0

        hod.param_dict["smhm_gamma_0"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["smhm_gamma_0"][0][0][1]
        hod.param_dict["smhm_gamma_a"] = 0

        hod.param_dict["scatter_model_param1"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["sig_logmstar"][0][0][1]

        hod.param_dict["alphasat"] = self.hod_post[
            self.hod_post["zbin"] == self.zlabel
        ]["alphasat"][0][0][1]

        hod.param_dict["betasat"] = self.hod_post[self.hod_post["zbin"] == self.zlabel][
            "betasat"
        ][0][0][1]

        hod.param_dict["bsat"] = self.hod_post[self.hod_post["zbin"] == self.zlabel][
            "bsat"
        ][0][0][1]

        hod.param_dict["betacut"] = self.hod_post[self.hod_post["zbin"] == self.zlabel][
            "betacut"
        ][0][0][1]

        hod.param_dict["bcut"] = self.hod_post[self.hod_post["zbin"] == self.zlabel][
            "bcut"
        ][0][0][1]

        return hod

    def convert_xyz_to_radec(self, x, y, z, zbin):
        """
        Convert (x, y, z) simulation coordinates to (RA, DEC) for an observer at redshift z_observer.

        Parameters:
            x, y, z : arrays
                Cartesian coordinates in Mpc/h.
            z_observer : float
                Redshift where the box is placed.
            zbin: float tuple
                (z1, z2) to match the Lbox to z-bin thickness of data

        Returns:
            RA, DEC : arrays
                Right Ascension (deg) and Declination (deg).
        """
        z_observer = self.redshift

        comoving_z1 = self.cosmo.comoving_distance(zbin[0]).value * self.cosmo.h
        comoving_z2 = self.cosmo.comoving_distance(zbin[1]).value * self.cosmo.h
        Ldata = comoving_z2 - comoving_z1
        D_Lbox = Ldata - self.Lbox

        print(Ldata / self.Lbox)
        if Ldata / self.Lbox > 1.15:
            # append a fraction of the box toward the end to reach the desired Lbox
            sel = (x <= D_Lbox) & (y <= D_Lbox) & (z <= D_Lbox)
            x = np.append(x, x[sel])
            y = np.append(y, y[sel])
            z = np.append(z, z[sel])
            self.Lbox = Ldata  # new box length after appending

        # First shift box to center at (0,0,0)
        x = x - self.Lbox / 2
        y = y - self.Lbox / 2
        z = z - self.Lbox / 2

        # 1 Compute comoving distance to redshift of the box
        delta_z = (
            self.cosmo.comoving_distance(z_observer).value * self.cosmo.h
        )  # in Mpc/h to be consistent with x,y,z

        x = x + delta_z

        # Compute radial distance from observer
        r = np.sqrt(x**2 + y**2 + z**2)

        # Compute spherical angles
        theta = np.arccos(z / r)  # Polar angle in radians
        phi = np.arctan2(y, x)  # Azimuthal angle in radians

        # Convert to RA and DEC
        RA = np.degrees(phi)  # Convert to degrees
        DEC = 90 - np.degrees(theta)  # Convert to degrees

        # Ensure RA is in the range [0, 360]
        # RA = np.mod(RA, 360)

        return RA, DEC

    def capture_footprint(self, field):
        polygon_regfile = (
            "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/weight_maps/"
            + field
            + "_edges.reg"
        )
        polygon_regfile_key = open(polygon_regfile, "r")
        polygon_regfile_lines = polygon_regfile_key.readlines()

        p = 3
        polygons = []
        while p < len(polygon_regfile_lines):
            polygon_line = polygon_regfile_lines[p]
            polygon_line = polygon_line[
                polygon_line.index("(") + 1 : polygon_line.index(")")
            ]
            polygon_coods = polygon_line.split(",")

            polygon_coods_formatted = []
            i = 0
            while i < len(polygon_coods):
                polygon_coods_formatted.append(
                    (float(polygon_coods[i]), float(polygon_coods[i + 1]))
                )
                i += 2
            polygons.append(polygon_coods_formatted)

            p += 1
        # print(polygons[0][0])
        min_tuple = np.min(polygons[0], axis=0)
        for poly in range(0, len(polygons)):
            polygons[poly] = polygons[poly] - min_tuple
            # print(polygons[poly])

        mock_tot_area_deg2 = (np.max(self.ra) - np.min(self.ra)) * (
            np.max(self.dec) - np.min(self.dec)
        )

        inside_polygons = []
        inside_polygons_ran = []
        poly_locator_N = []

        box_vertices = [
            (self.ra.min(), self.dec.min()),  # point 1
            (self.ra.min(), self.dec.max()),  # point 2
            (self.ra.max(), self.dec.max()),  # point 3
            (self.ra.max(), self.dec.min()),
            (self.ra.min(), self.dec.min()),  # back to start
        ]

        box_codes = [
            Path.MOVETO,  # move to first point
            Path.LINETO,  # line to second
            Path.LINETO,  # line to third
            Path.LINETO,
            Path.CLOSEPOLY,  # close polygon
        ]

        box_path = Path(box_vertices, box_codes)

        for s in range(0, self.N_samples):
            """
            Randomly shift the coverage polygons
            """
            delta_RA_max = np.median(self.ra) - np.min(self.ra) - 1
            delta_DEC_max = np.median(self.dec) - np.min(self.dec) - 1

            polygons_N_candidate_accept = False
            while polygons_N_candidate_accept == False:
                delta_RA = np.random.uniform(-delta_RA_max, delta_RA_max)
                delta_DEC = np.random.uniform(-delta_DEC_max, delta_DEC_max)
                polygons_N = []
                for poly in range(0, len(polygons)):
                    poly_candidate = [
                        (x + delta_RA, y + delta_DEC) for x, y in polygons[poly]
                    ]

                    # if polygon is not inside the box, try again with a new random shift
                    if np.all(box_path.contains_points(poly_candidate)) == False:
                        print(
                            "Randomly shifted polygon did not fall inside the simulation box. Trying again!"
                        )
                        break

                    polygons_N.append(poly_candidate)

                    if len(polygons_N) == len(polygons):
                        polygons_N_candidate_accept = True
            """
            Create patches based on each polygon in polygons_N and then select RA/DEC lying inside polygons
            """
            Patches = []
            path = Path(polygons_N[0])
            Patches.append(
                patches.PathPatch(path, facecolor="none", lw=0.5, edgecolor="red")
            )

            inside_polygons_N = path.contains_points(np.transpose([self.ra, self.dec]))
            inside_polygons_ran_N = path.contains_points(
                np.transpose([self.ran_ra, self.ran_dec])
            )

            for j in range(1, len(polygons)):
                path = Path(polygons_N[j])
                Patches.append(
                    patches.PathPatch(path, facecolor="none", lw=0.5, edgecolor="red")
                )

                inside_polygons_N += path.contains_points(
                    np.transpose([self.ra, self.dec])
                )
                inside_polygons_ran_N += path.contains_points(
                    np.transpose([self.ran_ra, self.ran_dec])
                )
            # print(np.sum(inside_polygons_N))
            """
            Write down current realizations's selected RA and DEC
            """
            inside_polygons.append(inside_polygons_N)
            inside_polygons_ran.append(inside_polygons_ran_N)

        self.inside_polygons = inside_polygons
        self.inside_polygons_ran = inside_polygons_ran
        self.poly_locator_N = poly_locator_N
        self.mock_tot_area_deg2 = mock_tot_area_deg2

    def evaluate_field_tpcf(self, field):
        """
        Get N mock correlation functions of a field
        """
        tpcfs = []
        # corrs_field = []
        DDs_field = []
        DRs_field = []
        RRs_field = []

        """get corrs"""
        Ngals = []
        """p loops through all the N_samples (mock realizations)"""
        for p in range(0, self.N_samples):
            Ngals.append(len(self.ra[self.inside_polygons[p]]))

            data_radec = (
                self.ra[self.inside_polygons[p]],
                self.dec[self.inside_polygons[p]],
            )
            data_radec = list(data_radec)
            rand_radec = (
                self.ran_ra[self.inside_polygons_ran[p]],
                self.ran_dec[self.inside_polygons_ran[p]],
            )
            rand_radec = list(rand_radec)

            tpcfs.append(
                TwoPointCorrelationFunction(
                    "theta",
                    self.theta_bin_edges,
                    data_positions1=data_radec,
                    randoms_positions1=rand_radec,
                    engine="corrfunc",
                    compute_sepsavg=False,
                    estimator="landyszalay",
                    position_type="rd",
                )
            )
            DDs_field.append(tpcfs[p].D1D2.normalized_wcounts())
            DRs_field.append(tpcfs[p].D1R2.normalized_wcounts())
            RRs_field.append(tpcfs[p].R1R2.normalized_wcounts())
            # corrs_field.append(tpcfs[p].corr)

        return DDs_field, DRs_field, RRs_field, Ngals

    def get_combined_corr(self, fields, coverage, DDs, DRs, RRs):
        total_area_arcmin2 = 0
        for j in range(0, len(fields)):
            total_area_arcmin2 += coverage[coverage["Field"] == fields[j]][
                "Area[arcmin2]"
            ].data[0]

        # print('Total area [arcmin2]:')
        # print(total_area_arcmin2)

        corrs_combined = []
        """
        index i runs through N_samples
        """
        for i in range(0, len(DDs[0])):
            DD_combined = np.zeros(len(DDs[0][0]))
            DR_combined = np.zeros(len(DRs[0][0]))
            RR_combined = np.zeros(len(RRs[0][0]))
            for f in range(0, len(fields)):
                weight = (
                    coverage[coverage["Field"] == fields[f]]["Area[arcmin2]"].data[0]
                    / total_area_arcmin2
                )
                DD_combined += weight * DDs[f][i]
                DR_combined += weight * DRs[f][i]
                RR_combined += weight * RRs[f][i]

            corr_combined = (
                DD_combined - (2 * (DR_combined)) + RR_combined
            ) / RR_combined
            corrs_combined.append(corr_combined)

        self.corrs_combined = corrs_combined
        self.mean_corr = np.nanmean(self.corrs_combined, axis=0)
        self.Cov = np.cov(self.corrs_combined, rowvar=False)
        self.error = np.sqrt(np.diagonal(self.Cov))

    def show_combined_corr(self):
        left = 5.5
        bottom = 4.5
        right = 0.75
        # top = 0.98
        fs = 14
        labelsize = 11
        pad = 0.01
        gs = GSpec(1, hspace=0, wspace=0)
        fig = pplt.figure(
            figsize=("89mm", "89mm"), left=left, right=right, bottom=bottom
        )
        ax = fig.add_subplot(gs[0])

        """p loops through all N_samples (mock realizations)"""
        for p in range(0, len(self.corrs_combined)):
            ax.plot(self.theta_bin_centers, self.corrs_combined[p], lw=0.2, c="orange")
        yticks = [1e-2, 1e-1, 1e0, 1e1]
        yticklabels = ["10$^{-2}$", "10$^{-1}$", "10$^{0}$", "10$^{1}$"]

        xticks = [1e-3, 1e-2, 1e-1]
        xticklabels = ["10$^{-3}$", "10$^{-2}$", "10$^{-1}$"]

        ax.errorbar(
            self.theta_bin_centers,
            self.mean_corr,
            self.error,
            c="k",
            marker="o",
            elinewidth=0.75,
            capthick=0.2,
        )

        ax.format(
            xlim=(0.0008, 0.3),
            ylim=(0.0008, 20),
            xscale="log",
            yscale="log",
            xtickminor=True,
            ytickminor=True,
            title="ASTRODEEP",
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xtickdirection="in",
            ytickdirection="in",
            xtickloc="both",
            ytickloc="both",
            fontsize=fs,
            lw=1,
            xlabel="\u03b8 [deg]",
            ylabel="w(\u03b8)",
        )

        fig.savefig(
            "/cluster/tufts/marchesini_lab/mzaidi01/ASTRODEEP/Clustering/SMDPL/mock_figures/corr_ASTRODEEP_hod"
            + self.hod_runid
            + "_"
            + self.zlabel
            + "_"
            + self.mlabel
            + "_N"
            + str(self.N_samples)
            + ".pdf"
        )
