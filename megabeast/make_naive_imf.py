import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.tools import beast_settings
from beast.tools.run import create_filenames

from astropy.table import Table
import asdf


def make_naive_imf(
    beast_settings_info,
    use_sd=True,
    compl_filter="F475W",
    max_age_myr=100,
    n_hist_bins=30,
    chi2_cut=None,
    reread_data=False,
):
    """
    Make a naive IMF for a field.  Saves the mass/completeness info into an asdf
    file so that histogram settings can easily be changed.

    Parameters
    ----------
    beast_settings_info : string or beast.tools.beast_settings.beast_settings instance
        if string: file name with beast settings
        if class: beast.tools.beast_settings.beast_settings instance

    use_sd : boolean (default=True)
        If True, the BEAST runs were split into source density or background bins

    compl_filter : string (default='F475W')
        filter name to use for completeness

    max_age_myr : float (default=100)
        only stars with ages less than this (in Myr) will be used to construct
        the naive IMF

    n_hist_bins : int (default=30)
        number of bins to use for the mass histogram (log spaced between 0.2 and
        100 Msun)

    chi2_cut : float (default=None)
        any sources with chi2 larger than this will be removed

    reread_data : boolean (default=False)
        If True, re-read all of the BEAST mass/completeness data, even if an
        asdf file of values already exists.

    """

    # process beast settings info
    if isinstance(beast_settings_info, str):
        settings = beast_settings.beast_settings(beast_settings_info)
    elif isinstance(beast_settings_info, beast_settings.beast_settings):
        settings = beast_settings_info
    else:
        raise TypeError(
            "beast_settings_info must be string or beast.tools.beast_settings.beast_settings instance"
        )

    # asdf file that has (or will have) the mass/completeness data
    imf_data_file = "{0}/{0}_imf_data.asdf"

    # =======================
    # assemble data
    # =======================

    # the data file doesn't exist OR user wishes to re-read all the data
    if (not os.path.isfile(imf_data_file)) or (reread_data):

        # ------- set up file names

        # make list of file names
        file_dict = create_filenames.create_filenames(
            settings, use_sd=use_sd, nsubs=settings.n_subgrid,
        )

        # no subgrids
        if settings.n_subgrid == 1:

            stats_file_list = file_dict["stats_files"]
            noise_file_dict = {
                stats_file_list[i]: file_dict["noise_trim_files"][i]
                for i in range(len(stats_file_list))
            }
            file_ref_label = {
                stats_file_list[i]: None for i in range(len(stats_file_list))
            }

        # with subgrids
        if settings.n_subgrid > 1:

            # start by converting names to merged (across subgrids) version
            stats_file_list_unmerged = file_dict["stats_files"]
            merged_stats = []
            for filename in stats_file_list_unmerged:
                gs_start = filename.rfind("_gridsub")
                st_start = filename.rfind("_stats.fits")
                merged_stats.append(filename[:gs_start] + filename[st_start:])

            # now get their corresponding noise files and gridsub info
            stats_file_list = list(set(merged_stats))
            noise_file_dict = {}
            file_ref_label = "best_gridsub_tag"
            file_ref_info = {}
            for filename in stats_file_list:
                noise_file_dict[filename] = [
                    nf
                    for i, nf in enumerate(file_dict["noise_trim_files"])
                    if filename == merged_stats[i]
                ]
                file_ref_info[filename] = [
                    nf
                    for i, nf in enumerate(file_dict["gridsub_info"])
                    if filename == merged_stats[i]
                ]

        # ------- grab the relevant mass/completeness data

        imf_data = {"mass_data": [], "compl_data": []}

        # get index for completeness
        compl_ind = settings.basefilters.index(compl_filter)

        for stats_file in stats_file_list:

            # read in the stats file
            stats_data = Table.read(str(stats_file))

            # grab info from corresponding noise file(s)
            for i, noise_file in enumerate(np.atleast_1d(noise_file_dict[stats_file])):

                # read in the noise model - includes bias, unc, and completeness
                noisemodel_vals = noisemodel.get_noisemodelcat(noise_file)
                noise_compl = noisemodel_vals["completeness"]

                # initialize array to hold whether to keep each row
                keep_row = np.ones(len(stats_data), dtype=bool)

                # if needed, get the subset for this noise model
                if file_ref_label is not None:
                    keep_row[
                        np.where(
                            stats_data[file_ref_label] != file_ref_info[stats_file][i]
                        )
                    ] = False

                # only do sources with the proper age
                keep_row[
                    np.where(stats_data["logA_Best"] > np.log10(max_age_myr * 10 ** 6))
                ] = False

                # if set, do a chi2 cut
                if chi2_cut is not None:
                    keep_row[np.where(stats_data["chi2min"] > chi2_cut)] = False

                # grab the grid indices for the best fit
                best_ind = stats_data["Pmax_indx"][keep_row]
                # the index lets us get the completeness
                imf_data["compl_data"] += noise_compl[best_ind][:, compl_ind].tolist()
                # and grab the best mass
                imf_data["mass_data"] += stats_data["M_ini_Best"][keep_row].tolist()

        # save data into asdf file
        imf_data["mass_data"] = np.array(imf_data["mass_data"])
        imf_data["compl_data"] = np.array(imf_data["compl_data"])
        asdf.AsdfFile(imf_data).write_to(imf_data_file)

    # otherwise, read in the existing data
    else:
        with asdf.open(imf_data_file) as af:
            imf_data = af.tree

    # =======================
    # do the plotting
    # =======================

    # ------- make histograms

    hist_bins = np.geomspace(0.2, 100, num=n_hist_bins)
    log_hist_bins = np.log10(hist_bins)
    bin_centers = 10 ** (log_hist_bins[:-1] + np.diff(log_hist_bins) / 2)
    # - no completeness correction
    hist_orig = np.zeros(n_hist_bins - 1)
    # - with completeness correction
    hist_corr = np.zeros(n_hist_bins - 1)

    # put masses in the bins
    hist_orig = np.histogram(imf_data["mass_data"], bins=hist_bins)[0]
    hist_orig_err = np.sqrt(hist_orig)
    hist_corr = np.histogram(
        imf_data["mass_data"], weights=(1 / imf_data["compl_data"]), bins=hist_bins
    )[0]
    hist_corr_err = np.sqrt(hist_orig)

    # save total mass
    tot_mass_orig = np.sum(imf_data["mass_data"])
    tot_mass_corr = np.sum(imf_data["mass_data"] / imf_data["compl_data"])

    # ------- do the plotting

    fig = plt.figure(figsize=(5, 4))

    ax = plt.gca()

    # BEAST IMFs
    plt.errorbar(
        bin_centers,
        hist_orig,
        yerr=hist_orig_err,
        marker="o",
        mew=0,
        color="black",
        markersize=5,
        linestyle="None",
        label="Observed IMF",
    )
    plt.errorbar(
        bin_centers,
        hist_corr,
        yerr=hist_corr_err,
        marker="o",
        mew=0,
        color="black",
        markersize=5,
        linestyle="None",
        label="Corrected IMF",
    )

    # overplot a salpeter IMF
    xi_orig = tot_mass_orig / 4.44
    xi_corr = tot_mass_corr / 4.44
    temp_mass = np.geomspace(0.2, 100, 50)
    plt.plot(
        temp_mass,
        xi_orig * temp_mass ** -2.35,
        markersize=0,
        linestyle=":",
        color="xkcd:azure",
        label="S55 IMF",
    )
    plt.plot(
        temp_mass,
        xi_corr * temp_mass ** -2.35,
        markersize=0,
        linestyle=":",
        color="xkcd:azure",
        # label="S55 IMF",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Mass (M$_{\odot}$)", fontsize=14)
    ax.set_ylabel("N", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12)

    ax.legend()

    plt.tight_layout()

    # save figure
    fig.savefig("{0}/{0}_imf.pdf".format(settings.project))

    plt.close("all")


if __name__ == "__main__":  # pragma: no cover
    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "beast_settings_file", type=str, help="file name with beast settings",
    )
    parser.add_argument(
        "--use_sd",
        type=int,
        default=1,
        help="set to True (1) if the fitting used source density bins",
    )
    parser.add_argument(
        "--compl_filter",
        type=str,
        default="F475W",
        help="filter name to use for completeness",
    )
    parser.add_argument(
        "--max_age_myr",
        type=float,
        default=100,
        help="only stars with ages less than this will be used to construct the IMF",
    )
    parser.add_argument(
        "--n_hist_bins",
        type=float,
        default=30,
        help="number of bins to use for the mass histogram (log spaced between 0.2 and 100 Msun)",
    )
    parser.add_argument(
        "--chi2_cut",
        type=float,
        default=None,
        help="any sources with chi2 larger than this will be removed",
    )
    parser.add_argument(
        "--reread_data",
        type=int,
        default=0,
        help="set to True (1) to re-read the BEAST mass/completeness data, even if saved file exists",
    )

    args = parser.parse_args()

    make_naive_imf(
        args.beast_settings_file,
        use_sd=bool(args.use_sd),
        compl_filter=args.compl_filter,
        max_age_myr=args.max_age_myr,
        n_hist_bins=args.n_hist_bins,
        chi2_cut=args.chi2_cut,
        reread_data=bool(args.reread_data),
    )
