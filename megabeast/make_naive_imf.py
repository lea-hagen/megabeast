import numpy as np
import matplotlib.pyplot as plt

import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.tools import beast_settings

from astropy.table import Table


def make_naive_imf(
    beast_settings_info,
    use_sd=True,
    compl_filter="F475W",
    max_age_myr=100,
    n_hist_bins=30,
    chi2_cut=None,
):
    """
    Make a naive IMF for a field

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

    """

    # =======================
    # set up file names
    # =======================

    # process beast settings info
    if isinstance(beast_settings_info, str):
        settings = beast_settings.beast_settings(beast_settings_info)
    elif isinstance(beast_settings_info, beast_settings.beast_settings):
        settings = beast_settings_info
    else:
        raise TypeError(
            "beast_settings_info must be string or beast.tools.beast_settings.beast_settings instance"
        )

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
        file_ref_label = {stats_file_list[i]: None for i in range(len(stats_file_list))}

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

    # =======================
    # set up histograms to hold masses
    # =======================

    hist_bins = np.geomspace(0.2, 100, num=n_hist_bins)
    log_hist_bins = np.log10(hist_bins)
    bin_centers = 10 ** (log_hist_bins[:-1] + np.diff(log_hist_bins) / 2)
    # - no completeness correction
    hist_orig = np.zeros(n_hist_bins - 1)
    # - with completeness correction
    hist_corr = np.zeros(n_hist_bins - 1)
    # also record total mass
    tot_mass = 0.0

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
                    np.where(stats_data[file_ref_label] != file_ref_info[stats_file][i])
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
            best_comp = model_compl[best_ind][:, 3]
            # and grab the best mass
            best_mass = stats_data["M_ini_Best"][keep_row]

            # put masses in the bins
            hist_orig += np.histogram(best_mass, bins=hist_bins)[0]
            hist_corr += np.histogram(best_mass / best_comp, bins=hist_bins)[0]
            # save total mass
            tot_mass += np.sum(best_mass)

    # =======================
    # do the plotting
    # =======================

    # make a plot
    fig = plt.figure(figsize=(5, 4))

    ax = plt.gca()

    # plot points
    # plt.hist(
    #    hist_orig, bins=hist_bins, log=True,
    #    facecolor='grey', linewidth=0.25, edgecolor='grey',
    # )

    # plt.plot(bin_centers, hist_orig,
    #    marker='o', mew=0, color='black', markersize=5,
    #    linestyle='None')
    plt.errorbar(
        bin_centers,
        hist_orig,
        yerr=np.sqrt(hist_orig),
        marker="o",
        mew=0,
        color="black",
        markersize=5,
        linestyle="None",
        label="Observed IMF",
    )

    # overplot a salpeter IMF
    xi = tot_mass / 4.44
    plt.plot(
        np.geomspace(0.2, 100, 50),
        xi * np.geomspace(0.2, 100, 50) ** -2.35,
        markersize=0,
        linestyle=":",
        color="xkcd:azure",
        label="S55 IMF",
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
