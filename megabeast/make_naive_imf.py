import numpy as np
import matplotlib.pyplot as plt

import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.physicsmodel.grid import FileSEDGrid

from astropy.table import Table


def make_naive_imf(
    stats_file,
    noise_file_list,
    reorder_tags=None,
    compl_filter="F475W",
    max_age_myr=100,
    n_hist_bins=30,
    chi2_cut=None,
):
    """
    Make a naive IMF for a field

    Parameters
    ----------
    stats_file : string
        BEAST file with the best-fit statistics

    noise_file_list : string or list of strings
        corresponding noise models

    reorder_tags : list of strings (default=None)
        if noise_file_list is multiple files, such that the stats file is a
        merged file, use this to denote the values of 'reorder_tag' that belong
        to each noise file

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

    # set up histograms to hold masses
    hist_bins = np.geomspace(0.2, 100, num=n_hist_bins)
    log_hist_bins = np.log10(hist_bins)
    bin_centers = 10 ** (log_hist_bins[:-1] + np.diff(log_hist_bins) / 2)
    # - no completeness correction
    hist_orig = np.zeros(n_hist_bins - 1)
    # - with completeness correction
    hist_corr = np.zeros(n_hist_bins - 1)
    # also record total mass
    tot_mass = 0.0

    # read in the stats file
    stats_data = Table.read(str(stats_file))

    # grab info from each catalog + noise file
    for i, noise_file in enumerate(np.atleast_1d(noise_file_list)):

        # read in the noise model - includes bias, unc, and completeness
        noisegrid = noisemodel.get_noisemodelcat(noise_file)
        #noise_err = noisemodel_vals["error"]
        #noise_bias = noisemodel_vals["bias"]
        noise_compl = noisemodel_vals["completeness"]

        # initialize array to hold whether to keep each row
        keep_row = np.ones(len(stats_data), dtype=bool)

        # if needed, get the subset for this noise model
        if reorder_tags is not None:
            keep_row[np.where(stats_data["reorder_tag"] != reorder_tags[i])] = False

        # only do sources with the proper age
        keep_row[
            np.where(stats_data["logA_Best"] > np.log10(max_age_myr * 10 ** 6))
        ] = False

        # if set, do a chi2 cut
        if chi2_cut is not None:
            keep_row[np.where(stats_data["chi2min"] > chi2_cut)] = False

        # grab the grid indices for the best fit
        best_ind = stats_data["specgrid_indx"][keep_row]
        # the index lets us get the completeness
        best_comp = model_compl[best_ind][:, 3]
        # and grab the best mass
        best_mass = stats_data["M_ini_Best"][keep_row]

        # put masses in the bins
        hist_orig += np.histogram(best_mass, bins=hist_bins)[0]
        hist_corr += np.histogram(best_mass / best_comp, bins=hist_bins)[0]
        # save total mass
        tot_mass += np.sum(best_mass)

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
    fig.savefig(stats_file.replace(".fits", "_imf.pdf"))

    plt.close("all")

    import pdb

    pdb.set_trace()
