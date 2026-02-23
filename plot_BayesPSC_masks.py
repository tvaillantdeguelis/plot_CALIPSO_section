#!/usr/bin/env python
# coding: utf8

import sys
import os

from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import cmocean

sys.path.append("./my_modules/")
from standard_outputs import print_time
from readers.calipso_reader import get_prof_min_max_indexes_from_lon
from paths import split_granule_date
from figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges


FILL_VALUE_FLOAT = -9999.0


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.14

    def plot_PSC_classification(self, mask):
        
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No detection", # 0
                   "STS", # 1
                   "NAT", # 2
                   "Ice" # 3
        ]

        
        # Discrete colormap
        palette = [
                   [ 68./255,  68./255,  68./255], # 0: No detection (dark grey)
                   [  0./255, 250./255, 154./255], # 1: STS (light green)
                   [255./255, 255./255,   0./255], # 2: NAT (yellow)
                   [  0./255, 187./255, 255./255], # 3: Ice (blue)
        ]
        my_cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.arange(len(palette) + 1) - 0.5
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)

        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        # Plot
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T,
                            cmap=my_cmap, norm=my_norm, rasterized=True)

        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)

        plt.title(r'$\mathbf{Bayes\ PSC\ classification\ %s}$' % (VERSION_BAYESPSC),
                fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)

        # Save
        filename = f"BayesPSC{VERSION_BAYESPSC}_classification"
        self.save_fig(filename)

        plt.close(fig)


    def plot_PSC_chunk_detection_value(self, mask):

        # Remove edges
        if self.edges_removal != 0:
            mask = remove_edges(mask, self.edges_removal)

        # Mask 0 for transparency
        masked = np.ma.masked_where(mask == 0, mask)

        # Colormap
        my_cmap = cmocean.cm.thermal
        my_cmap.set_bad("white")

        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, masked.T,
                            cmap=my_cmap, rasterized=True, vmin=0, vmax=215)

        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)

        plt.title(r'$\mathbf{PSC\ homogeneous\ chunk\ IDs}$',
                fontsize=self.axes_titlesize, y=self.axes_title_pad)

        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical')
        cbar.set_ticks([0, 50, 100, 150, 200, 215])
        cbar.set_label("Chunk ID")

        filename = f"BayesPSC{VERSION_BAYESPSC}_chunk_detection_value"
        self.save_fig(filename)

        plt.close(fig)


    def plot_ab_signal(self, ab_signal, title, filename):

        # Mask where fill_value
        ab_signal = np.ma.masked_where(ab_signal == FILL_VALUE_FLOAT, ab_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            ab_signal = remove_edges(ab_signal, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm(), rasterized=True)
        plt.clim(1e-6, 2e-3)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\beta^{\prime}$ (km$^{-1}$ sr$^{-1}$)", labelpad=5)

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    INDATA_FOLDER = "/home/vaillant/codes/projects/BayesPSC/out/data/"
    GRANULE_DATE = "2010-01-18T00-19-57ZN"
    GRANULE_SECTION = "_lon_170.68_27.93" # void if complete file
    VERSION_BAYESPSC = "V1.0.0"
    TYPE_BAYESPSC = "Prototype"
    SLICE_START_END_TYPE = 'longitude' # 'profindex' or 'longitude'
    SLICE_START = 170.59 # profindex or longitude
    SLICE_END = 27.95 # profindex or longitude
    EDGES_REMOVAL = 0 # number of 1/3-km prof to remove on both edges of plot
    INVERT_XAXIS = False
    YMIN = 8.4 # None
    YMAX = 30
    BROWSE_IMAGE_ASPECT_RATIO = True
    FLAG_GRANULE = True # Write granule name in the plots
    FIGURES_PATH = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    # **********************************
    # *** Load BayesPSC netCDF data file ***
    print("\n*****Load BayesPSC netCDF data file...*****")
    
    # Get filename and filepath
    filename_bayespsc = f"BayesPSC-{TYPE_BAYESPSC}-{VERSION_BAYESPSC.replace('.', '-')}." \
                       f"{GRANULE_DATE}{GRANULE_SECTION}.nc"
    granule_date_dict = split_granule_date(GRANULE_DATE)
    ncfile = os.path.join(INDATA_FOLDER, f"BayesPSC.{VERSION_BAYESPSC.replace('V', 'v')}",
                           str(granule_date_dict['year']),
                           f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                           f"{granule_date_dict['day']:02d}",
                           filename_bayespsc)

    # Open netCDF file
    print(f"\tGranule path: {ncfile}")
    ds_bayespsc = xr.open_dataset(ncfile)

    # Get prof_min and prof_max from longitudes
    lat_granule = ds_bayespsc["Latitude"].values
    lon_granule = ds_bayespsc["Longitude"].values
    if SLICE_START_END_TYPE == 'longitude':
        prof_min, prof_max = get_prof_min_max_indexes_from_lon(lon_granule, SLICE_START, SLICE_END)
    else:
        prof_min = SLICE_START
        prof_max = SLICE_END

    if prof_min is None:
        prof_min = 0
    if prof_max is None:
        prof_max = len(lat_granule) - 1

    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {prof_min:d} "
          f"(lat = {lat_granule[prof_min]:.2f} / lon = {lon_granule[prof_min]:.2f}) "
          f"to max profile index {prof_max:d} "
          f"(lat = {lat_granule[prof_max]:.2f} / lon = {lon_granule[prof_max]:.2f})")
    
    # Slice data
    lat = lat_granule[prof_min:prof_max+1]
    lon = lon_granule[prof_min:prof_max+1]
    alt = ds_bayespsc["Altitude"].values
    psc_classification = ds_bayespsc["PSC_classification"].values[prof_min:prof_max+1, :]
    psc_chunk_detection_value = ds_bayespsc["PSC_chunk_detection_value"].values[prof_min:prof_max+1, :]
    chunk_mean_ab532par = ds_bayespsc["Chunk_Mean_Parallel_Attenuated_Backscatter_532"].values[prof_min:prof_max+1, :]
    chunk_mean_ab532per = ds_bayespsc["Chunk_Mean_Perpendicular_Attenuated_Backscatter_532"].values[prof_min:prof_max+1, :]
    chunk_mean_ab1064 = ds_bayespsc["Chunk_Mean_Attenuated_Backscatter_1064"].values[prof_min:prof_max+1, :]


    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    if BROWSE_IMAGE_ASPECT_RATIO:
        plot_fig.fig_w = cm2in(16) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.87
        plot_fig.adj_top = 0.81
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, lon_granule[prof_min], lon_granule[prof_max])
    plot_fig.set_head_filename(GRANULE_DATE, lon_granule[prof_min], lon_granule[prof_max])
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_coordinates(lat, lon, alt)
    
    # Plot masks
    plot_fig.plot_PSC_classification(psc_classification)
    plot_fig.plot_PSC_chunk_detection_value(psc_chunk_detection_value)
    
    # Plot signals
    if True:
        filename = f"BayesPSC{VERSION_BAYESPSC}_chunk_mean_AB532par"
        title = r"$\mathbf{532\ nm\ Parallel\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\parallel}$"
        plot_fig.plot_ab_signal(chunk_mean_ab532par, title, filename)

        filename = f"BayesPSC{VERSION_BAYESPSC}_chunk_mean_AB532per"
        title = r"$\mathbf{532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\perp}$"
        plot_fig.plot_ab_signal(chunk_mean_ab532per, title, filename)

        filename = f"BayesPSC{VERSION_BAYESPSC}_chunk_mean_AB1064"
        title = r"$\mathbf{1064\ nm\ Attenuated\ Backscatter}\ \beta^{\prime}_{1064}$"
        plot_fig.plot_ab_signal(chunk_mean_ab1064, title, filename)

    print_time(tic_main_program)
