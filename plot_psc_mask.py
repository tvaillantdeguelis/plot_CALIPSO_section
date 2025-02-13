#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
from pyhdf.SD import SD
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import seaborn as sns
import os
import sys

from my_modules.standard_outputs import print_time
from my_modules.readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from my_modules.figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from my_modules.paths import split_granule_date


class FigureMaker(CALIOPFigureMaker):

    def plot_psc_composition(self, mask):

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["Likely tropo. ice", # -4
                  "Not determinable", # -1
                  "No detection", # 0
                  "STS", # 1
                  "NAT", # 2
                  "Ice", # 4
                  "Enhanced NAT", # 5
                  "Wave ice"] # 6

        # Colormap
        nb_colors = len(clabels)
        # palette = ["#000000",
        #            "#FFFFFF",
        #            "#888888",
        #            "#00FF26",
        #            "#FAFF00",
        #            "#00BBFF",
        #            "#FF0000",
        #            "#4700C3"]
        palette = ["#000000",
                   "#FFFFFF",
                   "#888888",
                   "#FFFF00",
                   "#FFA500",
                   "#87CEFA",
                   "#B22222",
                   "#4169E1"]
        colorbins = np.array((-5, -3, -0.5, 0.5, 1.5, 3.5, 4.5, 5.5, 6.5))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(f'PSC Mask Composition {VERSION_CAL_LID_L2_PSCMask}', weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = f"PSC_Composition_{VERSION_CAL_LID_L2_PSCMask}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_psc_mask(self, mask):

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        ### Change values to match colorbar order ###
        # Put non detection at 0
        mask[mask < 0] = 0
        # Get last to digit (N2 N3 = Horizontal averaging required and parameter used for detection)
        mask = np.abs(mask) // 10**1 % 10 + np.abs(mask) // 10**0 % 10
        # Change value
        mask[mask == 2] = 101
        mask[mask == 4] = 102
        mask[mask == 10] = 103
        mask[mask == 28] = 104
        mask[mask == 3] = 2
        mask[mask == 9] = 3
        mask[mask == 27] = 4

        # Labels
        clabels = ["No detection",
                   "R 5-km",
                   "R 15-km",
                   "R 45-km",
                   "R 135-km",
                   "$\\beta'_{\\perp}$ 5-km",
                   "$\\beta'_{\\perp}$ 15-km",
                   "$\\beta'_{\\perp}$ 45-km",
                   "$\\beta'_{\\perp}$ 135-km"
                  ] 

        # Colormap
        nb_colors = len(clabels)
        palette = ['#ffffff',
                   '#9d0000', '#e11301', '#fe6940', '#ffb07e',
                   '#0a337f', '#5767bc', '#97a1fb', '#dee5ff']
        colorbins = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 101.5, 102.5, 103.5, 104.5))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(f'PSC mask {VERSION_CAL_LID_L2_PSCMask}', weight='bold', y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 7
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = f"PSC_Feature_Mask_{VERSION_CAL_LID_L2_PSCMask}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    INDATA_FOLDER = "/DATA/LIENS/CALIOP/"
    GRANULE_DATE = "2011-06-25T00-11-52ZN"
    VERSION_CAL_LID_L2_PSCMask = "V2.00"
    TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"
    SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the PSCMask file) or 'longitude'
    SLICE_START = 5.95 # profindex or longitude
    SLICE_END = -150.07 # profindex or longitude
    EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
    INVERT_XAXIS = False
    YMIN = 8
    YMAX = 30
    FIGURES_PATH = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    # *******************************
    # *** Load PSC mask data file ***
    print("\n*****Load PSC mask data file...*****")
    
    # Get filename and filepath
    granule_date_dict = split_granule_date(GRANULE_DATE)
    filename_psc = f"CAL_LID_L2_PSCMask-{TYPE_CAL_LID_L2_PSCMask}-{VERSION_CAL_LID_L2_PSCMask.replace('.', '-')}." \
                   f"{granule_date_dict['year']}-{granule_date_dict['month']:02d}-{granule_date_dict['day']:02d}T00-00-00ZN.hdf"
    hdffile = os.path.join(INDATA_FOLDER, f"PSCMask.{VERSION_CAL_LID_L2_PSCMask.replace('V', 'v')}",
                           str(granule_date_dict['year']),
                           f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                           f"{granule_date_dict['day']:02d}",
                           filename_psc)

    # Open HDF file
    print(f"\tGranule path: {hdffile}")
    cal_psc = CALIPSOReader(hdffile)

    # Find granule section in the daily PSC file
    l1_input_filenames = cal_psc.get_data("L1_Input_Filenames")
    granule_names = []
    for i_filenames in np.arange(l1_input_filenames.shape[0]):
        granule_name = ''
        if VERSION_CAL_LID_L2_PSCMask == "V2.00":
            granule_name_char_indexes = np.arange(26, 47)
        elif VERSION_CAL_LID_L2_PSCMask == "V1.00":
            granule_name_char_indexes = np.arange(27, 48)
        else:
            sys.exit(f"Define 'granule_name_char_indexes' for VERSION_CAL_LID_L2_PSCMask = {VERSION_CAL_LID_L2_PSCMask}")
        for i_char in granule_name_char_indexes:
            granule_name = granule_name + l1_input_filenames[i_filenames][i_char].decode('UTF-8')
        granule_names.append(granule_name)
    granule_name_index = granule_names.index(GRANULE_DATE)
    granule_start_time = cal_psc.get_data("L1_Input_Start_Times")[granule_name_index]
    granule_end_time = cal_psc.get_data("L1_Input_End_Times")[granule_name_index]
    profile_utc_time = cal_psc.get_data("Profile_UTC_Time")
    granule_start_index = (np.abs(profile_utc_time - granule_start_time)).argmin()
    granule_end_index = (np.abs(profile_utc_time - granule_end_time)).argmin()
    

    # Get prof_min and prof_max from longitudes
    lat_granule = cal_psc.get_data("Latitude")[granule_start_index:granule_end_index+1]
    lon_granule = cal_psc.get_data("Longitude")[granule_start_index:granule_end_index+1]
    if SLICE_START_END_TYPE == 'longitude':
        prof_min_granule, prof_max_granule = get_prof_min_max_indexes_from_lon(lon_granule, SLICE_START, SLICE_END) 
        prof_min = prof_min_granule + granule_start_index
        prof_max = prof_max_granule + granule_start_index
    else:
        prof_min = SLICE_START
        prof_max = SLICE_END
        
    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {prof_min:d} "
          f"(lat = {lat_granule[prof_min_granule]:.2f} / lon = {lon_granule[prof_min_granule]:.2f}) "
          f"to max profile index {prof_max:d} "
          f"(lat = {lat_granule[prof_max_granule]:.2f} / lon = {lon_granule[prof_max_granule]:.2f})")
    
    # Load 2D-McDA parameters
    data_dict_cal_psc = {}
    cal_psc_keys = [
        "Latitude",
        "Longitude",
        "Profile_UTC_Time",
        "Altitude",
        "PSC_Feature_Mask",
        "PSC_Composition"
    ]
    for key in cal_psc_keys:
        data_dict_cal_psc[key] = cal_psc.get_data(key, prof_min, prof_max, 'profindex')
    
    lat = data_dict_cal_psc["Latitude"] 
    lon = data_dict_cal_psc["Longitude"]
    alt = data_dict_cal_psc["Altitude"]

    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    plot_fig.fig_w = cm2in(16) # cm
    plot_fig.fig_h = cm2in(8) # cm
    plot_fig.adj_left = 0.08
    plot_fig.adj_bottom = 0.11
    plot_fig.adj_right = 0.87
    plot_fig.adj_top = 0.81
    plot_fig.axes_title_pad = 1.14
    plot_fig.axes_titlesize = 8
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, lon[0], lon[-1])
    plot_fig.set_head_filename(GRANULE_DATE, lon[0], lon[-1])
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_coordinates(lat, lon, alt)
    
    # Plot PSC mask composition
    plot_fig.plot_psc_composition(data_dict_cal_psc["PSC_Composition"])

    # Plot PSC feature mask
    plot_fig.plot_psc_mask(data_dict_cal_psc["PSC_Feature_Mask"])
    
    print_time(tic_main_program)
