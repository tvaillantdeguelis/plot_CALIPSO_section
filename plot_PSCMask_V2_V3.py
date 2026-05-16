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

sys.path.append("./my_modules/")
from standard_outputs import print_time
from readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from paths import split_granule_date


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
                   "#444444",
                   "#00FA9A",
                   "#FFFF00",
                   "#00BBFF",
                   "#FF0000",
                   "#4700C3"]
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
        filename = f"PSC{VERSION_CAL_LID_L2_PSCMask}_composition_mask"
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
        plt.title(f'PSC mask {VERSION_CAL_LID_L2_PSCMask}', weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 7
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = f"PSC{VERSION_CAL_LID_L2_PSCMask}_feature_mask"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_psc_signal_distributions(self, atten_1064, par_532, perp_532, psc_composition):

        # Map PSC composition labels to class names
        class_labels = {
            1: ("STS", "#00FA9A"),
            2: ("NAT", "#FFBF00"),
            4: ("Ice", "#00BBFF"),
            5: ("Enhanced NAT", "#FF0000"),
            6: ("Wave ice", "#4700C3")
        }

        # Channels
        channels = {
            "1064 nm": atten_1064,
            "532 nm parallel": par_532,
            "532 nm perpendicular": perp_532
        }

        # Bins for log10 histograms
        bins = np.logspace(-7, -2, 60)

        # Dictionary to store estimated mean/std
        log_gaussian_params = {label: {} for label in class_labels}

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (ch_name, ch_data) in zip(axes, channels.items()):
            for label, (name, color) in class_labels.items():
                mask = psc_composition == label
                if np.any(mask):
                    values = ch_data[mask].flatten()
                    values = values[values > 0]  # remove zeros
                    log_values = np.log10(values)

                    # Estimate mean and std in log space
                    mean_log = np.mean(log_values)
                    std_log = np.std(log_values)
                    log_gaussian_params[label][ch_name] = (mean_log, std_log)

                    # Plot histogram as solid line (step)
                    ax.hist(values, bins=bins, histtype='step', lw=1.5, color=color, label=name)

            ax.set_xscale("log")
            ax.set_xlim(1e-7, 1e-2)
            ax.set_xlabel(f"Backscatter ({ch_name}) (km⁻¹ sr⁻¹)")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Number of pixels")
        axes[0].legend(fontsize=8)
        plt.suptitle("PSC Lidar Signal Distributions per Class", weight='bold', y=0.9)

        # Save figure
        filename = f"PSC{VERSION_CAL_LID_L2_PSCMask}_signal_distributions"
        self.save_fig(filename)
        plt.close(fig)

        # Return log Gaussian parameters
        return log_gaussian_params


if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        INDATA_FOLDER = "/DATA/LIENS/CALIOP/"
        GRANULE_DATE = sys.argv[1]
        VERSION_CAL_LID_L2_PSCMask = "V2.00"
        TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"
        SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the PSCMask file) or 'longitude'
        SLICE_START = float(sys.argv[2]) # profindex or longitude
        SLICE_END = float(sys.argv[3]) # profindex or longitude
        EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
        INVERT_XAXIS = False
        YMIN = 8.4
        YMAX = 30
        FIGURES_PATH = sys.argv[4] #"/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
    else:
        INDATA_FOLDER = "/DATA/LIENS/CALIOP/"
        GRANULE_DATE = "2010-01-18T00-19-57ZN"
        VERSION_CAL_LID_L2_PSCMask = "V3.00"
        TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"
        SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the PSCMask file) or 'longitude'
        SLICE_START = 170.59 # profindex or longitude
        SLICE_END = 27.95 # profindex or longitude
        EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
        INVERT_XAXIS = False
        YMIN = 8.4
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
        if VERSION_CAL_LID_L2_PSCMask in ("V2.00", "V3.00"):
            granule_name_char_indexes = np.arange(26, 47)
        elif VERSION_CAL_LID_L2_PSCMask == "V1.00":
            granule_name_char_indexes = np.arange(27, 48)
        else:
            raise ValueError(f"Define 'granule_name_char_indexes' for VERSION_CAL_LID_L2_PSCMask = {VERSION_CAL_LID_L2_PSCMask}")
        for i_char in granule_name_char_indexes:
            granule_name = granule_name + l1_input_filenames[i_filenames][i_char].decode('UTF-8')
        granule_names.append(granule_name)
    granule_name_index = granule_names.index(GRANULE_DATE)
    granule_start_time = cal_psc.get_data("L1_Input_Start_Times")[granule_name_index]
    granule_end_time = cal_psc.get_data("L1_Input_End_Times")[granule_name_index]
    profile_utc_time = cal_psc.get_data("Profile_UTC_Time")
    granule_start_index = (np.abs(profile_utc_time - granule_start_time)).argmin()
    granule_end_index = (np.abs(profile_utc_time - granule_end_time)).argmin()
    print('granule_start_index:', granule_start_index)
    print('granule_end_index:', granule_end_index)
    

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
        "PSC_Composition",
        "Total_Attenuated_Backscatter_1064",
        "Parallel_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532"
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
    
    # Plot signal distributions
    log_gaussian_params = plot_fig.plot_psc_signal_distributions(
        atten_1064=data_dict_cal_psc["Total_Attenuated_Backscatter_1064"],
        par_532=data_dict_cal_psc["Parallel_Attenuated_Backscatter_532"],
        perp_532=data_dict_cal_psc["Perpendicular_Attenuated_Backscatter_532"],
        psc_composition=data_dict_cal_psc["PSC_Composition"]
    )

    # Print all estimated means and stds
    print("\nEstimated log-Gaussian parameters (log10 space):")
    for label, channel_dict in log_gaussian_params.items():
        class_name = {1:"STS", 2:"NAT", 4:"Ice", 5:"Enhanced NAT", 6:"Wave ice"}[label]
        print(f"\nClass: {class_name} (label={label})")
        for channel, (mean_log, std_log) in channel_dict.items():
            print(f"  {channel}: mean = {mean_log:.3f}, std = {std_log:.3f}")
    
    print_time(tic_main_program)
