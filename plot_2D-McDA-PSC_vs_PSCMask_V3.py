#!/usr/bin/env python
# coding: utf8

import sys
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import seaborn as sns
import os
import cmocean
import cmlidar
import seaborn as sns

sys.path.append("./my_modules/")
from standard_outputs import print_time
from readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from paths import split_granule_date
from figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from geotools import neighbors

FILL_VALUE_FLOAT = -9999.0


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.14
        self.clabelpad = 40


    def set_max_detect_level(self, max_detect_level):
        self.max_detect_level = max_detect_level
        
        
    def plot_mask(self, ax0, cax, mask, channel):
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No\ndetect.",] +\
                  ["%d" % i for i in np.arange(self.max_detect_level)+1] +\
                  ["Invalid"]

        # Colormap
        nb_colors = self.max_detect_level
        palette = sns.cubehelix_palette(nb_colors, start=2, rot=1, hue=1., gamma=1., light=0.8,
                                        dark=0.2, reverse=True)
        palette.insert(0, [1.0, 1.0, 1.0]) # 0 = Nothing
        palette.append([0.5, 0.5, 0.5])  # last color = grey for 255
        colorbins = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 256))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)

        # Plot figure
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        if channel == '532_par':
            title = f"2D-McDA-PSC {VERSION_2D_McDA} 532 nm parallel detection feature mask"
        elif channel == '532_per':
            title = f"2D-McDA-PSC {VERSION_2D_McDA} 532 nm perpendicular detection feature mask"
        elif channel == '1064':
            title = f"2D-McDA-PSC {VERSION_2D_McDA} 1064 nm detection feature mask"
        else:
            raise ValueError(f"Unknown channel = {channel}")
        ax0.set_title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', drawedges=True)
        fontsize_clabel = 8
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
            if j == 3:
                cbar.ax.text(4, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), 'Detection level',
                             va='center', rotation=90, fontsize=fontsize_clabel, transform=cbar.ax.transAxes)


    def plot_ab_signal(self, ax0, cax, ab_signal, title):

        # Mask where fill_value
        ab_signal = np.ma.masked_where(ab_signal == FILL_VALUE_FLOAT, ab_signal)

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            ab_signal = remove_edges(ab_signal, self.edges_removal)

        # Figure style
        setstyle("ticks_nogrid")

        # Plot figure
        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9
        ax0.set_title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm(vmin=1e-6, vmax=2e-3), rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\beta^{\prime}$ (km$^{-1}$ sr$^{-1}$)", labelpad=5)


    def plot_sr_signal(self, ax0, cax, sr_signal, title):

        # Mask where fill_value
        sr_signal = np.ma.masked_where(sr_signal == FILL_VALUE_FLOAT, sr_signal)

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            sr_signal = remove_edges(sr_signal, self.edges_removal)

        # Plot figure
        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        sr_signal[(sr_signal<0) & ~sr_signal.mask] = 1e-9
        ax0.set_title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, sr_signal.T, cmap=my_cmap, norm=LogNorm(vmin=1, vmax=10), rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"Scattering ratio", labelpad=5)


    def plot_temperature(self, ax0, cax, temperature, title):

        # Mask where fill_value
        temperature = np.ma.masked_where(temperature == FILL_VALUE_FLOAT, temperature)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            temperature = remove_edges(temperature, self.edges_removal)

        # Plot figure
        ax0.set_facecolor('0.5')
        ax0.set_title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, temperature.T, cmap=my_cmap, vmin=180, vmax=220, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"Temperature (K)", labelpad=5)


    def plot_composite_mask_channel(self, ax0, cax, mask):
        """
        mask values:
        0 = no detection
        1 = 532 par
        2 = 532 per
        3 = 532 par + 532 per
        4 = 1064
        5 = 532 par + 1064
        6 = 532 per + 1064
        7 = all three
        255 = invalid
        """
        label_all_channel = r"532$\mathrm{\parallel}$ + 532$\mathrm{\bot}$ + 1064"
        clabels = [
            "No detection",
            r"532$\mathrm{\parallel}$",
            r"532$\mathrm{\bot}$",
            r"532$\mathrm{\parallel}$ + 532$\mathrm{\bot}$",
            r"1064",
            r"532$\mathrm{\parallel}$ + 1064",
            r"532$\mathrm{\bot}$ + 1064",
            label_all_channel,
            "Invalid"
        ]

        palette = [
                   [1.0,           1.0,      1.0], # 0: No detection (white)
                   [0.9,           0.0,      0.0], # 1: 532 par only (red)
                   [0.9,           0.9,      0.0], # 2: 532 per only (yellow)
                   [135./255, 206./255, 235./255], # 3: 1064 only (blue)
                   [255./255, 140./255,   0./255], # 4: 532 par + 532 per (orange)
                   [102./255,   0./255, 153./255], # 5: 532 par + 1064 (violet)
                   [152./255, 251./255, 152./255], # 6: 532 per + 1064 (green)
                   [0.0,           0.0,      0.0], # 7: 532 par + 532 per + 1064 (black)
                   [0.5,           0.5,      0.5]  # 255: Invalid (grey)
        ]

        cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 256])
        norm = mpl.colors.BoundaryNorm(colorbins, cmap.N)

        # Plot figure
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, mask.T,
                        cmap=cmap, norm=norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        ax0.set_title(f"2D-McDA-PSC {VERSION_2D_McDA} composite detection mask (by channel)", fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', drawedges=True)
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=4 if lab==label_all_channel else 7, transform=cbar.ax.transAxes)


    def plot_psc_composition(self, ax0, cax, mask, temp, title, plot_temp=False, plot_tropopause=False):
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # PSC class values in the desired display order
        psc_values = [
            -4,  # Likely tropo.
            -1,  # Not determinable
            0,  # No detection
            3,  # SBS
            1,  # STS
            2,  # NAT
            4,  # Ice
            5,  # Enhanced NAT
            6   # Wave ice
        ]

        # Labels
        clabels = [
            "Likely tropo.",   # -4
            "Not determinable",# -1
            "No detection",    # 0
            "SBS",             # 3
            "STS",             # 1
            "NAT",             # 2
            "Ice",             # 4
            "Enhanced NAT",    # 5
            "Wave ice"         # 6
        ]

        # Color palette associated with each class
        palette = [
            "#000000",  # Likely tropo.
            "#FFFFFF",  # Not determinable
            "#444444",  # No detection
            "#888888",  # SBS
            "#00FA9A",  # STS
            "#FFFF00",  # NAT
            "#00BBFF",  # Ice
            "#FF0000",  # Enhanced NAT
            "#4700C3"   # Wave ice
        ]

        # Create a mapping between PSC values and ordered indices
        mapping = {value: index for index, value in enumerate(psc_values)}

        # Convert original PSC values into ordered indices
        # Example:
        #   -4 -> 0
        #   -1 -> 1
        #    0 -> 2
        #    3 -> 3
        #    1 -> 4
        # etc.
        mask_plot = np.full(mask.shape, -1, dtype=np.int32)
        for value, idx in mapping.items():
            mask_plot[mask == value] = idx

        # Create colormap
        my_cmap = mpl.colors.ListedColormap(palette)

        # Define color boundaries for ordered indices
        colorbins = np.arange(len(clabels) + 1) - 0.5

        # Create normalization
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)

        # Plot figure
        pc = ax0.pcolormesh(self.pindexbins, self.altbins, mask_plot.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        
        if plot_temp:
            APPLY_SMOOTH = False
            SIGMA = (1, 1)   # (vertical, horizontal)
            temp_to_plot = temp.copy()
            if APPLY_SMOOTH:
                from scipy.ndimage import gaussian_filter
                temp_to_plot = gaussian_filter(temp_to_plot, sigma=SIGMA)
            x_centers = 0.5 * (self.pindexbins[:-1] + self.pindexbins[1:])
            y_centers = 0.5 * (self.altbins[:-1] + self.altbins[1:])
            levels = np.arange(150, 250, 5)
            cs = ax0.contour(x_centers, y_centers, temp_to_plot.T, levels=levels, colors='white', linewidths=0.8)
            ax0.clabel(cs, fmt='%d K', colors='white', fontsize=8)
        
        if plot_tropopause:
            x_centers = 0.5 * (self.pindexbins[:-1] + self.pindexbins[1:])
            tropopause_alt = data_dict_cal_2d_mcda["Tropopause_Altitude_MERRA2"] 
            ax0.plot(x_centers, tropopause_alt, c="w", lw=1.5, ls="-", zorder=10, label="Tropopause")
            if False:
                i_mid = len(x_centers) // 2
                ax0.text(x_centers[i_mid], tropopause_alt[i_mid] + 1.0,
                        "Tropopause", color="white", fontsize=10, ha="center", va="bottom", zorder=11, weight="bold")

        ax0.set_title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)

        # Plot colorbar
        cbar = plt.colorbar(pc, cax=cax, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        

if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        GRANULE_DATE = sys.argv[1]
        VERSION_2D_McDA = sys.argv[2]
    else:
        GRANULE_DATE = "2010-01-18T00-19-57ZN"
        VERSION_2D_McDA = "V2.7.2"
    INDATA_FOLDER = "/home/vaillant/codes/projects/2D_McDA_PSC/out/data/"
    INDATA_FOLDER_PSCMask = "/DATA/LIENS/CALIOP/"
    GRANULE_SECTION = "" # void if complete file
    TYPE_2D_McDA = "Prototype"
    VERSION_CAL_LID_L2_PSCMask = "V3.00"
    TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"
    SLICE_START_END_TYPE = 'profindex' # 'longitude' # 'profindex' (of the 2D-McDA file) or 'longitude'
    SLICE_START = None # 170.59 # profindex or longitude
    SLICE_END = None # 27.95 # profindex or longitude
    EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
    MAX_DETECT_LEVEL = 5
    INVERT_XAXIS = False
    YMIN = 8.4
    YMAX = 30
    FIGURES_FILETYPE = 'png' #'png' 'svg'
    FLAG_GRANULE = False # Write granule name in the subplots
    FIGURES_PATH = f"/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/2D-McDA-PSC_{VERSION_2D_McDA}_vs_PSCMask_V3/"
    PLOT_TEMPERATURE_CONTOURS_OVER_COMPOSITION = False
    PLOT_TROPOPAUSE = True
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    # **********************************
    # *** Load 2D-McDA netCDF data file ***
    print("\n*****Load 2D-McDA netCDF data file...*****")
    
    # Get filename and filepath
    filename_2d_mcda = f"CAL_LID_L2_2D_McDA_PSC-{TYPE_2D_McDA}-{VERSION_2D_McDA.replace('.', '-')}." \
                       f"{GRANULE_DATE}{GRANULE_SECTION}.nc"
    granule_date_dict = split_granule_date(GRANULE_DATE)
    ncfile = os.path.join(INDATA_FOLDER, f"2D_McDA_PSC.{VERSION_2D_McDA.replace('V', 'v')}",
                           str(granule_date_dict['year']),
                           f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                           f"{granule_date_dict['day']:02d}",
                           filename_2d_mcda)

    # Open netCDF file
    print(f"\tGranule path: {ncfile}")
    ds_2d_mcda = xr.open_dataset(ncfile, mask_and_scale=False)

    # Get prof_min and prof_max from longitudes
    lat_granule = ds_2d_mcda["Latitude"].values
    lon_granule = ds_2d_mcda["Longitude"].values
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
    
    lon_granule_2d_mcda_psc_min = lon_granule[prof_min]
    lon_granule_2d_mcda_psc_max = lon_granule[prof_max]

    # Load 2D-McDA parameters
    data_dict_cal_2d_mcda = {}
    cal_2d_mcda_keys = [
        "Latitude",
        "Longitude",
        "Profile_ID",
        "Profile_Time",
        "Altitude",
        "Temperature",
        "Tropopause_Altitude_MERRA2",
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064",
        "Parallel_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Total_Attenuated_Backscatter_1064",
        "Particulate_Parallel_Attenuated_Backscatter_532",
        "Particulate_Perpendicular_Attenuated_Backscatter_532",
        "Particulate_Total_Attenuated_Backscatter_1064",
        "PSC_Ice_Mixture_Boundary",
        "Homogeneous_Chunks_Mean_Particulate_Perpendicular_Attenuated_Backscatter_532",
        "Homogeneous_Chunks_Mean_Total_Attenuated_Scattering_Ratio_532",
        "Homogeneous_Chunks_Mean_PSC_Ice_Mixture_Boundary",
        "Homogeneous_Chunks_Mean_Temperature",
        "PSC_Composition"]
    for key in cal_2d_mcda_keys:
        if key not in ds_2d_mcda:
            raise KeyError(f"Variable '{key}' not found in file")

        var = ds_2d_mcda[key]

        # 1D variables along Profile_ID
        if "Profile_ID" in var.dims and var.ndim == 1:
            data_dict_cal_2d_mcda[key] = var.values[prof_min:prof_max+1]

        # 2D variables (Profile_ID, Altitude)
        elif "Profile_ID" in var.dims and "Altitude" in var.dims:
            data_dict_cal_2d_mcda[key] = var.values[prof_min:prof_max+1, :]

        # Altitude coordinate (1D)
        elif key == "Altitude":
            data_dict_cal_2d_mcda[key] = var.values

        else:
            # Fallback: take everything
            data_dict_cal_2d_mcda[key] = var.values


    # **********************************
    # *** Get composite channel mask ***
    print("\n\n*****Get composite channel mask...*****")
    
    # Composite channel mask
    det_par  = data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"]
    det_perp = data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"]
    det_1064 = data_dict_cal_2d_mcda["Detection_Flags_1064"]

    composite_mask_channel = (
        (det_par  >= 1).astype(int) * 1  +
        (det_perp >= 1).astype(int) * 2 +
        (det_1064 >= 1).astype(int) * 4
    )

    # Detect pixels invalid in all channels
    invalid_all = (
        (det_par  == 255) &
        (det_perp == 255) &
        (det_1064 == 255)
    )

    # Assign invalid value
    composite_mask_channel[invalid_all] = 255


    # *******************************
    # *** Load PSC mask data file ***
    print("\n*****Load PSC mask data file...*****")
    
    # Get filename and filepath
    granule_date_dict = split_granule_date(GRANULE_DATE)
    filename_psc = f"CAL_LID_L2_PSCMask-{TYPE_CAL_LID_L2_PSCMask}-{VERSION_CAL_LID_L2_PSCMask.replace('.', '-')}." \
                   f"{granule_date_dict['year']}-{granule_date_dict['month']:02d}-{granule_date_dict['day']:02d}T00-00-00ZN.hdf"
    hdffile = os.path.join(INDATA_FOLDER_PSCMask, f"PSCMask.{VERSION_CAL_LID_L2_PSCMask.replace('V', 'v')}",
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
    lat_granule_v3 = cal_psc.get_data("Latitude")[granule_start_index:granule_end_index+1]
    lon_granule_v3 = cal_psc.get_data("Longitude")[granule_start_index:granule_end_index+1]
    prof_min_granule, prof_max_granule = get_prof_min_max_indexes_from_lon(lon_granule_v3, lon_granule_2d_mcda_psc_min, lon_granule_2d_mcda_psc_max) 
    prof_min_v3 = prof_min_granule + granule_start_index
    prof_max_v3 = prof_max_granule + granule_start_index

    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {prof_min_v3:d} "
          f"(lat = {lat_granule_v3[prof_min_granule]:.2f} / lon = {lon_granule_v3[prof_min_granule]:.2f}) "
          f"to max profile index {prof_max_v3:d} "
          f"(lat = {lat_granule_v3[prof_max_granule]:.2f} / lon = {lon_granule_v3[prof_max_granule]:.2f})")
    
    # Load parameters
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
        data_dict_cal_psc[key] = cal_psc.get_data(key, prof_min_v3, prof_max_v3, 'profindex')
    
    lat = data_dict_cal_psc["Latitude"] 
    lon = data_dict_cal_psc["Longitude"]
    alt = data_dict_cal_psc["Altitude"]


    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    plot_fig.filetype = FIGURES_FILETYPE
    plot_fig.set_head_filename(GRANULE_DATE, lon_granule[prof_min], lon_granule[prof_max])
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_max_detect_level(MAX_DETECT_LEVEL)
    plot_fig.set_coordinates(data_dict_cal_2d_mcda["Latitude"], data_dict_cal_2d_mcda["Longitude"],
                             data_dict_cal_2d_mcda["Altitude"])

    # Figure style
    setstyle("ticks_nogrid")

    # Create figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 3, hspace=0.5, wspace=0.3)

    def create_ax_cax(fig, gs_cell, width_ratios=[100, 1], wspace=0.1):
        subgs = gs_cell.subgridspec(
            1, 2,
            width_ratios=width_ratios,
            wspace=wspace
        )
        ax = fig.add_subplot(subgs[0])
        cax = fig.add_subplot(subgs[1])
        return ax, cax

    # PAB 532 par
    ax, cax = create_ax_cax(fig, gs[0, 0])
    plot_fig.plot_ab_signal(
        ax, cax,
        data_dict_cal_2d_mcda["Particulate_Parallel_Attenuated_Backscatter_532"],
        r"532 nm Particulate Parallel Attenuated Backscatter"
    )

    # PAB 532 per
    ax, cax = create_ax_cax(fig, gs[0, 1])
    plot_fig.plot_ab_signal(
        ax, cax,
        data_dict_cal_2d_mcda["Particulate_Perpendicular_Attenuated_Backscatter_532"],
        r"532 nm Particulate Perpendicular Attenuated Backscatter"
    )

    # PAB 1064
    ax, cax = create_ax_cax(fig, gs[0, 2])
    plot_fig.plot_ab_signal(
        ax, cax,
        data_dict_cal_2d_mcda["Particulate_Total_Attenuated_Backscatter_1064"],
        r"1064 nm Particulate Attenuated Backscatter"
    )

    # 532 par mask
    ax, cax = create_ax_cax(fig, gs[1, 0])
    plot_fig.plot_mask(
        ax, cax,
        data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"],
        '532_par'
    )

    # 532 per mask
    ax, cax = create_ax_cax(fig, gs[1, 1])
    plot_fig.plot_mask(
        ax, cax,
        data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"],
        '532_per'
    )

    # 1064 mask
    ax, cax = create_ax_cax(fig, gs[1, 2])
    plot_fig.plot_mask(
        ax, cax,
        data_dict_cal_2d_mcda["Detection_Flags_1064"],
        '1064'
    )

    # Homogeneous SR
    ax, cax = create_ax_cax(fig, gs[2, 0])
    plot_fig.plot_sr_signal(
        ax, cax,
        data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Total_Attenuated_Scattering_Ratio_532"],
        r"Mean 532 nm Total Attenuated Scattering Ratio Over Homogeneous Chunks"
    )

    # Homogeneous PAB 532 per
    ax, cax = create_ax_cax(fig, gs[2, 1])
    plot_fig.plot_ab_signal(
        ax, cax,
        data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Particulate_Perpendicular_Attenuated_Backscatter_532"],
        r"Mean 532 nm Particulate Perpendicular Attenuated Backscatter Over Homogeneous Chunks"
    )

    # Temperature
    ax, cax = create_ax_cax(fig, gs[2, 2])
    plot_fig.plot_temperature(
        ax, cax,
        data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Temperature"],
        r"Mean Temperature Over Homogeneous Chunks"
    )

    # Composite mask
    ax, cax = create_ax_cax(fig, gs[3, 0])
    plot_fig.plot_composite_mask_channel(
        ax, cax,
        composite_mask_channel
    )

    # PSC composition
    ax, cax = create_ax_cax(fig, gs[3, 1])
    plot_fig.plot_psc_composition(
        ax, cax,
        data_dict_cal_2d_mcda["PSC_Composition"],
        data_dict_cal_2d_mcda["Temperature"],
        f'2D-McDA-PSC {VERSION_2D_McDA} PSC composition',
        PLOT_TEMPERATURE_CONTOURS_OVER_COMPOSITION,
        PLOT_TROPOPAUSE,
    )

    plot_fig.set_coordinates(data_dict_cal_psc["Latitude"], data_dict_cal_psc["Longitude"],
                             data_dict_cal_psc["Altitude"])
    
    # PSC composition V3
    ax, cax = create_ax_cax(fig, gs[3, 2])
    plot_fig.plot_psc_composition(
        ax, cax,
        data_dict_cal_psc["PSC_Composition"],
        data_dict_cal_2d_mcda["Temperature"],
        f'PSC Mask {VERSION_CAL_LID_L2_PSCMask} PSC composition',
        PLOT_TEMPERATURE_CONTOURS_OVER_COMPOSITION,
        False
    )

    plt.suptitle(f"{GRANULE_DATE}_lon_{lon_granule[prof_min]:.2f}_{lon_granule[prof_max]:.2f}", 
                 fontweight='bold', 
                 fontsize=14, 
                 y=0.99)

    plt.subplots_adjust(left=0.02, bottom=0.03, right=0.95, top=0.92)
    filename = f"{GRANULE_DATE}_lon_{lon_granule[prof_min]:.2f}_{lon_granule[prof_max]:.2f}_2D-McDA-PSC-{VERSION_2D_McDA}_vs_PSCMask-V3.png"
    os.makedirs(FIGURES_PATH, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_PATH, filename), format="png", dpi=300, transparent=False)
    print("\t%s saved" % filename)

    # Close figure
    plt.close(fig)


    print_time(tic_main_program)



