#!/usr/bin/env python
# coding: utf8
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import cartopy
import cartopy.crs as ccrs
import copy

from my_modules.standard_outputs import print_time
from my_modules.readers.calipso_reader import CALIOPReader
from my_modules.figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from my_modules.geotools import UTC_time_CALIPSO, geo_distance
from my_modules.calipso_constants import *


class FigureMaker(CALIOPFigureMaker):

    def plot_column_optical_depth_aerosol(self, cod_532, ucod_532, cod_1064, ucod_1064):
        
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        ax.grid(which='major', lw=0.75, zorder=1)
        ax.grid(which='minor', lw=0.25, zorder=1)
        plt.plot(self.pindex, cod_532, '#8DBB25', label="532 nm", zorder=3)
        plt.plot(self.pindex, cod_1064, '#E32322', label="1064 nm", zorder=3)
        plt.fill_between(self.pindex, cod_532+ucod_532, cod_532-ucod_532, color='#8DBB25', edgecolor=None, alpha=0.5, zorder=2)
        plt.fill_between(self.pindex, cod_1064+ucod_1064, cod_1064-ucod_1064, color='#E32322', edgecolor=None, alpha=0.5, zorder=2)
        plt.fill_between(self.pindex, 0, 0, color='0.7', edgecolor=None, label="Uncertainty")
        plt.legend(bbox_to_anchor=(1., 1.), loc=2, borderaxespad=2., fontsize=7)
        plt.ylabel(r"$\tau$")
        
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=True)
        plt.title("Aerosol Column Optical Depth", fontweight='bold', y=1.35)
    
        plt.ylim(0, 0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        
        # Save figure
        filename = "aerosol_COD"
        self.save_fig(filename, adjust=(None, None, 0.79, None))

        # Close figure
        plt.close(fig)
    
    
    def plot_surface_iab(self, surf_iab_532, surf_iab_1064):
        
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        ax.grid(which='major', lw=0.75, zorder=1)
        ax.grid(which='minor', lw=0.25, zorder=1)
        plt.plot(self.pindex, surf_iab_532, '#8DBB25', label="532 nm", zorder=3)
        plt.plot(self.pindex, surf_iab_1064, '#E32322', label="1064 nm", zorder=3)
        plt.legend(bbox_to_anchor=(1., 1.), loc=2, borderaxespad=2., fontsize=7)
        plt.ylabel(r"$\gamma'$")
        
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=True)
        plt.title("Surface Integrated Attenuated Backscatter", fontweight='bold', y=1.35)
        
        # Save figure
        filename = "surface_IAB"
        self.save_fig(filename, adjust=(None, None, 0.79, None))

        # Close figure
        plt.close(fig)
        
    
    def plot_surface_iacr(self, surf_iacr_532, surf_iacr_1064):
        
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        ax.grid(which='major', lw=0.75, zorder=1)
        ax.grid(which='minor', lw=0.25, zorder=1)
        plt.plot(self.pindex, surf_iacr_532, '#8DBB25', label="On 532 nm\nsurface range\ndetection", zorder=3)
        plt.plot(self.pindex, surf_iacr_1064, '#E32322', label="On 1064 nm\nsurface range\ndetection", zorder=3)
        plt.legend(bbox_to_anchor=(1., 1.), loc=2, borderaxespad=2., fontsize=7)
        plt.ylabel(r"$\chi'$")
        
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=True)
        plt.title("Surface Integrated Attenuated Color Ratio " + r"$\chi'=\langle\beta'_{1064}\rangle/\langle\beta'_{532}\rangle$", fontweight='bold', y=1.35)
        
        plt.ylim(0, 5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        
        # Save figure
        filename = "surface_IACR"
        self.save_fig(filename, adjust=(None, None, 0.79, None))

        # Close figure
        plt.close(fig)
        
    
    def plot_surface_iacr_corr(self, surf_iacr, surf_iacr_corr):
        
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        ax.grid(which='major', lw=0.75, zorder=1)
        ax.grid(which='minor', lw=0.25, zorder=1)
        plt.plot(self.pindex, surf_iacr, '#2A71AF', label="Initial", ls='--', zorder=3)
        # plt.plot(self.pindex, surf_iacr_corr, '#2A71AF', label="Corrected from\ncolumn AOD", zorder=3)
        plt.legend(bbox_to_anchor=(1., 1.), loc=2, borderaxespad=2., fontsize=7)
        plt.ylabel(r"$\chi'$")
        
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=True)
        plt.title("Surface Integrated Attenuated Color Ratio " + r"$\chi'=\langle\beta'_{1064}\rangle/\langle\beta'_{532}\rangle$", fontweight='bold', y=1.35)
        
        plt.ylim(0, 5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        
        # Save figure
        filename = "surface_IACR_corrected_from_AOD"
        self.save_fig(filename, adjust=(None, None, 0.79, None))

        # Close figure
        plt.close(fig)
    
    
    def plot_surface_altitudes(self, top_532, top_1064, base_532, base_1064):
        
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        ax.grid(which='major', lw=0.75, zorder=1)
        ax.grid(which='minor', lw=0.25, zorder=1)
        plt.plot(self.pindex, top_532, '#8DBB25', ls='--', label="532 nm top", zorder=3)
        plt.plot(self.pindex, top_1064, '#E32322', ls='--', label="1064 nm top", zorder=3)
        plt.plot(self.pindex, base_532, '#8DBB25', label="532 nm base", zorder=3)
        plt.plot(self.pindex, base_1064, '#E32322', label="1064 nm base", zorder=3)
        plt.legend(bbox_to_anchor=(1., 1.), loc=2, borderaxespad=2., fontsize=7)
        plt.ylabel(r"$Z_{Surface}$")
        
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=True)
        plt.title("Surface altitudes", fontweight='bold', y=1.35)
        
        # Save figure
        filename = "surface_altitudes"
        self.save_fig(filename, adjust=(None, None, 0.79, None))

        # Close figure
        plt.close(fig)
        
        
if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # Data configuration
    GRANULE_DATE = "2021-07-18T02-50-53ZN"
    VERSION_CAL_LID_L2 = "V4.21"
    TYPE_CAL_LID_L2 = "Standard"
    SLICE_START_END_TYPE = 'longitude' # 'profindex' or 'longitude'
    SLICE_START = 3.15 # profindex or longitude
    SLICE_END = 0.06 # profindex or longitude
    #-----------------------------------------------------------------------
    # Plot configuration
    INVERT_XAXIS = False
    FIGURES_PATH = os.path.join("..", "out", "figures")
    #-----------------------------------------------------------------------
    # Plot flags
    PLOT_COLUMN_OPTICAL_DEPTH_TROPO_AEROSOL = True
    PLOT_SURFACE_IAB = True
    PLOT_SURFACE_IACR = True
    PLOT_SURFACE_IACR_CORRECTED_FROM_AOD = True
    PLOT_SURFACE_ALT = True
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    if TYPE_CAL_LID_L2 == "Prov":
        cod_var_prefix = "Column_Optical_Depth"
    else:
        cod_var_prefix = "Column_Optical_Depth_Tropospheric"
    
    
    # *********************************
    # *** Load CALIOP L2 layer data ***
    print("\n*****Load CALIOP L2 layer data...*****")
    
    cal_l2_5km = CALIOPReader(product='L2_05kmALay',
                              version=VERSION_CAL_LID_L2,
                              data_type=TYPE_CAL_LID_L2,
                              granule_date=GRANULE_DATE,
                              slice_start=SLICE_START,
                              slice_end=SLICE_END,
                              slice_start_end_type=SLICE_START_END_TYPE)
    
    cal_l2_layer_keys = ["Latitude",
                         "Longitude",
                         f"{cod_var_prefix}_Aerosols_532",
                         f"{cod_var_prefix}_Aerosols_1064",
                         f"{cod_var_prefix}_Aerosols_Uncertainty_532",
                         f"{cod_var_prefix}_Aerosols_Uncertainty_1064",
                         "Surface_Integrated_Attenuated_Backscatter_532",
                         "Surface_Integrated_Attenuated_Backscatter_1064",
                         "Surface_532_Integrated_Attenuated_Color_Ratio",
                         "Surface_1064_Integrated_Attenuated_Color_Ratio",
                         "Surface_Integrated_Attenuated_Backscatter_532",
                         "Surface_Integrated_Attenuated_Backscatter_1064",
                         "Surface_Top_Altitude_532",
                         "Surface_Top_Altitude_1064",
                         "Surface_Base_Altitude_532",
                         "Surface_Base_Altitude_1064"]
    data_dict_cal_lid_l2_5km = {}
    for key in cal_l2_layer_keys:
        data_dict_cal_lid_l2_5km[key] = cal_l2_5km.get_data(key)
    
    # ************
    # *** Plot ***
    print("\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, cal_l2_5km.lon_min, cal_l2_5km.lon_max)
    plot_fig.set_head_filename(GRANULE_DATE, cal_l2_5km.lon_min, cal_l2_5km.lon_max)
    plot_fig.set_coordinates(data_dict_cal_lid_l2_5km["Latitude"][:, 1],
                             data_dict_cal_lid_l2_5km["Longitude"][:, 1])
    
    # Tropospheric aerosol COD
    if PLOT_COLUMN_OPTICAL_DEPTH_TROPO_AEROSOL:
        plot_fig.plot_column_optical_depth_aerosol(data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_532"],
                                                   data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_Uncertainty_532"],
                                                   data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_1064"],
                                                   data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_Uncertainty_1064"])
        
    # Surface IAB
    if PLOT_SURFACE_IAB:
        plot_fig.plot_surface_iab(data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_532"],
                                  data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_1064"])
        
    # Surface IACR
    if PLOT_SURFACE_IACR:
        plot_fig.plot_surface_iacr(data_dict_cal_lid_l2_5km["Surface_532_Integrated_Attenuated_Color_Ratio"],
                                   data_dict_cal_lid_l2_5km["Surface_1064_Integrated_Attenuated_Color_Ratio"])
    
    # Surface IACR corrected from column AOD
    if PLOT_SURFACE_IACR_CORRECTED_FROM_AOD:
        iab_532_corrected_from_aod = data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_532"]/\
                                     np.exp(-data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_532"])
        iab_1064_corrected_from_aod = data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_1064"]/\
                                      np.exp(-data_dict_cal_lid_l2_5km[f"{cod_var_prefix}_Aerosols_1064"])
        surf_iacr = data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_1064"]/data_dict_cal_lid_l2_5km["Surface_Integrated_Attenuated_Backscatter_532"]
        surf_iacr_corrected_from_aod = iab_1064_corrected_from_aod/iab_532_corrected_from_aod
        plot_fig.plot_surface_iacr_corr(surf_iacr, surf_iacr_corrected_from_aod)
        
    # Surface altitudes
    if PLOT_SURFACE_ALT:
        plot_fig.plot_surface_altitudes(data_dict_cal_lid_l2_5km["Surface_Top_Altitude_532"],
                                        data_dict_cal_lid_l2_5km["Surface_Top_Altitude_1064"],
                                        data_dict_cal_lid_l2_5km["Surface_Base_Altitude_532"],
                                        data_dict_cal_lid_l2_5km["Surface_Base_Altitude_1064"])

    print_time(tic_main_program)