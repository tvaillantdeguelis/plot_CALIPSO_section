#!/usr/bin/env python
# coding: utf8
import os
import sys
import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors, BoundaryNorm
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import cartopy
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime
import copy
import re
import matplotlib.patheffects as pe
import cmocean
import cmlidar
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "my_modules"))

from standard_outputs import print_time
from readers.calipso_reader import CALIOPRegularGridReader
from figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges, interactive_pixel_info
from geotools import UTC_time_CALIPSO, geo_distance, get_monotical_lon, granule_date_decomposition
from calipso_constants import *


def load_yaml_configuration(config_path, required_keys):
    """Load one complete single-granule YAML configuration."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError(f"The YAML root must be a mapping: {path}")

    document = dict(document)
    case = document.pop("case", None)
    if not isinstance(case, Mapping):
        raise ValueError("The configuration must contain a 'case' mapping")

    required_case_keys = {"granule", "mode", "start", "end"}
    missing_case_keys = sorted(required_case_keys - case.keys())
    unknown_case_keys = sorted(case.keys() - required_case_keys - {"name"})
    if missing_case_keys:
        raise ValueError(f"Missing case keys: {', '.join(missing_case_keys)}")
    if unknown_case_keys:
        raise ValueError(f"Unknown case keys: {', '.join(unknown_case_keys)}")
    if case["mode"] not in {"longitude", "profindex"}:
        raise ValueError("case.mode must be either 'longitude' or 'profindex'")

    flattened = {}

    def flatten(mapping):
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                flatten(value)
                continue
            normalized_key = str(key).upper()
            if normalized_key in flattened:
                raise ValueError(f"Duplicate configuration key: {key}")
            flattened[normalized_key] = value

    flatten(document)
    required_keys = set(required_keys)
    missing = sorted(required_keys - flattened.keys())
    unknown = sorted(flattened.keys() - required_keys)
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")

    # A null limit means the first/last profile of the granule.
    slice_start = None if case["start"] is None else float(case["start"])
    slice_end = None if case["end"] is None else float(case["end"])
    if case["mode"] == "profindex":
        if any(limit is not None and not limit.is_integer()
               for limit in (slice_start, slice_end)):
            raise ValueError("Profile-index limits must be integers")
        slice_start = None if slice_start is None else int(slice_start)
        slice_end = None if slice_end is None else int(slice_end)

    flattened.update(
        GRANULE_DATE=str(case["granule"]),
        SLICE_START_END_TYPE=str(case["mode"]),
        SLICE_START=slice_start,
        SLICE_END=slice_end,
        CASE_STUDY_NAME=case.get("name"),
    )
    for key in ("FOLDER_PATH", "FIGURES_PATH"):
        value = flattened.get(key)
        if value and not Path(value).is_absolute():
            flattened[key] = str((PROJECT_ROOT / value).resolve())
    return flattened


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot a CALIOP section")
    parser.add_argument(
        "configuration",
        nargs="?",
        default=Path(__file__).with_name(
            f"{Path(__file__).stem}_single_granule.yaml"),
        help="Complete single-granule YAML configuration.",
    )
    return parser.parse_args()


def get_cal_l1_keys():
    
    cal_l1_keys = ["Latitude", "Longitude", "Lidar_Data_Altitudes", "Lidar_Data_Altitudes_init", "Profile_UTC_Time"]

    if PLOT_AB_532:
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")

    if PLOT_AB_532_HIST:
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")
        
    if PLOT_AB_532_PAR:
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        
    if PLOT_AB_532_PAR_HIST:
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        
    if PLOT_AB_532_PER:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        
    if PLOT_AB_532_PER_HIST:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        
    if PLOT_AB_1064:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        
    if PLOT_AB_1064_HIST:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        
    if PLOT_ACR:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")
        
    if PLOT_DR:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        
    if PLOT_AB_MOL_532:
        cal_l1_keys.append("Molecular_Total_Attenuated_Backscatter_532")
        
    if PLOT_AB_MOL_532_PAR:
        cal_l1_keys.append("Molecular_Parallel_Attenuated_Backscatter_532")
        
    if PLOT_AB_MOL_532_PER:
        cal_l1_keys.append("Molecular_Perpendicular_Attenuated_Backscatter_532")
        
    if PLOT_AB_MOL_1064:
        cal_l1_keys.append("Molecular_Attenuated_Backscatter_1064")
        
    if PLOT_ASR_532_STD:
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular")
        
    if PLOT_ASR_532_PAR_STD:
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel")
        
    if PLOT_ASR_532_PER_STD:
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular")
        
    if PLOT_ASR_1064_STD:
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064")
        
    if PLOT_ASR_532:
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Total_Attenuated_Backscatter_532")
        
    if PLOT_ASR_532_HIST:
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Total_Attenuated_Backscatter_532")
        
    if PLOT_ASR_532_PAR:
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Parallel_Attenuated_Backscatter_532")
        
    if PLOT_ASR_532_PAR_HIST:
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Parallel_Attenuated_Backscatter_532")
        
    if PLOT_ASR_532_PER:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Perpendicular_Attenuated_Backscatter_532")
        
    if PLOT_ASR_532_PER_HIST:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Perpendicular_Attenuated_Backscatter_532")
        
    if PLOT_ASR_1064:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        cal_l1_keys.append("Molecular_Attenuated_Backscatter_1064")
        
    if PLOT_ASR_1064_HIST:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        cal_l1_keys.append("Molecular_Attenuated_Backscatter_1064")
        
    if PLOT_ASR_532_ABOVE_STD:
        cal_l1_keys.append("Total_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Total_Attenuated_Backscatter_532")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular")
        
    if PLOT_ASR_532_PAR_ABOVE_STD:
        cal_l1_keys.append("Parallel_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Parallel_Attenuated_Backscatter_532")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel")
        
    if PLOT_ASR_532_PER_ABOVE_STD:
        cal_l1_keys.append("Perpendicular_Attenuated_Backscatter_532")
        cal_l1_keys.append("Molecular_Perpendicular_Attenuated_Backscatter_532")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular")
        
    if PLOT_ASR_1064_ABOVE_STD:
        cal_l1_keys.append("Attenuated_Backscatter_1064")
        cal_l1_keys.append("Molecular_Attenuated_Backscatter_1064")
        cal_l1_keys.append("Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064")
        
    if PLOT_PARAMS_532_PAR:
        cal_l1_keys.append("Laser_Energy_532")
        cal_l1_keys.append("Calibration_Constant_532")
        cal_l1_keys.append("Parallel_Amplifier_Gain_532")
        cal_l1_keys.append("Parallel_RMS_Baseline_532")
        cal_l1_keys.append("Noise_Scale_Factor_532_Parallel")
        
    if PLOT_PARAMS_532_PER:
        cal_l1_keys.append("Laser_Energy_532")
        cal_l1_keys.append("Calibration_Constant_532")
        cal_l1_keys.append("Perpendicular_Amplifier_Gain_532")
        cal_l1_keys.append("Perpendicular_RMS_Baseline_532")
        cal_l1_keys.append("Noise_Scale_Factor_532_Perpendicular")
        
    if PLOT_PARAMS_1064:
        cal_l1_keys.append("Laser_Energy_1064")
        cal_l1_keys.append("Calibration_Constant_1064")
        cal_l1_keys.append("Amplifier_Gain_1064")
        cal_l1_keys.append("RMS_Baseline_1064")
        cal_l1_keys.append("Noise_Scale_Factor_1064")
        
    if PLOT_NB_BINS_SHIFT:
        cal_l1_keys.append("Number_Bins_Shift")
    
    # Remove duplicates
    cal_l1_keys = list(dict.fromkeys(cal_l1_keys))
    
    return cal_l1_keys


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.3
        self.clabelpad = 50
        self.colorbar_position = 'right' # 'right' or 'bottom

    def plot_map(self, lat_granule, lon_granule, prof_UTC_time):
        """Plot CALIPSO track on a map"""
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            prof_UTC_time = remove_edges(prof_UTC_time, EDGES_REMOVAL)
        
        # Get start and end UTC times
        start_UTC_time = UTC_time_CALIPSO(prof_UTC_time[0])
        end_UTC_time = UTC_time_CALIPSO(prof_UTC_time[-1])

        # Figure style
        setstyle("ticks_nogrid")

        # Get mid lat/lon
        lat_0 = np.squeeze(self.lat[int(self.lat.size / 2)])
        lon_0 = np.squeeze(self.lon[int(self.lon.size / 2)])

        # # Get monotical longitudes
        lon_granule_plot = get_monotical_lon(lon_granule)
        lon_plot = get_monotical_lon(self.lon)

        # Figure
        fig = plt.figure(figsize=(cm2in(15), cm2in(15)))
        ax_proj = ccrs.Orthographic(lon_0, lat_0)
        # ax_proj = ccrs.NearsidePerspective(lon_0, lat_0)
        ax = plt.axes(projection=ax_proj)
        ax.coastlines(rasterized=True)
        ax.add_feature(cartopy.feature.LAKES, edgecolor='k')
        ax.stock_img()
        # Add night shadow
        year, month, day, hour, minute, second = np.asarray(re.findall(r'\d+', start_UTC_time), dtype=int)
        start_date = datetime.datetime(year, month, day, hour, minute, second)
        ax.add_feature(Nightshade(start_date, alpha=0.2), rasterized=True)
        year, month, day, hour, minute, second = np.asarray(re.findall(r'\d+', end_UTC_time), dtype=int)
        end_date = datetime.datetime(year, month, day, hour, minute, second)
        ax.add_feature(Nightshade(end_date, alpha=0.2), rasterized=True)
        # Add lat-lon grid
        gl = ax.gridlines(color='k', linewidth=0.5, linestyle='--', alpha=0.2, rasterized=True)
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)
        gl = ax.gridlines(color='k', linewidth=1., linestyle='-', alpha=0.2, rasterized=True)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(30)
        ax.set_global()
        # Plot
        plt.plot(lon_granule_plot, lat_granule, c='b', lw=4, alpha=0.1, transform=ccrs.PlateCarree(), rasterized=True)
        _, _, _, _, _, _, day_night_flag = granule_date_decomposition(GRANULE_DATE)
        if day_night_flag == 'ZN':
            sat_track_color = '#d92409'
        else:
            sat_track_color = '#a81c07'
        plt.plot(lon_plot, self.lat, c=sat_track_color, lw=4, alpha=1, transform=ccrs.PlateCarree(), rasterized=True)
        if CASE_STUDY_NAME:
            print(CASE_STUDY_NAME)
            title = f"{CASE_STUDY_NAME}\n{self.granule_date}\n{start_UTC_time} – {end_UTC_time}"
            ypos = 0.93
            adjust_top = 0.87
        else:
            title = f"{self.granule_date}\n{start_UTC_time} – {end_UTC_time}"
            ypos = 0.95
            adjust_top = 0.9
        ax.text(0.5, ypos, title, weight='bold', ha='center', va='center',fontsize=16, transform=fig.transFigure)

        # # Save figure (for test_colorbar)
        # self.fig_folder = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/test_colorbar/"
        # filename = f"0map"
        # self.save_fig(filename, transparent=True, adjust=(0.02, 0.02, 0.98, 0.8))

        # Save figure
        filename = f"map"
        self.save_fig(filename, transparent=False, adjust=(0.02, 0.02, 0.98, adjust_top))

        # Close figure
        plt.close(fig)


    def plot_attenuated_backscatter(self, atb, wl, polar, grid):

        atb2 = np.ma.copy(atb)
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            atb2 = remove_edges(atb2, EDGES_REMOVAL)
            
        # Put negative values to 1e-9 in order to plot with LogNorm
        if False:
            atb2[atb2 < 0] = 1e-9

        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0], facecolor='k')

        # colormap_style = 1 # 1: CALIOP-like
        #                    # 2: CALIOP browse colorblind
        #                    # 3: CALIOP browse colorblind linear
        #                    # 4*: CALIOP browse colorblind linear with white
        #                    # 5: CALIOP-like colorblind linear fewer bins
        #                    # 6: CALIOP-like colorblind
        #                    # 7: few color, easy to read
        #                    # else: LogNorm (viridis)
        if COLORMAP == "LEGACY":
            colormap_style = 1
        elif COLORMAP == "FRIENDLY":
            colormap_style = 1000
        else:
            colormap_style = 0

        if colormap_style == 1: # CALIOP-like colormap
            my_cmap = takecmap('caliop_browse_both', 35)
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 8+.01, 0.5)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.arange(nb_colors))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap, norm=my_norm,
                                rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 2: # CALIOP-like colormap colorblind
            my_cmap = takecmap('caliop_browse_colorblind_both', 35)
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 8+.01, 0.5)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.arange(nb_colors))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 3: # CALIOP-like colormap colorblind linear
            # my_cmap = copy.copy(cm.viridis)
            my_cmap = takecmap('extviridis')
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 8+.01, 0.5)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 4: # CALIOP-like colormap colorblind linear with white
            # my_cmap = copy.copy(cm.viridis)
            my_cmap = takecmap('extviridis')
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 8+.01, 0.5)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = len(bounds) + 1 - 8 # 8 yellow-white colors added manually
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ colors,
                            [[255/255., 251/255., 217/255., 1.]],
                            [[255/255., 252/255., 222/255., 1.]],
                            [[255/255., 253/255., 228/255., 1.]],
                            [[255/255., 253/255., 233/255., 1.]],
                            [[255/255., 253/255., 238/255., 1.]],
                            [[255/255., 254/255., 244/255., 1.]],
                            [[255/255., 254/255., 249/255., 1.]],
                            [[255/255., 255/255., 255/255., 1.]] ]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 5: # CALIOP-like colormap colorblind linear fewer bins
            # my_cmap = copy.copy(cm.viridis)
            my_cmap = takecmap('extviridis')
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 8+.01, 0.5)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = 12
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors_repeat = [colors[0],
                            colors[0], colors[0], colors[0],
                            colors[1], colors[1], colors[1],
                            colors[2], colors[2], colors[2],
                            colors[3], colors[3], colors[3],
                            colors[4], colors[4], colors[4],
                            colors[5], colors[5], colors[5],
                            colors[6], colors[6], colors[6],
                            colors[7], colors[7], colors[7],
                            colors[8], colors[8], colors[8],
                            colors[9], colors[9], colors[9],
                            colors[10], colors[10], colors[10],
                            colors[11]]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors_repeat, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 6: # CALIOP-like colorblind colormap
            my_cmap = takecmap('caliop_colorblind_both', 29)
            b1 = np.arange(1, 9+.01, 1)*1e-4
            b2 = np.arange(1, 9+.01, 1)*1e-3
            b3 = np.arange(1, 9+.01, 1)*1e-2
            b4 = np.array((1e-1,))
            bounds = np.concatenate((b1, b2, b3, b4))
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.arange(nb_colors))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 7: # few color, easy to read
            my_cmap = takecmap('extviridis', nb_colors=7)
            bounds = np.array((1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2))
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.arange(nb_colors))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 8: # discrete LogNorm colormap
            my_cmap = takecmap('extviridis_black_white')
            bounds = np.logspace(-4, -1, 17)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=LogNorm(), rasterized=True)
            if polar=='per':
                plt.clim(1e-5, 1e-1)
            elif wl==1064:
                plt.clim(1e-5, 1e-1)
            else:
                plt.clim(1e-4, 1e-1)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 9: # extviridis with brown
            my_cmap = takecmap('extviridis_black_white')
            b1 = np.logspace(-5, -3, 4)
            b2 = np.logspace(-3, -2, 8)[1:]
            b3 = np.logspace(-2, -1, 2)[1:]
            bounds = np.concatenate((b1, b2, b3))
            nb_colors_viridis = b2.size + b3.size + 1
            colors_viridis = my_cmap(np.linspace(40, 255, nb_colors_viridis).astype(int))
            colors = np.r_[ [[0/255., 0/255., 0/255., 1.]],
                            [[89/255., 25/255., 0/255., 1.]],
                            [[92/255., 43/255., 47/255., 1.]],
                            [[87/255., 60/255., 90/255., 1.]],
                            colors_viridis]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 10: # CALIOP-like alternative
            # palette = ["#000000",
            #            "#2A3790",
            #            "#546bb4",
            #            "#77a2d9",
            #            "#99ddff",
            #            "#bd0026",
            #            "#f03b20",
            #            "#fd8d3c",
            #            "#feb24c",
            #            "#fed976",
            #            "#ffffb2",
            #            "#535353",
            #            "#959595",
            #            "#D8D8D8",
            #            "#ffffff"
            #           ]
            palette = ["#000000",
                       "#2A3790",
                       "#546bb4",
                       "#77a2d9",
                       "#99ddff",
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       "#535353",
                       "#959595",
                       "#D8D8D8",
                       "#ffffff"
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 5)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 11: # CALIOP-like alternative
            palette = ["#000000",
                       '#2a3790', 
                       '#4556a6', 
                       '#5b76bc', 
                       '#7197d2', 
                       '#85b9e8', 
                       '#99ddff',
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       "#535353",
                       "#959595",
                       "#D8D8D8",
                       "#ffffff"
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 7)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 12: # CALIOP-like alternative
            palette = ["#000000",
                       '#2a3790', 
                       '#4556a6', 
                       '#5b76bc', 
                       '#7197d2', 
                       '#85b9e8', 
                       '#99ddff',
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       '#535353', 
                       '#757575', 
                       '#989898', 
                       '#bababa', 
                       '#dddddd', 
                       '#ffffff'
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 7)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 6)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 13: # CALIOP-like alternative
            palette = ["#000000",
                       '#2a3790', '#2c3991', '#2e3b93', '#2f3c94', '#313e95', '#334097', '#344298', '#364499', '#38469a', '#39479c', '#3b499d', '#3c4b9e', '#3e4da0', '#3f4fa1', '#4151a2', '#4253a4', '#4455a5', '#4556a6', '#4758a8', '#485aa9', '#495caa', '#4b5eab', '#4c60ad', '#4e62ae', '#4f64af', '#5066b1', '#5268b2', '#536ab3', '#546cb5', '#566db6', '#576fb7', '#5871b9', '#5a73ba', '#5b75bb', '#5c77bd', '#5e79be', '#5f7bbf', '#607dc1', '#627fc2', '#6381c3', '#6483c5', '#6585c6', '#6787c7', '#6889c9', '#698bca', '#6b8dcb', '#6c8fcd', '#6d91ce', '#6e93cf', '#7095d1', '#7197d2', '#7299d3', '#739bd5', '#759ed6', '#76a0d8', '#77a2d9', '#78a4da', '#7aa6dc', '#7ba8dd', '#7caade', '#7dace0', '#7eaee1', '#80b0e2', '#81b2e4', '#82b4e5', '#83b6e6', '#85b9e8', '#86bbe9', '#87bdea', '#88bfec', '#89c1ed', '#8bc3ef', '#8cc5f0', '#8dc7f1', '#8ecaf3', '#8fccf4', '#91cef5', '#92d0f7', '#93d2f8', '#94d4fa', '#95d6fb', '#97d9fc', '#98dbfe', '#99ddff',
                       '#ffffb2', '#fffcb0', '#fefaae', '#fef7ac', '#fef5aa', '#fdf2a9', '#fdf0a7', '#fceda5', '#fceaa3', '#fce8a1', '#fbe59f', '#fbe39d', '#fae09c', '#fade9a', '#f9db98', '#f9d896', '#f8d694', '#f8d392', '#f7d191', '#f7ce8f', '#f6cc8d', '#f5c98b', '#f5c789', '#f4c488', '#f4c186', '#f3bf84', '#f2bc82', '#f2ba80', '#f1b77f', '#f0b57d', '#f0b27b', '#efaf79', '#eead78', '#eeaa76', '#eda874', '#eca572', '#eba371', '#eba06f', '#ea9d6d', '#e99b6b', '#e8986a', '#e79668', '#e79366', '#e69165', '#e58e63', '#e48b61', '#e3895f', '#e2865e', '#e2835c', '#e1815a', '#e07e59', '#df7c57', '#de7955', '#dd7654', '#dc7352', '#db7151', '#da6e4f', '#d96b4d', '#d8694c', '#d7664a', '#d66349', '#d56047', '#d45d45', '#d35a44', '#d25742', '#d15541', '#d0523f', '#cf4e3e', '#ce4b3c', '#cd483a', '#cc4539', '#cb4237', '#ca3e36', '#c93b34', '#c83733', '#c63331', '#c52f30', '#c42b2f', '#c3262d', '#c2212c', '#c11c2a', '#bf1529', '#be0c27', '#bd0026',
                       '#535353', '#555555', '#575757', '#595959', '#5a5a5a', '#5c5c5c', '#5e5e5e', '#606060', '#626262', '#646464', '#666666', '#686868', '#6a6a6a', '#6c6c6c', '#6d6d6d', '#6f6f6f', '#717171', '#737373', '#757575', '#777777', '#797979', '#7b7b7b', '#7d7d7d', '#7f7f7f', '#818181', '#838383', '#858585', '#878787', '#898989', '#8b8b8b', '#8d8d8d', '#8f8f8f', '#919191', '#939393', '#959595', '#979797', '#999999', '#9b9b9b', '#9d9d9d', '#a0a0a0', '#a2a2a2', '#a4a4a4', '#a6a6a6', '#a8a8a8', '#aaaaaa', '#acacac', '#aeaeae', '#b0b0b0', '#b2b2b2', '#b5b5b5', '#b7b7b7', '#b9b9b9', '#bbbbbb', '#bdbdbd', '#bfbfbf', '#c1c1c1', '#c3c3c3', '#c6c6c6', '#c8c8c8', '#cacaca', '#cccccc', '#cecece', '#d0d0d0', '#d3d3d3', '#d5d5d5', '#d7d7d7', '#d9d9d9', '#dbdbdb', '#dedede', '#e0e0e0', '#e2e2e2', '#e4e4e4', '#e6e6e6', '#e9e9e9', '#ebebeb', '#ededed', '#efefef', '#f2f2f2', '#f4f4f4', '#f6f6f6', '#f8f8f8', '#fafafa', '#fdfdfd', '#ffffff'
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 85)
            b2 = np.logspace(-3, -2, 85)[1:]
            b3 = np.logspace(-2, -1, 84)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style == 14: # CALIOP-like alternative
            palette = ["#000000",
                       '#000444', '#010846', '#030b48', '#050e4a', '#08104c', '#0b134f', '#0e1551', '#101753', '#131a55', '#151c57', '#171e59', '#1a215c', '#1c235e', '#1e2560', '#202862', '#222a65', '#242d67', '#262f69', '#28316b', '#2a346e', '#2c3670', '#2e3972', '#303b74', '#323e77', '#344079', '#36437b', '#37467d', '#394880', '#3b4b82', '#3d4d84', '#3f5087', '#415389', '#43558b', '#44588e', '#465b90', '#485d92', '#4a6095', '#4c6397', '#4e6599', '#4f689c', '#516b9e', '#536da1', '#5570a3', '#5773a5', '#5876a8', '#5a79aa', '#5c7bad', '#5e7eaf', '#6081b1', '#6284b4', '#6387b6', '#6589b9', '#678cbb', '#698fbe', '#6b92c0', '#6d95c3', '#6e98c5', '#709bc7', '#729eca', '#74a1cc', '#76a4cf', '#78a7d1', '#79a9d4', '#7bacd6', '#7dafd9', '#7fb2db', '#81b5de', '#83b8e0', '#85bbe3', '#86bee5', '#88c1e8', '#8ac4eb', '#8cc7ed', '#8ecbf0', '#90cef2', '#92d1f5', '#93d4f7', '#95d7fa', '#97dafc',
                       '#99ddff', '#a8e1f7', '#b6e4ef', '#c3e8e6', '#ceecde', '#d9efd5', '#e3f3cd', '#edf7c4', '#f6fbbb', '#ffffb2',
                       '#fffcb0', '#fff9ae', '#fef6ac', '#fef3aa', '#fdf1a8', '#fdeea6', '#fceba4', '#fce8a1', '#fbe59f', '#fbe29d', '#fae09b', '#fadd99', '#f9da97', '#f9d795', '#f8d493', '#f7d191', '#f7cf8f', '#f6cc8d', '#f6c98b', '#f5c689', '#f4c387', '#f4c085', '#f3be83', '#f2bb81', '#f1b87f', '#f1b57d', '#f0b27b', '#efaf79', '#eead77', '#eeaa76', '#eda774', '#eca472', '#eba170', '#ea9e6e', '#e99c6c', '#e9996a', '#e89668', '#e79366', '#e69064', '#e58d62', '#e48a61', '#e3875f', '#e2845d', '#e1825b', '#e07f59', '#df7c57', '#de7956', '#dd7654', '#dc7352', '#db7050', '#da6d4e', '#d96a4c', '#d8674b', '#d76449', '#d66047', '#d45d45', '#d35a44', '#d25742', '#d15440', '#d0503e', '#cf4d3d', '#ce493b', '#cc4639', '#cb4238', '#ca3e36', '#c93b34', '#c73633', '#c63231', '#c52e2f', '#c4292e', '#c2242c', '#c11e2b', '#c01629', '#be0d28',
                       '#bd0026', '#b31e2b', '#a82c30', '#9e3635', '#933d3a', '#88433f', '#7c4844', '#704c49', '#62504e', '#535353',
                       '#555555', '#575757', '#595959', '#5b5b5b', '#5d5d5d', '#5f5f5f', '#616161', '#636363', '#656565', '#676767', '#696969', '#6b6b6b', '#6d6d6d', '#6f6f6f', '#717171', '#737373', '#757575', '#777777', '#797979', '#7b7b7b', '#7d7d7d', '#7f7f7f', '#818181', '#848484', '#868686', '#888888', '#8a8a8a', '#8c8c8c', '#8e8e8e', '#909090', '#929292', '#949494', '#979797', '#999999', '#9b9b9b', '#9d9d9d', '#9f9f9f', '#a1a1a1', '#a4a4a4', '#a6a6a6', '#a8a8a8', '#aaaaaa', '#acacac', '#afafaf', '#b1b1b1', '#b3b3b3', '#b5b5b5', '#b8b8b8', '#bababa', '#bcbcbc', '#bebebe', '#c1c1c1', '#c3c3c3', '#c5c5c5', '#c7c7c7', '#cacaca', '#cccccc', '#cecece', '#d0d0d0', '#d3d3d3', '#d5d5d5', '#d7d7d7', '#dadada', '#dcdcdc', '#dedede', '#e0e0e0', '#e3e3e3', '#e5e5e5', '#e7e7e7', '#eaeaea', '#ececec', '#eeeeee', '#f1f1f1', '#f3f3f3', '#f6f6f6', '#f8f8f8', '#fafafa', '#fdfdfd', '#ffffff'
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 85)
            b2 = np.logspace(-3, -2, 85)[1:]
            b3 = np.logspace(-2, -1, 84)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style == 15: # CALIOP-like alternative
            palette = ["#000000",
                       "#2A3790",
                       "#327EB4",
                       "#45B4C1",
                       "#A0D8B3",
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       "#535353",
                       "#959595",
                       "#D8D8D8",
                       "#ffffff"
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 5)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 16: # CALIOP-like alternative
            palette = ["#000000",
                       "#2A3790",
                       "#546bb4",
                       "#77a2d9",
                       "#99ddff",
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       "#535353",
                       "#959595",
                       "#D8D8D8",
                       "#ffffff"
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 5)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 17: # CALIOP-like alternative
            palette = ["#000000",
                       "#2A3790",
                       "#546bb4",
                       "#77a2d9",
                       "#99ddff",
                       "#ffffb2",
                       "#fed976",
                       "#feb24c",
                       "#fd8d3c",
                       "#f03b20",
                       "#bd0026",
                       "#535353",
                       "#959595",
                       "#D8D8D8",
                       "#ffffff"
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-4, np.log10(5e-4), 5)
            b2 = np.logspace(np.log10(5e-4), -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 18: # CALIOP-like alternative
            palette = ['#000000', '#120016', '#24012c', '#350142', '#470e61', '#424086', '#306a8e', '#21918c', '#31b57b', '#7fd34e', '#e5e419', '#feec52', '#fef28b', '#fff9c4', '#fffffe'
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3, 5)
            b2 = np.logspace(-3, -2, 7)[1:]
            b3 = np.logspace(-2, -1, 4)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 19: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#211a7b', '#35318b', '#46489b', '#555fab', '#6477bc', '#7190cc', '#7fa9dd', '#8cc3ee', '#99ddff',
                       '#ffff00', '#fae514', '#f4ca1d', '#edb023', '#e59626', '#dc7a27', '#d25e28', '#c83d27', '#bd0026', '#8d413c',
                       '#535353', '#5d5d5d', '#666666', '#707070', '#7b7b7b', '#858585', '#8f8f8f', '#9a9a9a', '#a5a5a5', '#b0b0b0', '#bbbbbb', '#c6c6c6', '#d1d1d1', '#dcdcdc', '#e8e8e8', '#f3f3f3', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001 , 
                               0.0015, 0.002 , 0.0025, 0.003 , 0.0035, 0.004 , 0.0045, 0.005 , 0.0055, 0.006, 
                               0.0065, 0.007 , 0.0075, 0.008 , 0.0085, 0.009 , 0.0095, 0.01  , 0.02  , 0.03  , 0.04  , 0.05  , 0.06  , 0.07  , 0.08  , 0.09  , 0.1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 20: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#39378f', '#5c6bb3', '#7ca3d9', '#99ddff',
                       '#ffff00', '#f4ca1d', '#e59626', '#d25e28', '#bd0026',
                       '#535353', '#7b7b7b', '#a5a5a5', '#d1d1d1',
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 
                               0.002 , 0.003 , 0.004 , 0.005 , 0.006 , 
                               0.008 , 0.01  , 0.04  , 0.07  , 0.1])

            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 21: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#312d88', '#4f56a5', '#6981c2', '#82aee0', '#99ddff', '#c9ebd5',
                       '#f8f8ab', '#fcc33d', '#f48a06', '#de5213', '#bd0026',
                       '#696969', '#808080', '#989898', '#b1b1b1', '#cacaca', '#e4e4e4', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00004, 0.00007, 0.0001, 0.0004, 0.0007, 0.001, 
                               0.002, 0.003, 0.004, 0.005, 0.006,   
                               0.007, 0.008, 0.009, 0.01, 0.04, 0.07, 0.1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 22: # CALIOP-like alternative (discrete)
            palette = ['#000000', 
                       '#0f0e2b', '#181644', '#211f5d', '#2b2777', '#343389', '#3d4293', '#46519d', '#4e60a8', '#576fb2', '#607ebd', '#6a8dc7', '#739dd2', '#7caddd', '#86bce8', '#8fcdf4', '#99ddff', 
                       '#b4e5e7', '#ccecd2', '#e3f2be',
                       '#f8f8ab', '#fcc33d', '#f48a06', '#de5213', '#bd0026',
                       '#8b3e3a', '#4e4e4e', '#5c5c5c', '#6b6b6b', '#7a7a7a', '#8a8a8a', '#9a9a9a', '#aaaaaa', '#bbbbbb', '#cbcbcb', '#dcdcdc', '#eeeeee', 
                       '#ffffff']
            # palette = ['#2b2777', 
            #         '#33317f', '#3b3c87', '#42468f', '#495198', '#515ca0', '#5767a8', '#5e72b1', '#657db9', '#6c89c2', '#7295ca', '#79a0d3', '#7facdc', '#86b8e5', '#8cc4ed', '#93d1f6', '#99ddff',
            #         '#b4e5e7', '#ccecd2', '#e3f2be',
            #         '#f8f8ab', '#fcc33d', '#f48a06', '#de5213', '#bd0026',
            #         '#4e4e4e', '#484848', '#414141', '#3b3b3b', '#353535', '#2f2f2f', '#292929', '#232323', '#1e1e1e', '#181818', '#121212', '#0a0a0a', 
            #         '#000000']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.concatenate((np.arange(1, 10)*1e-5, np.arange(1, 10)*1e-4, np.arange(1, 10)*1e-3, np.arange(1, 10)*1e-2, np.array([1,])*1e-1))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 23: # CALIOP-like alternative (continuous)
            palette = ["#000000",
                       '#000444', '#010846', '#030b48', '#050e4a', '#08104c', '#0b134f', '#0e1551', '#101753', '#131a55', '#151c57', '#171e59', '#1a215c', '#1c235e', '#1e2560', '#202862', '#222a65', '#242d67', '#262f69', '#28316b', '#2a346e', '#2c3670', '#2e3972', '#303b74', '#323e77', '#344079', '#36437b', '#37467d', '#394880', '#3b4b82', '#3d4d84', '#3f5087', '#415389', '#43558b', '#44588e', '#465b90', '#485d92', '#4a6095', '#4c6397', '#4e6599', '#4f689c', '#516b9e', '#536da1', '#5570a3', '#5773a5', '#5876a8', '#5a79aa', '#5c7bad', '#5e7eaf', '#6081b1', '#6284b4', '#6387b6', '#6589b9', '#678cbb', '#698fbe', '#6b92c0', '#6d95c3', '#6e98c5', '#709bc7', '#729eca', '#74a1cc', '#76a4cf', '#78a7d1', '#79a9d4', '#7bacd6', '#7dafd9', '#7fb2db', '#81b5de', '#83b8e0', '#85bbe3', '#86bee5', '#88c1e8', '#8ac4eb', '#8cc7ed', '#8ecbf0', '#90cef2', '#92d1f5', '#93d4f7', '#95d7fa', '#97dafc',
                       '#99ddff', '#a8e1f7', '#b6e4ef', '#c3e8e6', '#ceecde', '#d9efd5', '#e3f3cd', '#edf7c4', '#f6fbbb', '#ffffb2',
                       '#fffcb0', '#fff9ae', '#fef6ac', '#fef3aa', '#fdf1a8', '#fdeea6', '#fceba4', '#fce8a1', '#fbe59f', '#fbe29d', '#fae09b', '#fadd99', '#f9da97', '#f9d795', '#f8d493', '#f7d191', '#f7cf8f', '#f6cc8d', '#f6c98b', '#f5c689', '#f4c387', '#f4c085', '#f3be83', '#f2bb81', '#f1b87f', '#f1b57d', '#f0b27b', '#efaf79', '#eead77', '#eeaa76', '#eda774', '#eca472', '#eba170', '#ea9e6e', '#e99c6c', '#e9996a', '#e89668', '#e79366', '#e69064', '#e58d62', '#e48a61', '#e3875f', '#e2845d', '#e1825b', '#e07f59', '#df7c57', '#de7956', '#dd7654', '#dc7352', '#db7050', '#da6d4e', '#d96a4c', '#d8674b', '#d76449', '#d66047', '#d45d45', '#d35a44', '#d25742', '#d15440', '#d0503e', '#cf4d3d', '#ce493b', '#cc4639', '#cb4238', '#ca3e36', '#c93b34', '#c73633', '#c63231', '#c52e2f', '#c4292e', '#c2242c', '#c11e2b', '#c01629', '#be0d28',
                       '#bd0026', '#b31e2b', '#a82c30', '#9e3635', '#933d3a', '#88433f', '#7c4844', '#704c49', '#62504e', '#535353',
                       '#555555', '#575757', '#595959', '#5b5b5b', '#5d5d5d', '#5f5f5f', '#616161', '#636363', '#656565', '#676767', '#696969', '#6b6b6b', '#6d6d6d', '#6f6f6f', '#717171', '#737373', '#757575', '#777777', '#797979', '#7b7b7b', '#7d7d7d', '#7f7f7f', '#818181', '#848484', '#868686', '#888888', '#8a8a8a', '#8c8c8c', '#8e8e8e', '#909090', '#929292', '#949494', '#979797', '#999999', '#9b9b9b', '#9d9d9d', '#9f9f9f', '#a1a1a1', '#a4a4a4', '#a6a6a6', '#a8a8a8', '#aaaaaa', '#acacac', '#afafaf', '#b1b1b1', '#b3b3b3', '#b5b5b5', '#b8b8b8', '#bababa', '#bcbcbc', '#bebebe', '#c1c1c1', '#c3c3c3', '#c5c5c5', '#c7c7c7', '#cacaca', '#cccccc', '#cecece', '#d0d0d0', '#d3d3d3', '#d5d5d5', '#d7d7d7', '#dadada', '#dcdcdc', '#dedede', '#e0e0e0', '#e3e3e3', '#e5e5e5', '#e7e7e7', '#eaeaea', '#ececec', '#eeeeee', '#f1f1f1', '#f3f3f3', '#f6f6f6', '#f8f8f8', '#fafafa', '#fdfdfd', '#ffffff'
                      ]
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -4+np.log10(8), 85)
            b2 = np.logspace(-4+np.log10(8), -3+np.log10(6.5), 85)[1:]
            b3 = np.logspace(-3+np.log10(6.5), -1, 84)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style == 24: # CALIOP-like alternative (continuous)
            palette_1 = ['#0000b3', '#0805b4', '#100bb5', '#1510b6', '#1a14b7', '#1e18b8', '#221cb9', '#251fba', '#2822ba', '#2a25bb', '#2d28bc', '#2f2bbd', '#322ebe', '#3431bf', '#3634c0', '#3836c1', '#3a39c2', '#3c3cc3', '#3d3ec4', '#3f41c5', '#4143c6', '#4246c7', '#4449c7', '#464bc8', '#474ec9', '#4850ca', '#4a53cb', '#4b55cc', '#4c58cd', '#4e5ace', '#4f5dcf', '#505fd0', '#5162d1', '#5264d2', '#5367d2', '#5469d3', '#556cd4', '#566ed5', '#5771d6', '#5873d7', '#5976d8', '#5a78d9', '#5b7bda', '#5b7ddb', '#5c80dc', '#5d82dc', '#5e85dd', '#5e88de', '#5f8adf', '#5f8de0', '#608fe1', '#6092e2', '#6194e3', '#6197e4', '#6199e5', '#629ce6', '#629fe6', '#62a1e7', '#62a4e8', '#62a6e9', '#62a9ea', '#62aceb', '#62aeec', '#62b1ed', '#62b4ee', '#62b6ef', '#61b9f0', '#61bbf1', '#60bef1', '#5fc1f2', '#5fc4f3', '#5dc6f4', '#5cc9f5', '#5bccf6', '#59cff7', '#57d1f8', '#55d4f9', '#52d7fa', '#4edafb', '#49ddfc', '#40e0fd', '#29e4ff', '#23e6fe', '#32e9fc', '#43ebf8', '#55edf3', '#65efec', '#75f1e5', '#84f2dd', '#92f4d3', '#9ff5c9', '#acf6be', '#b8f8b2', '#c4f9a6', '#cffa98', '#d9fb89', '#e3fc79', '#edfd68', '#f6fe53', '#ffff39']
            palette_2 = ['#fefb38', '#fdf837', '#fcf536', '#fbf135', '#faee34', '#f9ea33', '#f8e732', '#f6e431', '#f5e030', '#f4dd2f', '#f2da2e', '#f1d62d', '#efd32c', '#eed02c', '#eccd2b', '#ebc92a', '#e9c629', '#e7c328', '#e5c028', '#e4bd27', '#e2ba26', '#e0b726', '#deb425', '#dcb024', '#daad24', '#d8aa23', '#d6a723', '#d4a422', '#d2a121', '#cf9f21', '#cd9c20', '#cb9920', '#c9961f', '#c6931f', '#c4901e', '#c28d1e', '#bf8b1e', '#bd881d', '#bb851d', '#b8821c', '#b6801c', '#b37d1c', '#b07a1b', '#ae781b', '#ab751b', '#a9721a', '#a6701a', '#a36d1a', '#a16b19', '#9e6819', '#9b6619', '#986318', '#966118', '#935f18', '#905c17', '#8d5a17', '#8a5717', '#885516', '#855316', '#825016', '#7f4e16', '#7c4c15', '#794a15', '#764815', '#734515', '#704314', '#6d4114', '#6a3f14', '#673d13', '#643b13', '#613913', '#5e3713', '#5b3512', '#583312', '#553112', '#522f11', '#4f2d11', '#4c2c11', '#492a10', '#462810', '#432610', '#40240f', '#3d230f', '#3a210e', '#371f0d', '#341e0d', '#311c0c', '#2e1a0b', '#2b190a', '#281709', '#251608', '#221407', '#1f1206', '#1d1006', '#1a0e05', '#160b04', '#120903', '#0d0602', '#070301', '#000000']
            palette_3 = ['#040404', '#070707', '#0b0b0b', '#0e0e0e', '#111111', '#131313', '#151515', '#181818', '#191919', '#1b1b1b', '#1d1d1d', '#1f1f1f', '#222222', '#242424', '#262626', '#282828', '#2a2a2a', '#2c2c2c', '#2e2e2e', '#303030', '#323232', '#353535', '#373737', '#393939', '#3b3b3b', '#3e3e3e', '#404040', '#424242', '#444444', '#474747', '#494949', '#4b4b4b', '#4e4e4e', '#505050', '#525252', '#555555', '#575757', '#595959', '#5c5c5c', '#5e5e5e', '#616161', '#636363', '#666666', '#686868', '#6a6a6a', '#6d6d6d', '#6f6f6f', '#727272', '#747474', '#777777', '#797979', '#7c7c7c', '#7f7f7f', '#818181', '#848484', '#868686', '#898989', '#8b8b8b', '#8e8e8e', '#919191', '#939393', '#969696', '#989898', '#9b9b9b', '#9e9e9e', '#a0a0a0', '#a3a3a3', '#a6a6a6', '#a8a8a8', '#ababab', '#aeaeae', '#b0b0b0', '#b3b3b3', '#b6b6b6', '#b9b9b9', '#bbbbbb', '#bebebe', '#c1c1c1', '#c4c4c4', '#c6c6c6', '#c9c9c9', '#cccccc', '#cfcfcf', '#d1d1d1', '#d4d4d4', '#d7d7d7', '#dadada', '#dddddd', '#dfdfdf', '#e2e2e2', '#e5e5e5', '#e8e8e8', '#ebebeb', '#eeeeee', '#f1f1f1', '#f3f3f3', '#f6f6f6', '#f9f9f9', '#fcfcfc', '#ffffff']
            palette = palette_1 + palette_2 + palette_3
            print(len(palette_1))
            print(len(palette_2))
            print(len(palette_3))
            print(len(palette))
            my_cmap = mpl.colors.ListedColormap(palette)
            b1 = np.logspace(-5, -3+np.log10(2), len(palette_1)+1-1)
            b2 = np.logspace(-3+np.log10(2), -3+np.log10(5), len(palette_2)+1)[1:]
            b3 = np.logspace(-3+np.log10(5), -1, len(palette_3)+1-1)[1:]
            bounds = np.concatenate((b1, b2, b3))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style == 25: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#312d88', '#4f56a5', '#6981c2', '#82aee0', '#99ddff', '#c9ebd5',
                       '#f8f8ab', '#fcc33d', '#f48a06', '#de5213', '#bd0026',
                       '#696969', '#808080', '#989898', '#b1b1b1', '#cacaca', '#e4e4e4', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00004, 0.00007, 0.0001, 0.0004, 0.0007, 0.001, 
                               0.002, 0.003, 0.004, 0.005, 0.006,   
                               0.007, 0.008, 0.009, 0.01, 0.04, 0.07, 0.1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 26: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#150e73', '#221b7c', '#2d2884', '#37348d', '#404095', '#484c9e', '#5159a6', '#5965af', 
                       '#6072b8', '#687ec1', '#6f8bc9', '#7699d2', '#7da6db', '#84b4e4', '#8bc1ed', '#92cff6', '#99ddff',
                       '#f8f8ab', '#fcdd7b', '#fcc251', '#f9a72d', '#f48b07', '#e97011', '#dc541a', '#cd3620', '#bd0026',
                       '#696969', '#777777', '#868686', '#959595', '#a4a4a4', '#b4b4b4', '#c4c4c4', '#d4d4d4', '#e4e4e4', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.concatenate((np.linspace(1, 9, 9)*1e-5,
                                     np.linspace(1, 9, 9)*1e-4,
                                     np.linspace(1, 9, 9)*1e-3,
                                     np.linspace(1, 9, 9)*1e-2,
                                     np.array((0.1,))))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 27: # CALIOP-like alternative
            palette = ['#000000',
                       '#00006b', '#150e73', '#221b7c', '#2d2884', '#37348d', '#404095', '#484c9e', '#5159a6', '#5965af', 
                       '#6072b8', '#687ec1', '#6f8bc9', '#7699d2', '#7da6db', '#84b4e4', '#8bc1ed', '#92cff6', '#99ddff',
                       '#f8f8ab', '#fcdd7b', '#fcc251', '#f9a72d', '#f48b07', '#e97011', '#dc541a', '#cd3620', '#bd0026',
                       '#696969', '#777777', '#868686', '#959595', '#a4a4a4', '#b4b4b4', '#c4c4c4', '#d4d4d4', '#e4e4e4', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.concatenate((np.logspace(-5, -4, 10)[:-1],
                                     np.logspace(-4, -3, 10)[:-1],
                                     np.logspace(-3, -2, 10)[:-1],
                                     np.logspace(-2, -1, 10)[:-1],
                                     np.array((0.1,))))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 28: # CALIOP-like alternative
            # palette = ['#312d88', 
            #            '#4f56a5', '#6981c2', 
            #            '#82aee0', '#99ddff', 
            #            '#c9ebd5', '#f8f8ab', '#fcc33e', '#ff8400', '#ff1200', '#be0026', '#800000', '#303030', '#5e5e5e', 
            #            '#919191', '#c6c6c6', 
            #            '#ffffff']
            # palette = ['#000e61', 
            #            '#264189', '#4d75b0', '#73a9d8', '#99ddff',
            #            '#f8f8ab', '#fcc33e', '#ff8400', '#ff1200', '#be0026', '#800000', '#303030', '#4e4e4e', '#6b6b6b', 
            #            '#898989', '#a6a6a6', '#c4c4c4', '#e1e1e1', 
            #            '#ffffff']
            # palette = ['#000355', 
            #            '#1d2d75', '#395494', '#567db4', '#7ca9c8', 
            #            '#a5d5d8', '#f0ff33', '#f9c914', '#ff8400', '#ff2100', '#cf0002', '#880005', '#500000', '#000000', '#1a1a1a', 
            #            '#2c2c2c', '#404040', '#565656', '#6c6c6c', '#828282', '#9a9a9a', '#b2b2b2', '#cbcbcb', '#e5e5e5', 
            #            '#ffffff']
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#80aeca', 
                       '#bad9db', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', '#130000', '#212121', 
                       '#3b3b3b', '#4d4d4d', '#646464', '#787878', '#8e8e8e', '#a6a6a6', '#bdbdbd', '#d7d7d7', '#eeeeee', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 29: # CALIOP-like alternative (continuous)
            palette_1 = ['#000355', '#020657', '#040959', '#060b5b', '#080e5e', '#0a1160', '#0c1462', '#0d1664', '#0f1966', '#111c68', '#131e6a', '#15216c', '#16236e', '#182670', '#1a2872', '#1c2b73', '#1d2d75', '#1f3077', '#213279', '#23357b', '#24377d', '#263a7f', '#283c81', '#2a3f83', '#2c4185', '#2d4487', '#2f4689', '#31498b', '#334b8d', '#354e8f', '#365091', '#385393', '#3a5595', '#3c5897', '#3e5a99', '#3f5d9b', '#41609d', '#43629f', '#4565a1', '#4767a3', '#486aa5', '#4a6ca7', '#4c6fa9', '#4e72ab', '#5074ad', '#5277af', '#547ab1', '#557cb3', '#587fb5', '#5a82b6', '#5c84b7', '#5f87b9', '#618aba', '#638dbb', '#668fbc', '#6892be', '#6a95bf', '#6d97c0', '#6f9ac1', '#729dc3', '#74a0c4', '#76a3c5', '#79a5c6', '#7ba8c8', '#7eabc9', '#80aeca', '#83b0cb', '#85b3cc', '#88b6cd', '#8bb9ce', '#8dbccf', '#90bed0', '#93c1d1', '#95c4d2', '#98c7d3', '#9acad4', '#9dccd5', '#a0cfd6', '#a2d2d7', '#a5d5d8', '#b0d7da', '#bad9db', '#c4dcdd', '#cededf', '#d8e0e0']
            palette_2 = ['#e8e7c9', '#ecebb1', '#eff097', '#f0f57b', '#f1fa5c', '#f0ff33', '#f0ff33', '#f1fa30', '#f2f52d', '#f2f12b', '#f3ec28', '#f4e725', '#f5e222', '#f6dd1f', '#f7d81c', '#f7d31a', '#f8cd17', '#f9c814', '#fac312', '#fabe11', '#fab90f', '#fbb40e', '#fbae0c', '#fca90b', '#fca309', '#fd9e07', '#fd9806', '#fe9204', '#fe8c02', '#ff8500', '#ff7f00', '#ff7900', '#ff7200', '#ff6c00', '#ff6400', '#ff5d00', '#ff5500', '#ff4c00', '#ff4100', '#ff3600', '#ff2600', '#fb1f00', '#f61b00', '#f11701', '#ec1401', '#e71001', '#e10d01', '#dc0901', '#d70502', '#d20202', '#cc0002', '#c70002', '#c10003', '#bb0003', '#b60003', '#b00003', '#ab0004', '#a50004', '#a00004', '#9a0004', '#950004', '#900005', '#8a0005', '#850005', '#800004', '#7b0004', '#760003', '#710003', '#6b0002', '#660002', '#610002', '#5d0001', '#580001', '#530000', '#4e0000', '#490000', '#450000', '#400000', '#3b0000', '#370000', '#320000', '#2c0000', '#260000', '#1e0000', '#130000']
            palette_3 = ['#000000', '#040404', '#090909', '#0d0d0d', '#101010', '#131313', '#161616', '#181818', '#1a1a1a', '#1d1d1d', '#1f1f1f', '#212121', '#242424', '#262626', '#292929', '#2b2b2b', '#2e2e2e', '#303030', '#333333', '#353535', '#383838', '#3b3b3b', '#3d3d3d', '#404040', '#434343', '#454545', '#484848', '#4b4b4b', '#4d4d4d', '#505050', '#535353', '#565656', '#595959', '#5b5b5b', '#5e5e5e', '#616161', '#646464', '#676767', '#6a6a6a', '#6d6d6d', '#707070', '#727272', '#757575', '#787878', '#7b7b7b', '#7e7e7e', '#818181', '#848484', '#878787', '#8a8a8a', '#8e8e8e', '#919191', '#949494', '#979797', '#9a9a9a', '#9d9d9d', '#a0a0a0', '#a3a3a3', '#a6a6a6', '#a9a9a9', '#adadad', '#b0b0b0', '#b3b3b3', '#b6b6b6', '#b9b9b9', '#bdbdbd', '#c0c0c0', '#c3c3c3', '#c6c6c6', '#cacaca', '#cdcdcd', '#d0d0d0', '#d3d3d3', '#d7d7d7', '#dadada', '#dddddd', '#e1e1e1', '#e4e4e4', '#e7e7e7', '#ebebeb', '#eeeeee', '#f1f1f1', '#f5f5f5', '#f8f8f8', '#fcfcfc', '#ffffff']
            palette = palette_1 + palette_2 + palette_3
            print(len(palette))
            my_cmap = mpl.colors.ListedColormap(palette)
            nb_color_minus5_minus4 = int(len(palette_1)*2/5)
            b1 = np.linspace(1, 10, nb_color_minus5_minus4+1-1)*1e-5
            b2 = np.linspace(1, 15, len(palette_1)-nb_color_minus5_minus4+1)[1:]*1e-4
            b3 = np.linspace(1.5, 8.5, len(palette_2)+1)[1:]*1e-3
            nb_color_8_10_black = int(len(palette_2)/(8.5-1.5)*(10-8.5))
            b4 = np.linspace(8.5, 10, nb_color_8_10_black+1)[1:]*1e-3
            b5 = np.linspace(1, 10, len(palette_3)-nb_color_8_10_black+1-1)[1:]*1e-2
            bounds = np.concatenate((b1, b2, b3, b4, b5))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style == 30: # CALIOP-like alternative (continuous)
            palette_1 = ['#000355', '#020657', '#040959', '#060b5b', '#080e5e', '#0a1160', '#0c1462', '#0d1664', '#0f1966', '#111c68', '#131e6a', '#15216c', '#16236e', '#182670', '#1a2872', '#1c2b73', '#1d2d75', '#1f3077', '#213279', '#23357b', '#24377d', '#263a7f', '#283c81', '#2a3f83', '#2c4185', '#2d4487', '#2f4689', '#31498b', '#334b8d', '#354e8f', '#365091', '#385393', '#3a5595', '#3c5897', '#3e5a99', '#3f5d9b', '#41609d', '#43629f', '#4565a1', '#4767a3', '#486aa5', '#4a6ca7', '#4c6fa9', '#4e72ab', '#5074ad', '#5277af', '#547ab1', '#557cb3', '#587fb5', '#5a82b6', '#5c84b7', '#5f87b9', '#618aba', '#638dbb', '#668fbc', '#6892be', '#6a95bf', '#6d97c0', '#6f9ac1', '#729dc3', '#74a0c4', '#76a3c5', '#79a5c6', '#7ba8c8', '#7eabc9', '#80aeca', '#83b0cb', '#85b3cc', '#88b6cd', '#8bb9ce', '#8dbccf', '#90bed0', '#93c1d1', '#95c4d2', '#98c7d3', '#9acad4', '#9dccd5', '#a0cfd6', '#a2d2d7', '#a5d5d8', '#b0d7da', '#bad9db', '#c4dcdd', '#cededf', '#d8e0e0']
            palette_2 = ['#e8e7c9', '#ecebb1', '#eff097', '#f0f57b', '#f1fa5c', '#f0ff33', '#f0ff33', '#f1fa30', '#f2f52d', '#f2f12b', '#f3ec28', '#f4e725', '#f5e222', '#f6dd1f', '#f7d81c', '#f7d31a', '#f8cd17', '#f9c814', '#fac312', '#fabe11', '#fab90f', '#fbb40e', '#fbae0c', '#fca90b', '#fca309', '#fd9e07', '#fd9806', '#fe9204', '#fe8c02', '#ff8500', '#ff7f00', '#ff7900', '#ff7200', '#ff6c00', '#ff6400', '#ff5d00', '#ff5500', '#ff4c00', '#ff4100', '#ff3600', '#ff2600', '#fb1f00', '#f61b00', '#f11701', '#ec1401', '#e71001', '#e10d01', '#dc0901', '#d70502', '#d20202', '#cc0002', '#c70002', '#c10003', '#bb0003', '#b60003', '#b00003', '#ab0004', '#a50004', '#a00004', '#9a0004', '#950004', '#900005', '#8a0005', '#850005', '#800004', '#7b0004', '#760003', '#710003', '#6b0002', '#660002', '#610002', '#5d0001', '#580001', '#530000', '#4e0000', '#490000', '#450000', '#400000', '#3b0000', '#370000', '#320000', '#2c0000', '#260000', '#1e0000', '#130000']
            palette_3 = ['#000000', '#040404', '#090909', '#0d0d0d', '#101010', '#131313', '#161616', '#181818', '#1a1a1a', '#1d1d1d', '#1f1f1f', '#212121', '#242424', '#262626', '#292929', '#2b2b2b', '#2e2e2e', '#303030', '#333333', '#353535', '#383838', '#3b3b3b', '#3d3d3d', '#404040', '#434343', '#454545', '#484848', '#4b4b4b', '#4d4d4d', '#505050', '#535353', '#565656', '#595959', '#5b5b5b', '#5e5e5e', '#616161', '#646464', '#676767', '#6a6a6a', '#6d6d6d', '#707070', '#727272', '#757575', '#787878', '#7b7b7b', '#7e7e7e', '#818181', '#848484', '#878787', '#8a8a8a', '#8e8e8e', '#919191', '#949494', '#979797', '#9a9a9a', '#9d9d9d', '#a0a0a0', '#a3a3a3', '#a6a6a6', '#a9a9a9', '#adadad', '#b0b0b0', '#b3b3b3', '#b6b6b6', '#b9b9b9', '#bdbdbd', '#c0c0c0', '#c3c3c3', '#c6c6c6', '#cacaca', '#cdcdcd', '#d0d0d0', '#d3d3d3', '#d7d7d7', '#dadada', '#dddddd', '#e1e1e1', '#e4e4e4', '#e7e7e7', '#ebebeb', '#eeeeee', '#f1f1f1', '#f5f5f5', '#f8f8f8', '#fcfcfc', '#ffffff']
            palette = palette_1 + palette_2 + palette_3
            print("len(palette_1):", len(palette_1))
            print("len(palette_2):", len(palette_2))
            print("len(palette_3):", len(palette_3))
            print("len(palette):", len(palette))
            my_cmap = mpl.colors.ListedColormap(palette)
            nb_color_minus5_minus4 = int(len(palette_1)*2/5)
            b1 = np.linspace(1, 10, nb_color_minus5_minus4+1-1)*1e-5
            nb_color_1_1_5_blue = int(len(palette_2)/(8.5-1.5)*(1.5-1.0))
            b2 = np.linspace(1, 10, len(palette_1)-nb_color_minus5_minus4-nb_color_1_1_5_blue+1)[1:]*1e-4
            b3 = np.linspace(1.0, 1.5, nb_color_1_1_5_blue+1)[1:]*1e-3
            b4 = np.linspace(1.5, 8.5, len(palette_2)+1)[1:]*1e-3
            nb_color_8_10_black = int(len(palette_2)/(8.5-1.5)*(10-8.5))
            b5 = np.linspace(8.5, 10, nb_color_8_10_black+1)[1:]*1e-3
            b6 = np.linspace(1, 10, len(palette_3)-nb_color_8_10_black+1-1)[1:]*1e-2
            bounds = np.concatenate((b1, b2, b3, b4, b5, b6))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
            # Extract discrete colormap
            bounds_discrete = np.array([0.00001, 0.00005, 
                                        0.0001, 0.0005, 
                                        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                                        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                                        0.1])
            mid_points = (bounds_discrete[:-1] + bounds_discrete[1:])/2
            palette_discrete = ['#000355',]
            for mid_point in mid_points:
                idx_mid = np.abs(bounds - mid_point).argmin()
                palette_discrete.append(palette[idx_mid])
            palette_discrete.append('#ffffff')
            print(palette_discrete)
        elif colormap_style == 31: # CALIOP-like alternative (continuous)
            palette_1 = ['#000000', '#00011e', '#00022e', '#00023a', '#000244', '#00034c', '#000354', '#020657', '#040959', '#060c5c', '#080f5e', '#0a1260', '#0c1563', '#0e1865', '#101b67', '#121e69', '#14206b', '#16236e', '#182670', '#1a2972', '#1c2c74', '#1e2e76', '#203178', '#22347a', '#24377c', '#25397e', '#273c81', '#293f83', '#2b4185', '#2d4487', '#2f4789', '#314a8b', '#334d8e', '#354f90', '#375292', '#395594', '#3b5896', '#3d5a99', '#3f5d9b', '#41609d', '#43639f', '#4566a2', '#4769a4', '#496ba6', '#4b6ea8', '#4e71aa', '#5074ac', '#5377ad', '#557aaf', '#577db1', '#5a80b2', '#5c83b4', '#5f86b5', '#6189b7', '#648cb9', '#668fba', '#6992bc', '#6b95be', '#6e98bf', '#709bc1', '#739ec2', '#75a1c4', '#78a4c6', '#7aa7c7', '#7eaac9', '#82acca', '#87afcb', '#8bb2cc', '#90b5cd', '#94b7cf', '#99bad0', '#9dbdd1', '#a2bfd2', '#a6c2d3', '#abc5d4', '#afc8d6', '#b4cad7', '#b8cdd8', '#bdd0d9', '#c1d2da', '#c6d5db', '#cad8dd', '#cfdbde', '#d4dddf', '#d8e0e0']
            palette_2 = ['#e8e7c9', '#ecebb1', '#eff097', '#f0f57b', '#f1fa5c', '#f0ff33', '#f1fc31', '#f1f830', '#f2f52e', '#f2f12c', '#f3ee2a', '#f3ea29', '#f4e727', '#f4e325', '#f5df23', '#f5dc22', '#f6d820', '#f6d51e', '#f7d11c', '#f7cd1a', '#f8ca19', '#f8c617', '#f9c215', '#f9be13', '#faba11', '#fbb70f', '#fbb30e', '#fbaf0e', '#fbab0d', '#fba80d', '#faa40d', '#faa00c', '#fa9c0c', '#fa980b', '#fa940b', '#fa910b', '#fa8c0a', '#fa880a', '#f9840a', '#f98009', '#f97c09', '#f97708', '#f97308', '#f96e08', '#f96907', '#f86407', '#f85f06', '#f85a06', '#f85405', '#f84e05', '#f74804', '#f74103', '#f73a03', '#f73102', '#f62701', '#f61b00', '#f21a00', '#ee1900', '#e91801', '#e51701', '#e11601', '#dd1501', '#d91401', '#d51301', '#d11202', '#cd1102', '#c91002', '#c50f02', '#c10e02', '#be0e02', '#ba0d03', '#b60c03', '#b20b03', '#ae0a03', '#aa0903', '#a60804', '#a20704', '#9f0604', '#9b0504', '#970404', '#930304', '#900305', '#8c0205', '#890105', '#850005', '#860e13', '#861d21', '#872b2f', '#873a3d', '#88484b']
            palette_3 = ['#885759', '#896566', '#897474', '#8a8282', '#8b8b8b', '#8c8c8c', '#8e8e8e', '#8f8f8f', '#919191', '#929292', '#949494', '#959595', '#979797', '#989898', '#999999', '#9b9b9b', '#9c9c9c', '#9e9e9e', '#9f9f9f', '#a1a1a1', '#a2a2a2', '#a4a4a4', '#a5a5a5', '#a7a7a7', '#a8a8a8', '#aaaaaa', '#ababab', '#adadad', '#aeaeae', '#b0b0b0', '#b1b1b1', '#b3b3b3', '#b4b4b4', '#b6b6b6', '#b7b7b7', '#b9b9b9', '#bababa', '#bcbcbc', '#bdbdbd', '#bfbfbf', '#c0c0c0', '#c2c2c2', '#c3c3c3', '#c5c5c5', '#c6c6c6', '#c8c8c8', '#cacaca', '#cbcbcb', '#cdcdcd', '#cecece', '#d0d0d0', '#d1d1d1', '#d3d3d3', '#d4d4d4', '#d6d6d6', '#d7d7d7', '#d9d9d9', '#dbdbdb', '#dcdcdc', '#dedede', '#dfdfdf', '#e1e1e1', '#e2e2e2', '#e4e4e4', '#e6e6e6', '#e7e7e7', '#e9e9e9', '#eaeaea', '#ececec', '#ededed', '#efefef', '#f1f1f1', '#f2f2f2', '#f4f4f4', '#f5f5f5', '#f7f7f7', '#f9f9f9', '#fafafa', '#fcfcfc', '#fdfdfd', '#ffffff']
            palette = palette_1 + palette_2 + palette_3
            print("len(palette_1):", len(palette_1))
            print("len(palette_2):", len(palette_2))
            print("len(palette_3):", len(palette_3))
            print("len(palette):", len(palette))
            my_cmap = mpl.colors.ListedColormap(palette)
            lim_palettes_1_2 = 1.5
            lim_palettes_2_3 = 6.5
            nb_color_1_1_5_blue = int(len(palette_2)/(lim_palettes_2_3-lim_palettes_1_2)*(lim_palettes_1_2-1.0))
            nb_color_minus5_minus4 = int((len(palette_1)-nb_color_1_1_5_blue)/2)
            b1 = np.linspace(1, 10, nb_color_minus5_minus4+1-1)*1e-5
            b2 = np.linspace(1, 10, len(palette_1)-nb_color_minus5_minus4-nb_color_1_1_5_blue+1)[1:]*1e-4
            b3 = np.linspace(1.0, lim_palettes_1_2, nb_color_1_1_5_blue+1)[1:]*1e-3
            b4 = np.linspace(lim_palettes_1_2, lim_palettes_2_3, len(palette_2)+1)[1:]*1e-3
            nb_color_8_10_black = int(len(palette_2)/(lim_palettes_2_3-lim_palettes_1_2)*(10-lim_palettes_2_3)/2)
            print(nb_color_8_10_black)
            b5 = np.linspace(lim_palettes_2_3, 10, nb_color_8_10_black+1)[1:]*1e-3
            b6 = np.linspace(1, 10, len(palette_3)-nb_color_8_10_black+1-1)[1:]*1e-2
            bounds = np.concatenate((b1, b2, b3, b4, b5, b6))
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
            # Extract discrete colormap
            bounds_discrete = np.array([0.00001, 0.00005, 
                                        0.0001, 0.0005, 
                                        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                                        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                                        0.1])
            mid_points = (bounds_discrete[:-1] + bounds_discrete[1:])/2
            palette_discrete = ['#000000',]
            for mid_point in mid_points:
                idx_mid = np.abs(bounds - mid_point).argmin()
                palette_discrete.append(palette[idx_mid])
            palette_discrete.append('#ffffff')
            print(palette_discrete)
        elif colormap_style == 32: # CALIOP-like alternative
            palette = ['#000000', 
                       '#040959', '#293f83', '#4e71aa', '#7eaac9', 
                       '#c1d2da', '#f0f57b', '#f6d51e', '#fa910b', '#f73102', '#b60c03', '#88484b', '#919191', '#9e9e9e', '#ababab', 
                       '#b6b6b6', '#bdbdbd', '#c6c6c6', '#cecece', '#d6d6d6', '#dfdfdf', '#e7e7e7', '#f1f1f1', '#f9f9f9', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 33: # CALIOP-like alternative
            palette = ["#000355", 
                       "#0D1664", "#2C4185", "#4E72AB", "#80AECA",  
                       '#BAD9DB', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', '#000000', '#646363', 
                       '#B2B2B2', '#C6C6C6', '#DFDFDF', '#E9E9E9', '#F0F0F0', '#F2F2F2', '#F5F5F5', '#F9F9F9', '#FDFDFD', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            # colors = my_cmap(np.arange(len(palette)))
            # my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            for i in range(my_cmap.N):
                rgba = my_cmap(i)
                # rgb2hex accepts rgb or rgba"
                print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 34: # CALIOP-like alternative
            palette = ["#000355", 
                       "#0D1664", "#2C4185", "#4E72AB", "#80AECA",  
                       '#BAD9DB', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#860806', '#780d08', '#691009', 
                       '#5a1109', '#4a1009', '#3b0e08', '#2a0b06', '#1a0604', '#090201', '#000000', '#ffffff', '#ffffff', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            # colors = my_cmap(np.arange(len(palette)))
            # my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            for i in range(my_cmap.N):
                rgba = my_cmap(i)
                # rgb2hex accepts rgb or rgba"
                print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 35: # CALIOP-like alternative
            palette = ["#000355", 
                       "#0D1664", "#2C4185", "#4E72AB", "#80AECA",  
                       '#BAD9DB', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#860806', '#780d08', '#691009', 
                       '#5a1109', '#4a1009', '#3b0e08', '#2a0b06', '#1a0604', '#090201', '#000000', '#ffffff', '#ffffff', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            # colors = my_cmap(np.arange(len(palette)))
            # my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            for i in range(my_cmap.N):
                rgba = my_cmap(i)
                # rgb2hex accepts rgb or rgba"
                print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 36: # CALIOP-like alternative
            palette = ["#000355", 
                        "#0D1664", "#2C4185", "#4E72AB", "#80AECA",  
                        '#ffffff', '#fff936', '#fec42d', '#fe842a', '#f70b0b', '#950004', '#641009', '#300c07',
                        '#000000' ]
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.008,
                                0.01])
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            for i in range(my_cmap.N):
                rgba = my_cmap(i)
                # rgb2hex accepts rgb or rgba"
                print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 37: # CALIOP-like alternative
            palette = ["#27268d", 
                        "#5560a8", "#838dc2", "#afb5d8", "#d8daec",  
                        '#ffffff', '#fff936', '#fec42d', '#fe842a', '#f70b0b', '#950004', '#641009', '#300c07',
                        '#000000' ]
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.008,
                                0.01])
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            for i in range(my_cmap.N):
                rgba = my_cmap(i)
                # rgb2hex accepts rgb or rgba"
                print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 38: # Rob test
            palette = ["#1D2333", 
                       "#243A60", "#24599E", "#0B79E0", "#649BFA", "#A8BDF7", 
                       "#FFFF00", "#FFCF00", "#FE9D00", "#FB6000", "#E50F00", "#AE0500", "#790200", "#480100", 
                       "#000000", "#303030", "#6F6F6F", "#B4B4B4", 
                       "#FFFFFF"]
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                               0.01, 0.02, 0.04, 0.06,
                               0.08])
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            # for i in range(my_cmap.N):
            #     rgba = my_cmap(i)
            #     # rgb2hex accepts rgb or rgba"
            #     print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 39:
            palette = ["#1D2333", 
                       "#243A60", "#24599E", "#0B79E0", "#649BFA", "#A8BDF7", 
                       "#FFFF00", "#FFCF00", "#FE9D00", "#FB6000", "#E50F00", "#AE0500", "#790200", "#480100", 
                       "#000000", "#303030", "#6F6F6F", "#B4B4B4", 
                       "#FFFFFF"]
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                               0.01, 0.02, 0.03, 0.04,
                               0.05])
            my_norm = BoundaryNorm(boundaries=bounds, ncolors=len(palette), extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            # Print hex colors
            # for i in range(my_cmap.N):
            #     rgba = my_cmap(i)
            #     # rgb2hex accepts rgb or rgba"
            #     print(f"'{mpl.colors.rgb2hex(rgba)}', ", end='')
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 40: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#80aeca', 
                       '#bad9db', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', '#130000', '#332929', 
                       '#574e4e', '#7e7777', '#a7a2a2', '#d2d0d0', 
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 
                               0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 41: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#80aeca', 
                       '#bad9db', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', '#130000', '#473D3D', 
                       '#7A7272', '#A7A2A2', '#CDCACA', '#EBEAEA', 
                       '#FFFFFF']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 
                               0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 42: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#80aeca', 
                       '#bad9db', '#eff097', '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', '#130000', '#453b3b', 
                       '#7e7777', '#bdb9b9', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 43: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#80aeca', 
                       '#bad9db', '#eff097', '#fbb20f', '#ff6501', '#dd1202', '#980302', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 44: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#7689a9', '#97a2a6', '#b6bba2', '#d3d59d', '#eff097', 
                       '#f4e725', '#fca90b', '#ff5d00', '#d70502', '#950004', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 
                               0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 45: # CALIOP-like alternative
            palette = ['#000355', 
                       '#0d1664', '#2c4185', '#4e72ab', '#7689a9', '#97a2a6', '#b6bba2',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 46: # CALIOP-like alternative
            palette = ['#000355', '#192771', '#31498b', '#4a6ca6', '#6d90b9', '#93b4ca', '#bad9db',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 47: # CALIOP-like alternative
            palette = ['#000355', '#192771', '#31498b', '#4e72ab', '#7390b8', '#98afbf', '#c0cfbc',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 48: # CALIOP-like alternative
            palette = ['#000355', '#192771', '#31498b', '#4a6ca6', '#6d90b9', '#93b4ca', '#c6cfad',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 49: # CALIOP-like alternative
            palette = ['#000355', '#15226d', '#293e83', '#3e5b99', '#5678ac', '#7396bc', '#93b4ca',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 50: # CALIOP-like alternative
            palette = ['#192771', '#31498b', '#4a6ca6', '#6d90b9', '#93b4ca', '#c6cfad',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 51: # CALIOP-like alternative
            palette = ['#000355', '#232870', '#414b87', '#60709c', '#8795a8', '#b6bba2',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 52: # CALIOP-like alternative
            palette = ['#000355', '#101360', '#1e226b', '#2b3176', '#373f80', '#444e89', '#515d92', '#5e6d9b', '#6b7ca3', '#7a8ca9', '#8f9ba7', '#a3aba5', '#b6bba2',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad   
        elif colormap_style == 53: # CALIOP-like alternative
            palette = ['#000355', '#121663', '#22276f', '#31387b', '#404986', '#505b8e', '#616d96', '#727f9e', '#8391a6', '#98a4a5', '#adb7a2', '#c3c99e', '#d9dd9b',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad   
        elif colormap_style == 54: # CALIOP-like alternative
            palette = ['#000355', '#2b3176', '#3e4885', '#535e90', '#69769a', '#808da5', '#9aa6a5', '#b6bea0',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001', 
                       '#130000', '#2e2424', '#4d4343', '#6d6565', '#908989', '#b3afaf', '#d9d6d6', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad   
        elif colormap_style == 55: # CALIOP-like alternative
            palette = ['#000000', '#060B54', '#262c72', '#3b4482', '#525d8f', '#69769a', '#808da5', '#9aa6a5', '#b6bea0',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001',
                       '#2a0000', '#352929', '#4e4a4a', '#6d6b6b', '#8f8e8e', '#b2b2b2', '#d8d8d8', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 56: # CALIOP-like alternative
            palette = ['#000000', '#181c63', '#3b4482', '#525d8f', '#69769a', '#808da5', '#9aa6a5', '#b6bea0',  
                       '#eff097', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001',
                       '#2a0000', '#352929', '#4e4a4a', '#6d6b6b', '#8f8e8e', '#b2b2b2', '#d8d8d8', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001,
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad 
        elif colormap_style == 57: # CALIOP-like alternative
            palette = ['#000000', '#060b54', '#22296c', '#3a4683', '#2f6c99', '#2490ad', '#19b4c2', '#0ed9d7', '#02ffed',  
                       '#B0DE9E', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001',
                       '#2a0000', '#352929', '#4e4a4a', '#6d6b6b', '#8f8e8e', '#b2b2b2', '#d8d8d8', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 58: # CALIOP-like alternative
            palette = ['#000000', '#30123b', '#3f3b97', '#4661d6', '#4687fb', '#35abf8', '#1ccdd8', '#1ce6b4', '#43f787', 
                       '#79fe59', '#f4e725', '#fd9708', '#ee3801', '#a70103', '#580001',
                       '#2a0000', '#352929', '#4e4a4a', '#6d6b6b', '#8f8e8e', '#b2b2b2', '#d8d8d8', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 59: # CALIOP-like alternative
            palette = ['#000000', '#000064', '#2b088b', '#5a1d9d', '#82389e', '#a25697', '#bd768a', '#d2977a', '#e4b965', 
                       '#f3dc49', '#ffff00', '#ffad00', '#e16102', '#a52107', '#580000',
                       '#2a0000', '#352929', '#4e4a4a', '#6d6b6b', '#8f8e8e', '#b2b2b2', '#d8d8d8', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 60: # CALIOP-like alternative
            palette = ['#000000', '#060b54', '#1e1f64', '#323372', '#444881', '#555d8f', '#66739d', '#778aab', '#92adc0', '#bcd2c1', '#f0f6b9', 
                       '#f3e641', '#ff8a00', '#e70101', '#870303', '#2e0606',
                       '#462f2f', '#5e5757', '#7f7f7f', '#a8a8a8', '#d3d3d3', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 
                               0.001, 0.0012, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                               0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 61: # CALIOP-like alternative
            palette = ['#000000', '#060B54', '#34137e', '#571e97', '#7d2fb0', '#a54ac9', '#ca6cdf', '#ea97f1', '#fdc8fc', '#ffe0d1', '#fdea96',
                       '#f3e641', '#ff8a00', '#e70101', '#870303', '#2e0606',
                       '#462f2f', '#5e5757', '#7f7f7f', '#a8a8a8', '#d3d3d3', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 
                               0.001, 0.0012, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                               0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 62: # CALIOP-like alternative
            palette = ['#000000', '#060B54', '#34137e', '#453c91', '#595f9f', '#7382a6', '#95a4a2', '#c1c588',
                       '#f3e641', '#ff8a00', '#e70101', '#870303', '#2e0606',
                       '#462f2f', '#5e5757', '#7f7f7f', '#a8a8a8', '#d3d3d3', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0004, 0.0007, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                               0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad  
        elif colormap_style == 63: # CALIOP-like alternative
            palette = ['#000000', '#380061', '#2a2b9a', '#3c48bf', '#6966cf', '#9186d1', '#b4aac0', '#ced297',
                       '#f3e641', '#ff8a00', '#e70101', '#870303', '#2e0606',
                       '#462f2f', '#5e5757', '#7f7f7f', '#a8a8a8', '#d3d3d3', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.000001, 0.00001, 
                               0.0001, 0.0004, 0.0007, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                               0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 64: # CALIOP-like alternative
            palette = ['#000000', '#0a2472', '#2046c1', '#6375d6', '#a4a5d2', '#ebdeb5',
                        '#ebde39', '#f88b0f', '#e70101', '#8c120d', '#42191c',
                        '#5f4545', '#7d7272', '#a1a0a0', '#cfcfcf', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                                0.01, 0.015, 0.025, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 65: # CALIOP-like alternative
            palette = ['#000000', '#0a2472', '#2a40a3', '#5262bf', '#8987b1', '#b8ae8d',
                        '#fff23a', '#fda321', '#ec4809', '#a11a14', '#42191c',
                        '#5f4545', '#7d7272', '#a1a0a0', '#cfcfcf', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                                0.01, 0.015, 0.025, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 66: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#40417b', '#6a6c94', '#a1a0a9', '#ebdeb5',
                        '#ebde39', '#f88b0f', '#e70101', '#8c120d', '#42191c',
                        '#5f4545', '#7d7272', '#a1a0a0', '#cfcfcf', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                                0.01, 0.015, 0.025, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 67: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#40417b', '#61648e', '#888998', '#b8ae8d',
                        '#fff23a', '#fda321', '#ec4809', '#a11a14', '#42191c',
                        '#5f4545', '#7d7272', '#a1a0a0', '#cfcfcf', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                                0.01, 0.015, 0.025, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 68: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#4f4c7e', '#7f7c8f', '#b8ae8d',
                        '#fff23a', '#fda321', '#ec4809', '#a11a14', '#42191c',
                        '#573f41', '#6c6565', '#8a8a8a', '#b0b0b0', '#d7d7d7', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005, 
                                0.001, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.008, 
                                0.01, 0.015, 0.025, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 69: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2d507a', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#ffff81', '#ffa23a', '#e53803', '#85120a', '#2b0000', 
                        '#483333', '#656565', '#959595', '#c9c9c9', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.0055, 0.0075, 
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 70: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#283c71', '#2f597e', '#8f99a2', '#d8d2c0',
                        '#ffff81', '#ffa23a', '#e53803', '#85120a', '#2b0000', 
                        '#483333', '#656565', '#959595', '#c9c9c9', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.0055, 0.0075, 
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 71: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2d507a', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fca722', '#ec4809', '#841b0d', '#2b0000',
                        '#412c2c', '#565656', '#7d7d7d', '#a6a6a6', '#d2d2d2', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 72: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#283c71', '#2f597e', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fca722', '#ec4809', '#841b0d', '#2b0000',
                        '#412c2c', '#565656', '#7d7d7d', '#a6a6a6', '#d2d2d2', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 73: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2d507a', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#412c2c', '#565656', '#7d7d7d', '#a6a6a6', '#d2d2d2', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0005,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 74: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2f3e73', '#465d82', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#412c2c', '#565656', '#7d7d7d', '#a6a6a6', '#d2d2d2', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0004, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 75: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2b376f', '#3a4e7a', '#4d6486', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#412c2c', '#565656', '#7d7d7d', '#a6a6a6', '#d2d2d2', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0003, 0.0005, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 76: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2f3e73', '#465d82', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#5a5a5a', '#787878', '#989898', '#b9b9b9', '#dcdcdc', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0004, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.013, 0.018, 0.025, 0.035, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 77: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2f3e73', '#465d82', '#647a91', '#9ea5a9', '#d8d2c0',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0004, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 78: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2f3e73', '#465d82', '#647a91', '#a7aaa0', '#e9daae',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0004, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 79: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#2c3770', '#3c4f7b', '#4f6687', '#687d92', '#879399', '#a7aaa0', '#e9daae',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 80: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#344376', '#4f6687', '#788895', '#a7aaa0', '#e9daae',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0004, 0.0007,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 81: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#344376', '#4f6687', '#788895', '#a7aaa0', '#e9daae',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0003, 0.0006,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 82: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#29346e', '#374878', '#465c82', '#5a708b', '#718394', '#a7aaa0', '#e9daae',
                        '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 83: # CALIOP-like alternative
            palette = ['#000000', '#1e1f64', '#344376', '#4f6687', '#7588a8', '#a1abbd', '#d5d1b1', 
                       '#fbff41', '#fea100', '#ec2401', '#84110c', '#2b0000',
                        '#443b3b', '#606060', '#858585', '#acacac', '#d5d5d5', '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            bounds = np.array([0.00001, 
                                0.0001, 0.0003, 0.0006,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                0.01, 0.015, 0.02, 0.03, 0.05])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 100: # PSC
            my_cmap = cmocean.cm.thermal
            my_cmap.colorbar_extend = 'both'
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=LogNorm(), rasterized=True)
            if polar=='per':
                plt.clim(1e-6, 1e-3)
            elif wl==1064:
                plt.clim(1e-6, 1e-3)
            else:
                plt.clim(1e-6, 1e-3)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style in [155,]:
            palette = ['#000000', '#01020e', '#02041c', '#03062a', '#040738', '#050946',
                       '#060b54', '#0a0e57', '#0e1159', '#11145c', '#14175f', '#171a61', '#1a1d64', '#1c2067', '#1f236a', '#21266c', '#24296f', 
                       '#262c72', '#282e73', '#2a3075', '#2c3276', '#2e3578', '#303779', '#32397b', '#343b7c', '#363d7e', '#37407f', '#394281', 
                       '#3b4482', '#3d4683', '#3f4884', '#424b86', '#444d87', '#464f88', '#485189', '#4a548a', '#4c568b', '#4e588d', '#505b8e', 
                       '#525d8f', '#545f90', '#566191', '#586492', '#5b6693', '#5d6894', '#5f6b95', '#616d96', '#636f97', '#657198', '#677499', 
                       '#69769a', '#6b789b', '#6d7a9c', '#6f7c9d', '#717e9e', '#74809f', '#7682a0', '#7885a1', '#7a87a2', '#7c89a3', '#7e8ba4', 
                       '#808da5', '#828fa5', '#8591a5', '#8794a5', '#8a96a5', '#8c98a5', '#8e9ba5', '#919da5', '#939fa5', '#95a1a5', '#98a4a5', 
                       '#9aa6a5', '#9da8a5', '#9faaa4', '#a2aca4', '#a4afa3', '#a7b1a3', '#a9b3a2', '#acb5a2', '#afb7a1', '#b1baa1', '#b4bca0', 
                       '#b6bea0', '#bbc29f', '#c1c79f', '#c6cb9e', '#cbd09d', '#d0d49c', '#d5d99c', '#dbde9b', '#e0e29a', '#e5e799', '#eaeb98',
                       '#eff097', '#f0ef8f', '#f1ee86', '#f2ed7e', '#f2ed75', '#f3ec6c', '#f3eb62', '#f4ea58', '#f4e94e', '#f4e942', '#f4e835', 
                       '#f4e725', '#f6e022', '#f7d920', '#f8d21d', '#f9cb1b', '#fac418', '#fbbc16', '#fcb513', '#fcae10', '#fda60e', '#fd9f0b', 
                       '#fd9708', '#fc9007', '#fb8805', '#fa8104', '#f97903', '#f77102', '#f66902', '#f46101', '#f35801', '#f14e01', '#f04401', 
                       '#ee3801', '#e73401', '#e13002', '#da2c02', '#d42802', '#cd2302', '#c71f03', '#c01a03', '#ba1503', '#b40f03', '#ad0803', 
                       '#a70103', '#9f0104', '#980104', '#910104', '#890104', '#820004', '#7b0004', '#740004', '#6d0003', '#660003', '#5f0002', 
                       '#580001', '#540101', '#4f0101', '#4b0201', '#460201', '#420301', '#3e0301', '#390301', '#350300', '#310300', '#2e0200',
                       '#2a0000', '#2b0506', '#2b0a0b', '#2d0e10', '#2e1213', '#2f1517', '#30181a', '#311c1d', '#321f20', '#332223', '#342626', 
                       '#352929', '#372c2c', '#3a2f2f', '#3c3232', '#3e3535', '#403838', '#433b3b', '#453e3e', '#474141', '#494444', '#4c4747',
                       '#4e4a4a', '#514d4d', '#545050', '#565353', '#595656', '#5c5959', '#5f5c5c', '#625f5f', '#646262', '#676565', '#6a6868', 
                       '#6d6b6b', '#706e6e', '#737171', '#767474', '#797878', '#7c7b7b', '#7f7e7e', '#828181', '#868484', '#898888', '#8c8b8b', 
                       '#8f8e8e', '#929191', '#959494', '#989898', '#9c9b9b', '#9f9e9e', '#a2a1a1', '#a5a5a5', '#a8a8a8', '#acabab', '#afafaf', 
                       '#b2b2b2', '#b5b5b5', '#b9b9b9', '#bcbcbc', '#c0c0c0', '#c3c3c3', '#c7c7c7', '#cacaca', '#cecece', '#d1d1d1', '#d4d4d4',
                       '#d8d8d8', '#dfdfdf', '#e5e5e5', '#ececec', '#f2f2f2', '#f9f9f9',
                       '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            discrete_bounds = np.array([0.000001, 0.00001, 
                                        0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                                        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                                        0.01, 0.02, 0.03, 0.04, 0.05])
            nb_colors_by_discrete_bin = 11
            bounds = np.array(())
            for i in range(discrete_bounds.size-1): # color of discrete bounds at the mid value of the range bin
                bounds = np.append(bounds, np.linspace(discrete_bounds[i], discrete_bounds[i+1], nb_colors_by_discrete_bin+1)[:-1])
            bounds = np.append(bounds, discrete_bounds[-1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style in [181,]:
            palette = [ '#000000', '#090612', '#110b1d', '#141028', '#171433', '#19163f', '#1b194b', '#1d1c57',
                        '#1e1f64', '#202165', '#222466', '#232668', '#252969', '#262b6a', '#282d6b', '#29306c', '#2b326e', '#2c356f', '#2e3770', '#2f3971', '#303c72', '#323e74', '#334175',
                        '#344376', '#364577', '#384878', '#3a4a79', '#3c4c7b', '#3d4e7c', '#3f517d', '#41537e', '#43557f', '#455880', '#465a81', '#485c82', '#4a5f84', '#4c6185', '#4d6486',
                        '#4f6687', '#526888', '#556a89', '#586d8a', '#5a6f8b', '#5d718c', '#60738d', '#63768e', '#65788f', '#687a8f', '#6b7c90', '#6d7f91', '#708192', '#738393', '#758694', 
                        '#788895', '#7b8a96', '#7e8c96', '#828f97', '#859198', '#889399', '#8b9599', '#8e989a', '#919a9b', '#949c9c', '#989e9c', '#9ba19d', '#9ea39e', '#a1a59f', '#a4a89f',
                        '#a7aaa0', '#abada1', '#b0b0a2', '#b4b3a3', '#b9b7a4', '#bdbaa5', '#c1bda6', '#c6c0a7', '#cac3a8', '#cfc7a9', '#d3caa9', '#d7cdaa', '#dcd0ab', '#e0d3ac', '#e5d7ad',
                        '#e9daae', '#ebdca8', '#ecdfa2', '#eee19c', '#f0e496', '#f1e68f', '#f2e989', '#f4eb82', '#f5ee7c', '#f6f075', '#f7f36d', '#f8f566', '#f9f75e', '#fafa55', '#fafc4c',
                        '#fbff41', '#fcf93e', '#fdf33a', '#fded36', '#fee633', '#fee02f', '#ffda2c', '#ffd428', '#ffce24', '#ffc720', '#ffc11c', '#ffbb17', '#ffb412', '#ffae0d', '#fea806',
                        '#fea100', '#fd9b00', '#fc9400', '#fb8d00', '#fb8700', '#f98000', '#f87900', '#f77100', '#f66a00', '#f56200', '#f35a00', '#f25200', '#f14900', '#ef3e00', '#ee3300', 
                        '#ec2401', '#e52302', '#de2104', '#d62005', '#cf1f06', '#c81d07', '#c11c08', '#ba1b09', '#b31a09', '#ac180a', '#a5170b', '#9f160b', '#98150b', '#91130c', '#8b120c',
                        '#84110c', '#7d100c', '#770f0c', '#710e0b', '#6a0d0b', '#640c0a', '#5e0b0a', '#580909', '#520808', '#4c0807', '#460706', '#400605', '#3a0504', '#350402', '#300301',
                        '#2b0000', '#2c0506', '#2e0a0c', '#300e11', '#311215', '#331618', '#351a1b', '#371e1f', '#392122', '#3b2526', '#3c2829', '#3e2c2d', '#403030', '#413434', '#433737',
                        '#443b3b', '#463d3d', '#484040', '#4a4242', '#4b4545', '#4d4747', '#4f4949', '#514c4c', '#534e4e', '#555151', '#575353', '#585656', '#5a5858', '#5c5b5b', '#5e5d5d',
                        '#606060', '#626262', '#656565', '#676767', '#6a6a6a', '#6c6c6c', '#6f6f6f', '#717171', '#737373', '#767676', '#787878', '#7b7b7b', '#7d7d7d', '#808080', '#828282',
                        '#858585', '#888888', '#8a8a8a', '#8d8d8d', '#8f8f8f', '#929292', '#949494', '#979797', '#9a9a9a', '#9c9c9c', '#9f9f9f', '#a1a1a1', '#a4a4a4', '#a7a7a7', '#a9a9a9',
                        '#acacac', '#afafaf', '#b1b1b1', '#b4b4b4', '#b7b7b7', '#b9b9b9', '#bcbcbc', '#bfbfbf', '#c2c2c2', '#c4c4c4', '#c7c7c7', '#cacaca', '#cdcdcd', '#cfcfcf', '#d2d2d2',
                        '#d5d5d5', '#dadada', '#dfdfdf', '#e5e5e5', '#eaeaea', '#efefef', '#f4f4f4', '#fafafa',
                        '#ffffff']
            my_cmap = mpl.colors.ListedColormap(palette)
            discrete_bounds = np.array([0.00001, 
                                        0.0001, 0.0003, 0.0006,
                                        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
                                        0.01, 0.015, 0.02, 0.03, 0.05])
            nb_colors_between_mid_discrete_bin = 15 # 16*15+2 = 242
            bounds = np.array(())
            for i in range(discrete_bounds.size-1): # color of discrete bounds at the mid value of the range bin
                bounds = np.append(bounds, np.linspace(discrete_bounds[i], discrete_bounds[i+1], nb_colors_between_mid_discrete_bin+1)[:-1])
            bounds = np.append(bounds, discrete_bounds[-1])
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style in [1000,]:
            my_cmap = cmlidar.cm.backscatter_18
            my_norm = cmlidar.cm.backscatter_18_norm
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap, norm=my_norm,
                                rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        elif colormap_style in [1001,]:
            my_cmap = cmlidar.cm.backscatter_242
            my_norm = cmlidar.cm.backscatter_242_norm
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap, norm=my_norm,
                                rasterized=True)
            cbar_edges = False
            clabelpad = self.clabelpad
        else: # LogNorm colormap
            # my_cmap = cm.inferno
            # my_cmap = takecmap('extviridis_black_white')
            # my_cmap = cmocean.cm.thermal
            my_cmap = takecmap('extthermal')
            # my_cmap = cmlidar.cm.backscatter
            # my_cmap = takecmap('extviridis')
            my_cmap.colorbar_extend = 'both'
            pc = plt.pcolormesh(self.pindexbins, self.altbins, atb2.T, cmap=my_cmap,
                                norm=LogNorm(), rasterized=True)
            if polar=='per':
                plt.clim(1e-7, 1e-4)
            elif wl==1064:
                plt.clim(1e-5, 2e-3)
            else:
                plt.clim(1e-5, 2e-3)
            cbar_edges = False
            clabelpad = self.clabelpad
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        deconv_txt = "(deconvoluted)" if APPLY_DECONVOLUTION else ""
        if wl == 532:
            if polar == 'par':
                title = r"Parallel 532 nm Attenuated Backscatter $\mathit{\beta'_{532,\parallel}}$" + f" ({VERSION_CAL_LID_L1}) {deconv_txt}"
            elif polar == 'per':
                title = r"Perpendicular 532 nm Attenuated Backscatter $\mathit{\beta'_{532,\bot}}$"+f" ({VERSION_CAL_LID_L1}) {deconv_txt}"
            else:
                title = r"Total 532 nm Attenuated Backscatter $\mathit{\beta'_{532}}$"+f" ({VERSION_CAL_LID_L1}) {deconv_txt}"
        elif wl == 1064:
            title = f"1064 nm Attenuated Backscatter"+r" $\mathit{\beta'_{1064}}$"+f" ({VERSION_CAL_LID_L1}) {deconv_txt}"
        plt.title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=cbar_edges, extend='both')
        cbar.set_label(label=r"$\beta'$ (km$^{-1}$ sr$^{-1}$)", labelpad=clabelpad)
        
        if colormap_style in [1, 2, 3, 4, 5]:
            # Colorbar ticks
            bounds_major_index = [0, 9, 24, -1]
            cbar_major = bounds[bounds_major_index]
            # cbar_minor = np.delete(bounds, bounds_major_index)
            cbar_minor = np.copy(bounds)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(cbar_minor))
            cbar.ax.yaxis.set_major_locator(FixedLocator(cbar_major))
            cbar.ax.tick_params(which='both', labelright=False)
            # Major labels
            #cbar_major_label = bounds[bounds_major_index]
            cbar_major_label = [r'$\mathbf{×10^{-4}}$', r'$\mathbf{×10^{-3}}$', r'$\mathbf{×10^{-2}}$', r'$\mathbf{×10^{-1}}$']
            for j, lab in enumerate(cbar_major_label):
                cbar.ax.text(3.5, cbar_major[j], lab, va='center', fontsize=self.ytick_labelsize)
                # Minor labels
                # cbar_minor_label = np.delete(bounds, bounds_major_index)
                cbar_minor_label = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0',
                                    '9.0', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0',
                                    '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5',
                                    '8.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0',
                                    '8.0', '9.0', '1.0']
            for j, lab in enumerate(cbar_minor_label):
                cbar.ax.text(2, cbar_minor[j], lab, va='center', fontsize=4)
        elif colormap_style == 5: # CALIOP-like colorblind colormap
            # Colorbar ticks
            bounds_major_index = [0, 9, 18, -1]
            cbar_major = bounds[bounds_major_index]
            cbar_minor = np.delete(bounds, bounds_major_index)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(cbar_minor))
            cbar.ax.yaxis.set_major_locator(FixedLocator(cbar_major))
            cbar.ax.tick_params(which='both', labelright=False)
            # Major labels
            #cbar_major_label = bounds[bounds_major_index]
            cbar_major_label = ['1.0×$10^{-4}$', '1.0×$10^{-3}$', '1.0×$10^{-2}$',
                                '1.0×$10^{-1}$']
            for j, lab in enumerate(cbar_major_label):
                cbar.ax.text(2, cbar_major[j], lab, va='center', fontsize=self.ytick_labelsize)
                # Minor labels
                # cbar_minor_label = np.delete(bounds, bounds_major_index)
                cbar_minor_label = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                    '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                    '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
            for j, lab in enumerate(cbar_minor_label):
                cbar.ax.text(1.5, cbar_minor[j], lab, va='center', fontsize=2)
        elif colormap_style == 7: # few color, easy to read
            # Colorbar ticks
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar.ax.tick_params(which='both', labelright=False)
            # Major labels
            cbar_major_label = ['1.0×$10^{-4}$', '3.0×$10^{-4}$', '1.0×$10^{-3}$', '3.0×$10^{-3}$',
                                '1.0×$10^{-2}$', '3.0×$10^{-2}$']
            for j, lab in enumerate(cbar_major_label):
                cbar.ax.text(2, bounds[j], lab, va='center', fontsize=self.ytick_labelsize7)
        elif colormap_style == 19:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            # cbar_major_label = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
            # c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            cbar_major_label = ['×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-4, 1e-3, 1e-2, 1e-1))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
            minor_locators = np.concatenate((np.arange(1,10)*1e-4,
                                             np.arange(1,10,0.5)*1e-3,
                                             np.arange(1,11)*1e-2))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
            cbar_minor_label = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', 
                                '1.0']
            for j, bound in enumerate(minor_locators):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style == 20:
            # cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            # cbar.ax.tick_params(which='both', labelright=False)
            # cbar_major_label = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
            # c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            cbar_major_label = ['×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-4, 2e-3, 1e-2, 1e-1))
            cbar.ax.yaxis.set_major_locator(FixedLocator(c_bar_major_values))
            cbar.ax.tick_params(which='both', labelright=False)
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
            minor_locators = np.concatenate((np.array((1, 3, 5, 7, 9))*1e-4,
                                             np.array((2, 3, 4, 5, 6, 8))*1e-3,
                                             np.array((1, 4, 7))*1e-2,
                                             np.array((1,))*1e-1))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
            cbar_minor_label = ['1.0', '3.0', '5.0', '7.0', '9.0',
                                '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '4.0', '7.0',
                                '1.0']
            for j, bound in enumerate(minor_locators):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style == 21:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            # cbar_major_label = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
            # c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
            minor_locators = np.concatenate((np.array((1, 4, 7))*1e-5,
                                             np.array((1, 4, 7))*1e-4,
                                             np.arange(1,10)*1e-3,
                                             np.array((1, 4, 7))*1e-2,
                                             np.array((1,))*1e-1))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
            cbar_minor_label = ['1.0', '4.0', '7.0',
                                '1.0', '4.0', '7.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '4.0', '7.0',
                                '1.0']
            for j, bound in enumerate(minor_locators):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [22, 26]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [28, 32, 33, 34, 35]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            # cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            # c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            # cbar_minor_label = ['1.0', '5.0',
            #                     '1.0', '5.0',
            #                     '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
            #                     '1.0', '5.0',
            #                     '1.0']
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [29, 30]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style == 31:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$', '×$10^{-1}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            bounds = np.array([0.00001, 0.00005, 
                               0.0001, 0.0005, 
                               0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.009,
                               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
                               0.1])
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [36, 37]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$',]
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.5', '3.5', '4.5', '5.5', '6.5', '8.0',
                                '1.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style == 38:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$',]
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1', '5',
                                '1', '5',
                                '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                '1', '2', '4', '6',
                                '8']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style == 39:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$',]
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1', '5',
                                '1', '5',
                                '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                '1', '2', '3', '4',
                                '5']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [40, 41, 43, 50, 51]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [42,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [44,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [45, 46, 47, 48, 49]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '5.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [52, 53]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0', '5.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [54,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [55, 57, 58, 59]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-6}$', '×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [56,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-6}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-6, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [60, 61]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-6}$', '×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.2', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [62, 63]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-6}$', '×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0',
                                '1.0', '4.0', '7.0', 
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [64, 65, 66, 67]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '5.0', 
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.5', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [68,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '5.0', 
                                '1.0', '1.5', '2.5', '3.5', '4.5', '5.5', '6.5', '8.0',
                                '1.0', '1.5', '2.5', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [69, 70]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.5', '7.5',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [71, 72, 73]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '5.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [74, 77, 78, 80]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '4.0', '7.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [75,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '3.0', '5.0', '7.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [76,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '4.0', '7.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.3', '1.8', '2.5', '3.5', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [79, 82]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [81, 83]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = [r'$\mathbf{×10^{-5}}$', r'$\mathbf{×10^{-4}}$', r'$\mathbf{×10^{-3}}$', r'$\mathbf{×10^{-2}}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                if PLOT_ASPECT_RATIO == "spec":
                    cbar.ax.text(4.5, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
                else:
                    cbar.ax.text(3.5, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '3.0', '6.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=6)
        elif colormap_style in [100,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$']
            c_bar_major_values = np.array((1e-6, 1e-5, 1e-4, 1e-3))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
            minor_locators = np.concatenate((np.arange(2,10)*1e-6,
                                             np.arange(2,10)*1e-5,
                                             np.arange(2,10)*1e-4,
                                             np.arange(2,10)*1e-3))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
        elif colormap_style in [155,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-6}$', '×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            minor_bounds = np.array([0.000001, 0.00001, 
                                0.0001, 0.0002, 0.0004, 0.0006, 0.0008,
                                0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
                                0.01, 0.02, 0.03, 0.04, 0.05])
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_bounds))
            cbar_minor_label = ['1.0',
                                '1.0',
                                '1.0', '2.0', '4.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0',
                                '1.0', '2.0', '3.0', '4.0', '5.0']
            for j, bound in enumerate(minor_bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [181,]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = ['×$10^{-5}$', '×$10^{-4}$', '×$10^{-3}$', '×$10^{-2}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(3.2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            minor_bounds = np.array([0.00001, 
                                     0.0001, 0.0003, 0.0006,
                                     0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                                     0.01, 0.015, 0.02, 0.03, 0.05])
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '3.0', '6.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(minor_bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=4)
        elif colormap_style in [1000, 1001]:
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            cbar_major_label = [r'$\mathbf{×10^{-5}}$', r'$\mathbf{×10^{-4}}$', r'$\mathbf{×10^{-3}}$', r'$\mathbf{×10^{-2}}$']
            c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
            for j, bound in enumerate(c_bar_major_values):
                if PLOT_ASPECT_RATIO == "spec":
                    cbar.ax.text(4.5, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
                else:
                    cbar.ax.text(3.5, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            bounds = cmlidar.cm.BACKSCATTER_DISCRETE_BOUNDS
            cbar.ax.yaxis.set_minor_locator(FixedLocator(bounds))
            cbar_minor_label = ['1.0',
                                '1.0', '3.0', '6.0',
                                '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                                '1.0', '1.5', '2.0', '3.0', '5.0']
            for j, bound in enumerate(bounds):
                cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=6)
        else: # LogNorm colormap
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            cbar.ax.tick_params(which='both', labelright=False)
            # cbar_major_label = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
            # c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
            cbar_major_label = ['$10^{-7}$', '$10^{-6}$', '$10^{-5}$', '$10^{-4}$']
            c_bar_major_values = np.array((1e-7, 1e-6, 1e-5, 1e-4))
            for j, bound in enumerate(c_bar_major_values):
                cbar.ax.text(2, bound, cbar_major_label[j], va='center', fontsize=self.ytick_labelsize)
            # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
            minor_locators = np.concatenate((np.arange(2,10)*1e-7,
                                             np.arange(2,10)*1e-6,
                                             np.arange(2,10)*1e-5,
                                             np.arange(2,10)*1e-4))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))

        # Set colorbar fontsizes
        cbar.ax.yaxis.label.set_size(self.axes_labelsize)

        # Save figure
        filename = f"AB{wl:d}{polar}_{grid}"
        self.save_fig(filename)

        # Save figure (for test_colorbar)
        # self.fig_folder = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/test_colorbar/"
        # filename = f"AB{wl:d}{polar}_{grid}_{colormap_style}"
        # filename = f"AB{wl:d}{polar}_{grid}_0Current"
        # self.save_fig(filename)
        
        # # Save data in pickle (for test_colorbar)
        # atb2.dump("/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/test_colorbar/"+self.head_filename+f"_AB{wl:d}{polar}"+'.pkl')

        # Close figure
        plt.close(fig)

  
    def plot_attenuated_color_ratio(self, acr, grid):
    
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            acr = remove_edges(acr, EDGES_REMOVAL)
        
        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0], facecolor='k')

        # colormap_style = 1 # 1: CALIOP-like
        #                    # 2: linear
        #                    # 3: linear extviridis
        #                    # 4: diverging
        #                    # else*: diverging with few bins
        if COLORMAP == "LEGACY":
            colormap_style = 1
        elif COLORMAP == "FRIENDLY":
            colormap_style = 1000
        else:
            colormap_style = 0

        if colormap_style == 1: # CALIOP-like colormap
            bounds = np.arange(0, 1.61, 0.1)
            colors = ['#000000',
                    '#22A6F9',
                    '#1ED036',
                    '#FFFF4F',
                    '#FDAA41',
                    '#FD2F36',
                    '#FD30F9',
                    '#FFD2FD',
                    '#A659F9',
                    '#7D19A2',
                    '#A6227C',
                    '#A8D0FB',
                    '#A8FDFD',
                    '#A8FDD3',
                    '#D2FFD4',
                    '#FFFFD6',
                    '#FFFFFF',
                    '#999999']
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins,self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 2: # Colorblind linear
            # my_cmap = takecmap('cubeh1_r', nb_colors=256, clight=1., cdark=0.25)
            my_cmap = copy.copy(cm.YlGnBu_r)
            # my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 1.61, 0.1)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors[0] = [0., 0., 0., 1.] # replace extreme by black
            colors[-1] = [0.2, 0.2, 0.2, 1.] # replace extreme by grey
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 3: # Colorblind linear extviridis
            my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 1.61, 0.1)
            nb_colors = len(bounds) - 1 - 2 # add 2 yellow-white colors manually
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [[0., 0., 0., 1.]],
                            colors,
                            [[255/255., 253/255., 236/255., 1.]],
                            [[255/255., 255/255., 255/255., 1.]],
                            [[0.4, 0.4, 0.4, 1.]] ]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            clabelpad = self.clabelpad
        elif colormap_style == 4: # Colorblind diverging
            my_cmap1 = copy.copy(cm.YlOrRd_r)
            nb_colors = 10
            colors1 = my_cmap1(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap2 = copy.copy(cm.Purples)
            nb_colors = 6
            colors2 = my_cmap2(np.linspace(0, 255, nb_colors).astype(int))
            bounds = np.arange(0, 1.61, 0.1)
            colors = np.r_[ [[0/255., 0/255., 0/255., 1.]],
                            colors1, colors2,
                            [[51/255., 51/255., 51/255., 1.]] ]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
            print('acr:', colors)
        elif colormap_style == 5: # New CALIOP browse image colorbar 
            bounds = np.array((0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6))
            # colors = ["#C7C7C7", "#000000", "#160D84", "#700AA4", "#B83A87", "#E8735C", "#FCC140", "#EFF941", "#FFFFFF"] # error in paper 1st submission
            colors = ["#C8C8C8", "#000000", "#0D0887", "#7100A8", "#BA3388", "#E97257", "#FDC229", "#F0F921", "#FFFFFF"]
            # my_cmap = mpl.colors.ListedColormap(cmaplist)
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 6: # Test for Xiaomei Lu to distinguish surface land, snow, and water
            bounds = np.array((0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6))
            colors = ["#C8C8C8", "#000000", "#0D0887", "#7100A8", "#BA3388", "#E97257", "#FDC229", "#F0F921", "#FFFFFF"]
            # my_cmap = mpl.colors.ListedColormap(cmaplist)
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 1000:
            my_cmap = cmlidar.cm.colorratio_9
            my_norm = cmlidar.cm.colorratio_9_norm
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
        else: # Colorblind diverging with few bins
            bounds = np.arange(0, 1.61, 0.2)
            YlOrBr_cmap = True
            if YlOrBr_cmap: # YlOrBr_r - Purples
                my_cmap1 = copy.copy(cm.YlOrBr_r)
                nb_colors = 6 # don't use first last (too light)
                colors1 = my_cmap1(np.linspace(0, 255, nb_colors).astype(int))
                my_cmap2 = copy.copy(cm.Purples)
                nb_colors = 4
                colors2 = my_cmap2(np.linspace(0, 255, nb_colors).astype(int))
                colors = np.r_[ [[0/255., 0/255., 0/255., 1.]],
                                [colors1[0]],
                                [colors1[1]],
                                [colors1[2]],
                                [colors1[3]],
                                [colors1[4]],
                                [colors2[0]],
                                [colors2[1]],
                                [colors2[2]],
                                [colors2[3]]]
            else: # YlGnBu_r - Greys
                my_cmap1 = copy.copy(cm.YlGnBu_r)
                nb_colors = 5
                colors1 = my_cmap1(np.linspace(0, 255, nb_colors).astype(int))
                my_cmap2 = copy.copy(cm.Greys)
                nb_colors = 6 # don't use first (white) last (black)
                colors2 = my_cmap2(np.linspace(0, 255, nb_colors).astype(int))
                colors = np.r_[ [[0/255., 0/255., 0/255., 1.]],
                                [colors1[0]],
                                [colors1[1]],
                                [colors1[2]],
                                [colors1[3]],
                                [colors1[4]],
                                [colors2[1]],
                                [colors2[2]],
                                [colors2[3]],
                                [colors2[4]]]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, acr.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        # print('acr:', colors)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_dist=True, flag_granule=False)
        # plt.text(0.02, 0.85, f"({grid})", ha='left', va='center', transform=fig.transFigure)
        plt.title(r"Attenuated Color Ratio $\mathit{\frac{\beta'_{1064}}{\beta'_{532}}}$"+f" ({VERSION_CAL_LID_L1})", 
                  fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, extend='both', drawedges=cbar_edges)
        # if cbar_edges: # comment when new version of matplotlib fixed bug of drawedges
        #     plt.axhline(max(bounds), color='k', linewidth=1.)
        #     plt.axhline(min(bounds), color='k', linewidth=1.)
        cbar.set_label(label=r"Attenuated Color Ratio", labelpad=5, fontsize=self.ytick_labelsize)
        
        # Colorbar ticks
        if COLORMAP == "LEGACY":
            cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
            cbar.ax.tick_params(labelsize=self.axes_labelsize) 
        # if self.colorbar_position == 'right':
        #     # cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        #     cbar.ax.yaxis.set_minor_locator(MultipleLocator(1000.))
        # elif self.colorbar_position == 'right':
        #     cbar.ax.xaxis.set_minor_locator(MultipleLocator(1000.))

        # Save figure
        filename = f"ACR_{grid}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)
        

    def plot_depolarization_ratio(self, depol, grid):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            depol = remove_edges(depol, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0], facecolor='k')

        # colormap_style = 1 # 1: CALIOP-like
        #                    # 2: linear
        #                    # 3: quantitative RdYlBu
        #                    # 4*: quantitative cubeh1_r
        #                    # else: linear with few bins
        if COLORMAP == "LEGACY":
            colormap_style = 1
        elif COLORMAP == "FRIENDLY":
            colormap_style = 1000
        else:
            colormap_style = 0

        if colormap_style == 1: # CALIOP-like colormap
            bounds = np.arange(0, 1.01, 0.1)
            colors = ['#000000',
                      '#16A8FC',
                      '#12D226',
                      '#FFFF39',
                      '#FEAA2E',
                      '#FE2025',
                      '#FE20FC',
                      '#FFD3FE',
                      '#A857FC',
                      '#FFFFFF',
                      '#FFFFFF',
                      '#FFFFFF']
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 2: # Colorblind linear
            my_cmap = takecmap('cubeh1_r', nb_colors=256, clight=1., cdark=0.15)
            # my_cmap = copy.copy(cm.YlGnBu_r)
            # my_cmap = copy.copy(cm.YlOrBr_r)
            # my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 1.01, 0.1)
            nb_colors = len(bounds) - 2 # -2 because same color above 0.8
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [[0., 0., 0., 1.]],
                            colors, [colors[-1]],
                            [[1., 1., 1., 1.]]] # same color above 0.8, min black, max white
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 3: # Colorblind quantitative
            # my_cmap = takecmap('cubeh1_r', nb_colors=256, clight=1., cdark=0.15)
            my_cmap = copy.copy(cm.RdYlBu_r)
            bounds = np.arange(0, 1.01, 0.1)
            nb_colors = len(bounds) - 4 # only 7 colors from cmap
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [[0.00, 0.00, 0.00, 1.00]],
                            colors,
                            [[0.52, 0.07, 0.15, 1.00]],
                            [[0.40, 0.05, 0.12, 1.00]],
                            [[0.28, 0.04, 0.08, 1.00]],
                            [[0.16, 0.02, 0.05, 1.00]]]
            # Replace 2 first blues to lighter blues
            colors[1] = [0.24, 0.27, 0.59, 1.00]
            colors[2] = [0.49, 0.58, 0.76, 1.00]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 4: # Colorblind quantitative extviridis
            my_cmap = takecmap('cubeh1_r')
            bounds = np.arange(0, 0.51, 0.1)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 5: # Proposal for new CALIOP browse image colorbar 
            bounds = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
            nb_colors = len(bounds) - 3
            my_cmap = cm.inferno
            color_indexes = np.linspace(100, 215, nb_colors).astype(int) # max 215 - min 70, 80, 90 or 100
            colors = my_cmap(color_indexes) 
            colors = np.r_[ \
                            # [[230/255, 230/255, 230/255, 1.00]],
                            [[135/255, 206/255, 250/255, 1.00]], #[[135/255, 206/255, 250/255, 1.00]] = light blue, [[219/255, 239/255, 255/255, 1.00]] = lighter blue, 230 = grey
                            # [[0/255, 0/255, 0/255, 1.00]], # [[56/255, 123/255, 224/255, 1.00]] = blue, [[135/255, 206/255, 250/255, 1.00]] = light blue
                            [[0/255, 0/255, 0/255, 1.00]], # black
                            colors,
                            [[239/255, 249/255, 127/255, 1.00]], # bright yellow
                            [[1.00, 1.00, 1.00, 1.00]]]
            # my_cmap = mpl.colors.ListedColormap(cmaplist)
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 6: # Test 
            bounds = np.array((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
            # palette = ['#0a0a0a', '#000004', '#85216b', '#f78410', '#fcae12', '#f5db4c', '#fcffa4', '#ffffff'] # contrast low values
            # palette = ['#87CEFA', '#000004', '#140b34', '#85216b', '#e65d2f', '#f5db4c', '#fcffa4', '#ffffff'] # contrast medium values
            # palette = ['#87CEFA', '#000004', '#140b34', '#390963', '#5f136e', '#cb4149', '#fcae12', '#ffffff'] # contrast high values
            # palette = ['#87CEFA', '#000004', '#0e092b', '#2d0b59', '#4f0d6c', '#6d186e', '#8d2369', '#e15635', '#fbbc21', '#ffffff'] # contrast very high values
            # palette = ['#87CEFA', '#000000', '#333333', '#686868', '#979797', '#c6c6c6', '#e9e9e9', '#ffffff', ] # Greys
            # palette = ["#87CEFA", '#ff00ff', '#ff2ad5', '#ff55aa', '#ff7f80', '#ffaa55', '#ffd42b', '#ffff00'] # spring1
            # palette = ["#87CEFA", '#0b0000', '#790000', '#ea0000', '#ff5900', '#ffca00', '#ffff56', '#ffffff'] # hot1
            # palette = ['#BBBBBB', '#000000', '#902116', '#ef9800', '#ffff00', '#e4e4e4', '#f2f2f1', '#ffffff'] # RdYlG1
            # palette = ['#55baf8', '#000000', '#902116', '#cf7007', '#f4ba00', '#ffff00', '#dddddd', '#ffffff'] # RdYlG2
            # palette = ['#aaaaaa', '#000000', '#d62f27', '#fba05b', '#fff2ac', '#b3a3cc', '#684f9a', '#000000'] # RdYlBu10
            # palette = ['#aaaaaa', '#000000', '#d62f27', '#fba05b', '#fff2ac', '#a3d3e6', '#588cc0', '#313695'] # RdYlBu9
            # palette = ['#999999', '#000000', '#a60f00', '#ef9800', '#ffff00', '#5fc6ff', '#3068e3', '#000ac6'] # RdYlBu8
            # palette = ['#DDDDDD', '#000000', '#a60f00', '#ef9800', '#ffff00', '#5fc6ff', '#3068e3', '#000ac6'] # RdYlBu7
            # palette = ['#000ac6', '#3068e3', '#5fc6ff', '#ffff00', '#ef9800', '#a60f00', '#000000', '#DDDDDD'] # RdYlBu7_r
            # palette = ['#DDDDDD', '#000000', '#a60f00', '#ef9800', '#ffff00', '#3f87ec', '#000ac6', '#ffffff'] # RdYlBu6
            # palette = ['#DDDDDD', '#000000', '#a60f00', '#ef9800', '#ffff00', '#5fc6ff', '#000ac6', '#ffffff'] # RdYlBu5
            # palette = ['#ffffff', '#000ac6', '#5fc6ff', '#ffff00', '#ef9800', '#a60f00', '#000000', '#DDDDDD'] # RdYlBu5_r
            # palette = ['#DDDDDD', '#000000', '#902116', '#ef9800', '#ffff00', '#85BBD9', '#313695', '#ffffff'] # RdYlBu4
            # palette = ['#000000', '#a50026', '#f46d43', '#fee090', '#e0f3f8', '#74add1', '#313695', '#1e215a'] # RdYlBu3
            # palette = ["#000000", '#FF0000', '#FFAA00', '#FFFF55', '#FFFFFF', '#AFDEF2', '#5A89F2', '#0000FF'] # RdYlBu2
            # palette = ["#444444", '#FF0000', '#FFAA00', '#FFFF55', '#AFDEF2', '#5A89F2', '#0000FF', "#000000"] # RdYlBu1
            # palette = ["#444444", '#902116', '#ef9800', '#ffff00', '#ffc0cb', '#cb5dca', '#0000a4', "#000000"] # RdYlPkPuBu2
            # palette = ["#87CEFA", '#902116', '#ef9800', '#ffff00', '#ffc0cb', '#cb5dca', '#0000a4', "#ffffff"] # RdYlPkPuBu1
            # palette = ["#87CEFA", '#000000', '#902568', '#CA404A', '#F1731D', '#FBBA1F', '#EFF97F', "#ffffff"] # Colorbar4
            # palette = ["#0F7098", '#000000', '#9f395e', '#e7582e', '#ff9b23', '#ffdf3f', '#ffff55', '#ffffff'] # Colorbar14
            # palette = ["#aaaaaa", '#000000', '#a50026', '#da362a', '#f67a49', '#fdbf71', '#feeda4', "#ffffff"] # Colorbar15
            # palette = ["#AAAAAA", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR16
            # palette = ["#87CEFA", '#ffffff', '#f8cd37', '#e55c30', '#8a226a', '#57106e', '#210c4a', '#000004'] # InfernoR15
            # palette = ['#ffffff', '#f4e156', '#f67e14', '#c03a51', '#6d186e', '#440a68', '#180c3c', '#000004'] # InfernoR14
            # palette = ['#ffffff', '#f8cd37', '#f8870e', '#d94d3d', '#a32c61', '#69166e', '#290b55', '#000004'] # InfernoR13
            # palette = ["#87CEFA", '#ffffff', '#fbb61a', '#ed6925', '#ba3655', '#781c6d', '#320a5e', '#000004'] # InfernoR12
            # palette = ["#87CEFA", '#ffffff', '#ffd822', '#ff480b', '#c10064', '#52008a', '#050062', '#000000'] # InfernoR11
            # palette = ["#87CEFA", '#ffffff', '#f8cd37', '#ec6726', '#a32c61', '#4a0c6b', '#1b0c41', '#000000'] # InfernoR10
            # palette = ["#87CEFA", '#fcffa4', '#fbb61a', '#ed6925', '#ba3655', '#781c6d', '#320a5e', '#000004'] # InfernoR9
            # palette = ["#ffffff", '#fcffa4', '#fbb61a', '#ed6925', '#ba3655', '#781c6d', '#320a5e', '#000004'] # InfernoR8
            # palette = ["#87CEFA", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7 
            # palette = ["#87CEFA", "#ffffff", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR6 
            # palette = ["#333333", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#21225E', '#000000'] # InfernoR5
            # palette = ["#DDDDDD", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#21225E', '#000000'] # InfernoR4
            # palette = ["#60A9D4", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#21225E', '#000000'] # InfernoR3
            # palette = ["#ffffff", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#21225E', '#000000'] # InfernoR2
            # palette = ["#333333", '#fcffa4', '#fbb61a', '#ed6925', '#ba3655', '#781c6d', '#320a5e', '#000004'] # InfernoR1 (standard)
            # palette = ["#AAAAAA", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno11
            # palette = ["#87CEFA", '#ffffff', '#ffff00', '#ffb40d', '#ff6b26', '#cf0097', '#0029e4', "#000000"] # Inferno10r
            # palette = ["#87CEFA", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10
            # palette = ["#87CEFA", '#000000', '#4040ff', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno9
            # palette = ["#000000", '#3333cc', '#ac007e', '#ff4627', '#FAC228', '#F3E55D', '#FCFFA4', '#ffffff'] # Inferno8
            # palette = ["#BBBBBB", '#4A0C6B', '#A52C60', '#ED6925', '#f7d13d', '#fcf014', '#ffff55', '#ffffff'] # Inferno7
            # palette = ["#000000",  '#5d38d8', '#bb0d81', '#e20715', '#f8771b', '#ffbc12', '#f5fb4a', "#ffffff"] # Inferno6
            # palette = ["#c1e7ff", '#000000', '#3333cc', '#ac007e', '#ff4627', '#ffb40d', '#ffff00', "#ffffff"] # Inferno5
            # palette = ["#87CEFA", '#000000', '#3333cc', '#ac007e', '#ff4627', '#ffb40d', '#ffff00', "#ffffff"] # Inferno4 
            # palette = ["#60A9D4", "#ffffff", '#ffff00', '#ffb40d', '#ff4627', '#ac007e', '#3333cc', '#000000'] # Inferno4_r 
            # palette = ["#87CEFA", '#000000', '#6f0081', '#d70054', '#ff5116', '#ffad0e', '#ffff00', "#ffffff"] # Inferno3 (100 % saturation)
            # palette = ["#87CEFA", '#000000', '#0000ff', '#ac007e', '#ff4627', '#ffb40d', '#efff13', "#ffffff"] # Inferno2 (100 % saturation)
            # palette = ["#87CEFA", '#000000', '#0000a7', '#ac007e', '#fc482a', '#f6b116', '#efff13', "#ffffff"] # Inferno1 (more saturated colors)
            # palette = ["#87CEFA", '#000000', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4', "#ffffff"] # Inferno0 (standard)
            # palette = ["#87CEFA", "#680605", "#455cc1", "#d548b8", "#c1ae00", "#a0e577", "#fff568", "#ffffff"]
            # palette = ["#000000", "#a42222", "#9360e2", "#f9615c", "#ff9c4b", "#ffcc59", "#f8f965", "#ffffff"]
            # palette = ["#87CEFA", "#000000", "#810702", "#4048F0", "#279D81", "#F0AC34", "#F2FD50", "#ffffff"]
            # palette = ["#87CEFA", '#000000', '#481d6f', '#33628d', '#1f9a8a', '#69cd5b', '#fde725', "#ffffff"] # Viridis1 (saturated)
            # palette = ["hsl(0, 0%, 78%)", '#000000', '#3333cc', '#33628d', '#1f9a8a', '#69cd5b', '#fde725', "#ffffff"] # Viridis2 (saturated blue)
            # palette = ["#C7C7C7", '#000000', '#0003c8', '#008eb8', '#00ca94', '#6aff23', '#ffe923', "#ffffff"] # Viridis3 (saturated)
            # palette = ["#87CEFA", '#000000', '#3333cc', '#33628d', '#1f9a8a', '#69cd5b', '#fde725', "#ffffff"] # Viridis4
            # palette = ['#000000', '#3333cc', '#33628d', '#1f9a8a', '#69cd5b', '#fde725', "#ffffff", "#ffffff"] # Viridis5
            # palette = ["#87CEFA", '#000000', '#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#ffffff'] # Viridis6
            # palette = ['#000000', '#0003c8', '#0099d8', '#00c152', '#00e400', '#fbfb3b', '#ffffff', '#ffffff'] # Viridis7
            # palette = ['#000000', '#453781', '#31668e', '#218f8d', '#34b679', '#8ed645', '#fde725', '#ffffff'] # Viridis8
            # palette = ['#87CEFA', '#000000', '#31668e', '#218f8d', '#34b679', '#8ed645', '#fde725', '#ffffff'] # Viridis9
            # palette = ['#555555', '#000000', '#23272a', '#218f8d', '#34b679', '#8ed645', '#fde725', '#ffffff'] # Viridis10
            # palette = ['#000000', '#8d0000', '#3333cc', '#0099d8', '#00c152', '#00e400', '#fbfb3b', '#ffffff'] # BrownViridis1
            # palette = ["#87CEFA", '#ffffff', '#fde725', '#5cc863', '#21908d', '#3b518b', '#440154', '#000000'] # ViridisR1
            # palette = ["#333333", '#ffffff', '#cae11f', '#3fbc73', '#24868e', '#3e4c8a', '#440154', '#000000'] # ViridisR2
            # palette = ["#87CEFA", '#fde725', '#7ad151', '#22a884', '#2a788e', '#414487', '#440154', '#000000'] # ViridisR3
            # palette = ["#999999", '#ffffff', '#cae11f', '#3fbc73', '#038b97', '#233ba4', '#66007f', '#000000'] # ViridisR4
            # palette = ["#555555", '#000000', '#7E03A8', '#CC4778', '#F89540', '#F0F921', '#fefca0', '#ffffff'] # Plasma1
            # palette = ["#000000", '#8E0CA4', '#C43E7F', '#E97257', '#FDAF31', '#F0F921', '#fefca0', '#ffffff'] # Plasma2
            # palette = ["#000000", '#6A00A8', '#B12A90', '#E16462', '#FCA636', '#F0F921', '#fefca0', '#ffffff'] # Plasma3
            # palette = ["#AAAAAA", '#0d0887', '#7e03a8', '#cc4778', '#f89540', '#fdc527', '#f0f921', '#ffffff'] # Plasma4
            # palette = ["#000000", '#0d0887', '#7e03a8', '#cc4778', '#f89540', '#fdc527', '#f0f921', '#ffffff'] # Plasma5
            # palette = ["#000000", '#0d0887', '#8f0da4', '#e16462', '#fca636', '#fcce25', '#f0f921', '#ffffff'] # Plasma6
            # palette = ["#000000", '#4903a0', '#b42e8d', '#f3854b', '#fada24', '#f0f921', '#fefca0', '#ffffff'] # Plasma7
            # palette = ["#87CEFA", '#000000', '#b42e8d', '#f3854b', '#fada24', '#f0f921', '#fefca0', '#ffffff'] # Plasma8
            # palette = ["#777777", '#000000', '#7E03A8', '#CC4778', '#F89540', '#F0F921', '#fefca0', '#ffffff'] # Plasma9
            # palette = ["#aaaaaa", '#000000', '#8707a6', '#c03a83', '#e87059', '#fdae32', '#f0f921', '#ffffff'] # Plasma10
            # palette = ["#aaaaaa", '#000000', '#c43e7f', '#e26561', '#f79044', '#fdc229', '#f0f921', '#ffffff'] # Plasma11
            # palette = ["#aaaaaa", '#000000', '#5901a5', '#a72197', '#dd5e66', '#fca338', '#f0f921', '#ffffff'] # Plasma12
            # palette = ["#aaaaaa", '#000000', '#0000ff', '#c13f88', '#e97b59', '#f6b93f', '#f0f921', '#ffffff'] # Plasma13 (from Plasma12 and saturating the blue in chroma.js)
            # palette = ["#aaaaaa", '#000000', '#5d34ff', '#d15a7f', '#ed8d54', '#f4c340', '#f0f921', '#ffffff'] # Plasma14 (from Plasma13 and lightening the blue in chroma.js)
            # palette = ["#aaaaaa", '#000000', '#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921', '#ffffff'] # Plasma15
            # palette = ["#aaaaaa", '#000000', '#0000ff', '#c33f7f', '#e57d5f', '#f3ba43', '#f0f921', '#ffffff'] # Plasma16 (from Plasma15 and saturating the blue in chroma.js)
            # palette = ["#aaaaaa", '#000000', '#5d34ff', '#d6596f', '#eb8e57', '#f4c33f', '#f0f921', '#ffffff'] # Plasma17 (from Plasma16 and lightening the blue in chroma.js)
            # palette = ["#aaaaaa", '#000000', '#7b4bff', '#df6c6a', '#ef9a4f', '#f4ca3d', '#f0f921', '#ffffff'] # Plasma18
            # palette = ["#87CEFA", '#000000', '#7b4bff', '#df6c6a', '#ef9a4f', '#f4ca3d', '#f0f921', '#ffffff'] # Plasma19
            # palette = ["#444444", '#ffffff', '#f0f921', '#f89441', '#cb4679', '#7d03a8', '#0d0887', '#000000'] # PlasmaR1
            # palette = ["#444444", '#ffffff', '#fdc627', '#ec7754', '#bc3587', '#7201a8', '#0d0887', '#000000'] # PlasmaR2
            # palette = ["#cccccc", '#ffffff', '#fdc627', '#ec7754', '#bc3587', '#7201a8', '#0d0887', '#000000'] # PlasmaR3
            # palette = ["#aaaaaa", '#ffffff', '#fdc627', '#ec7754', '#bc3587', '#7201a8', '#0d0887', '#000000'] # PlasmaR4
            # palette = ["#87CEFA", '#ffffff', '#fdc627', '#ec7754', '#bc3587', '#7201a8', '#0d0887', '#000000'] # PlasmaR5
            # palette = ["#60A9D4", '#ffffff', '#fdc627', '#ec7754', '#bc3587', '#7201a8', '#0d0887', '#000000'] # PlasmaR6
            # palette = ["#888888", '#fbd724', '#f79044', '#d7566c', '#a72197', '#6400a7', '#0d0887', '#000000'] # PlasmaR7
            # palette = ["#87CEFA", '#ffffff', '#ffc400', '#f7874c', '#d44f89', '#9221b8', '#0000cb', '#000000'] # PlasmaR8
            # palette = ["#87CEFA", '#ffffff', '#f4cd2f', '#f0874a', '#cb4679', '#8307a5', '#1b078d', '#000000'] # Decreasing_Plasma_1 (full plasma with ligthness corrected)
            # palette = ["#87CEFA", '#ffffff', '#ffd324', '#ff843d', '#ff146e', '#8800ad', '#160094', '#000000'] # Decreasing_Plasma_2 (Decreasing_Plasma_1 + saturation at 100%)
            # palette = ["#87CEFA", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # Decreasing_Plasma_3 (Decreasing_Plasma_2 + correct lightness)
            # palette = ["#AAAAAA", '#000000', '#1a078b', '#8307a5', '#cb4679', '#f0874a', '#f4cd2f', '#ffffff'] # Increasing_Plasma_1 (full plasma with ligthness corrected)
            # palette = ["#AAAAAA", '#000000', '#160094', '#8800ad', '#ff146e', '#ff843d', '#ffd324', '#ffffff'] # Increasing_Plasma_2 (Increasing_Plasma_1 + saturation at 100%)
            # palette = ["#AAAAAA", '#000000', '#160094', '#8700ad', '#ed1178', '#ff873c', '#ffd324', '#ffffff'] # Increasing_Plasma_3 (Increasing_Plasma_2 + correct lightness)
            # palette = ["#AAAAAA", '#000000', '#5901a5', '#b32e8c', '#e46f5b', '#fab234', '#f0f921', '#ffffff'] # Increasing_Plasma_4 (40-256 plasma with ligthness corrected except black-blue and yellow-white steps)
            # palette = ["#AAAAAA", '#000000', '#5a00a8', '#e0009d', '#ff5f42', '#ffb22e', '#f7ff1a', '#ffffff'] # Increasing_Plasma_5 (Increasing_Plasma_4 + saturation at 100%)
            # palette = ["#AAAAAA", '#000000', '#5a00a8', '#c2009f', '#ff6242', '#ffb72d', '#f7ff1a', '#ffffff'] # Increasing_Plasma_6 (Increasing_Plasma_5 + correct lightness except black-blue and yellow-white steps)
            # palette = ["#AAAAAA", '#000000', '#7801a8', '#c13d81', '#ea7854', '#fbb730', '#f0f921', '#ffffff'] # Increasing_Plasma_7 (60-256 plasma with ligthness corrected except black-blue and yellow-white steps)
            # palette = ["#AAAAAA", '#000000', '#7900a8', '#c13d81', '#ea7854', '#fbb730', '#f0f921', '#ffffff'] # Increasing_Plasma_8 (Increasing_Plasma_7 + saturation at 100%)
            # palette = ["#AAAAAA", '#000000', '#5a00a8', '#c2009f', '#ff6242', '#ffb72d', '#f7ff1a', '#ffffff'] # Increasing_Plasma_9 (Increasing_Plasma_8 + correct lightness except black-blue and yellow-white steps)
            # palette = ["#87CEFA", '#000004', '#320a5e', '#781c6d', '#bc3754', '#ed6925', '#fbb61a', '#fcffa4'] # Increasing_Inferno_1
            # palette = ["#000000", '#366525', '#358876', '#728bc5', '#c68fd1', '#ecb1ba', '#e9e0c6', '#ffffff'] # Cubehelix
            # palette = ["#000000", '#8d0000', '#813ad9', '#8f7fa5', '#93b652', '#bcde24', '#ffff00', '#ffffff'] # Chroma1
            # palette = ["#87CEFA", '#000000', '#bf0000', '#1b1a1c', '#94b06a', '#b9dc29', '#ffff00', '#ffffff'] # Chroma2
            # palette = ["#bbbbbb", '#000000', '#bf0000', '#5379f2', '#4cb4cb', '#a3e03c', '#fbff00', '#ffffff'] # Chroma3
            # palette = ["#999999", '#000000', '#a50026', '#b3559d', '#d08e94', '#ebc670', '#ffff00', '#ffffff'] # Chroma4
            # palette = ['#000000', '#424bb5', '#31aff5', '#38f491', '#bef434', '#fea933', '#df3f08', '#7a0403'] # Turbo1
            # palette = ['#000000', '#da3907', '#fe9b2d', '#d2e935', '#61fc6c', '#1bd0d5', '#4776ee', '#30123b'] # Turbo2
            # palette = ["#AAAAAA", '#000000', '#1f61f8', '#c86ab8', '#f38f8c', '#ebc764', '#d3fc25', '#ffffff'] # Viscm_1 (with ligthness corrected except black-blue and yellow-white steps)
            # palette = ["#AAAAAA", '#000000', '#0061fe', '#f35b53', '#f69c0d', '#f4d10c', '#ffff00', '#ffffff'] # Viscm_2 (with ligthness corrected except black-blue and yellow-white steps and bezier interp, gap in the viscm colormap)
            # palette = ["#AAAAAA", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # Viscm_3 (with ligthness corrected except black-blue and yellow-white steps)
            # palette = ["#AAAAAA", '#000000', '#bb0fff', '#ff4da3', '#ff9780', '#ffd557', '#ffff00', '#ffffff'] # Viscm_4 (Viscm_3 + saturation at 100%)
            # palette = ["#AAAAAA", '#000000', '#bb0fff', '#ff52a1', '#ff9a7e', '#ffcf5b', '#ffff00', '#ffffff'] # Viscm_5 (Viscm_4 + correct lightness except black-blue and yellow-white steps)
            # palette = ["#808080", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingG1 (= Viscm_3 with '0.5' negative color)
            # palette = ["#9E9E9E", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingG2 (= Viscm_3 with '0.62' negative color)
            # palette = ["#BFBFBF", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingG3 (= Viscm_3 with '0.75' negative color)
            # palette = ["#DDDDDD", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingG4 (= Viscm_3 with '0.87' negative color) 
            # palette = ["#0A9BF4", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingB1 (= Viscm_3 with '0.5' negative color)
            # palette = ["#44B3F7", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingB2 (= Viscm_3 with '0.62' negative color)
            # palette = ["#87CEFA", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingB3 (= Viscm_3 with '0.75' negative color)
            # palette = ["#C1E6FC", '#000000', '#b61bf2', '#e962a4', '#f79b86', '#f6d15e', '#ffff00', '#ffffff'] # IncreasingB4 (= Viscm_3 with '0.87' negative color)
            # palette = ["#808080", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingG1 (= Decreasing_Plasma_3 with '0.5' negative color)
            # palette = ["#9E9E9E", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingG2 (= Decreasing_Plasma_3 with '0.62' negative color)
            # palette = ["#BFBFBF", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingG3 (= Decreasing_Plasma_3 with '0.75' negative color)
            # palette = ["#DDDDDD", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingG4 (= Decreasing_Plasma_3 with '0.87' negative color)
            # palette = ["#0A9BF4", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingB1 (= Decreasing_Plasma_3 with '0.5' negative color)
            # palette = ["#44B3F7", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingB2 (= Decreasing_Plasma_3 with '0.62' negative color)
            # palette = ["#87CEFA", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingB3 (= Decreasing_Plasma_3 with '0.75' negative color)
            # palette = ["#C1E6FC", '#ffffff', '#ffd324', '#ff873c', '#ed1178', '#8700ad', '#160094', '#000000'] # DecreasingB4 (= Decreasing_Plasma_3 with '0.87' negative color)
            # palette = ["#808080", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10G1
            # palette = ["#9E9E9E", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10G2
            # palette = ["#BFBFBF", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10G3
            # palette = ["#DDDDDD", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10G4
            # palette = ["#0A9BF4", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10B1
            # palette = ["#44B3F7", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10B2
            # palette = ["#87CEFA", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10B3
            # palette = ["#C1E6FC", '#000000', '#0029e4', '#cf0097', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10B4
            # palette = ["#808080", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7G1 
            # palette = ["#9E9E9E", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7G2
            # palette = ["#BFBFBF", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7G3
            # palette = ["#DDDDDD", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7G4
            # palette = ["#0A9BF4", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7B1
            # palette = ["#44B3F7", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7B2
            # palette = ["#87CEFA", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7B3
            # palette = ["#C1E6FC", '#ffffff', '#ffd309', '#ff9716', '#ff4627', '#ac007e', '#3333cc', '#000000'] # InfernoR7B4
            # palette = ["#808080", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bG1
            # palette = ["#9E9E9E", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bG2
            # palette = ["#BFBFBF", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bG3
            # palette = ["#DDDDDD", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bG4
            # palette = ["#0A9BF4", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bB1
            # palette = ["#44B3F7", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bB2
            # palette = ["#87CEFA", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bB3
            # palette = ["#C1E6FC", '#000000', '#3b40ff', '#e919a9', '#ff6b26', '#ffb40d', '#ffff00', "#ffffff"] # Inferno10bB4
            # palette = ["#808080", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bG1 
            # palette = ["#9E9E9E", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bG2
            # palette = ["#BFBFBF", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bG3
            # palette = ["#DDDDDD", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bG4
            palette = ["#0A9BF4", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bB1 
            # palette = ["#44B3F7", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bB2
            # palette = ["#87CEFA", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bB3
            # palette = ["#C1E6FC", '#ffffff', '#ffd309', '#ff961b', '#ff4627', '#b4237e', '#3333cc', '#000000'] # InfernoR7bB4
            

            my_cmap = mpl.colors.ListedColormap(palette)
            colors = my_cmap(np.arange(len(palette)))
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        elif colormap_style == 1000:
            my_cmap = cmlidar.cm.depol_8
            my_norm = cmlidar.cm.depol_8_norm
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = False
        else: # Colorblind linear with few bins
            my_cmap = takecmap('cubeh1_r', nb_colors=256, clight=1., cdark=0.)
            # my_cmap = copy.copy(cm.YlGnBu_r)
            # my_cmap = copy.copy(cm.YlOrBr_r)
            # my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 1.01, 0.1)
            nb_colors = len(bounds) + 1 - 6 # -6 we'll duplicate 5 colors and max same previous
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [colors[0]],
                            [colors[1]], [colors[1]],
                            [colors[2]], [colors[2]],
                            [colors[3]], [colors[3]],
                            [colors[4]], [colors[4]],
                            [colors[5]], [colors[5]],
                            [colors[5]] ]
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
            pc = plt.pcolormesh(self.pindexbins, self.altbins, depol.T, cmap=my_cmap,
                                norm=my_norm, rasterized=True)
            cbar_edges = True
        
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        # self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_dist=False)
        # plt.text(0.02, 0.85, f"({grid})", ha='left', va='center', transform=fig.transFigure)
        plt.title(r"Depolarization Ratio $\mathit{\frac{\beta'_{\perp}}{\beta'_{\parallel}}}$"+f" ({VERSION_CAL_LID_L1})",
                  fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # colorbar_name = "Current"
        # plt.title("$\mathbf{%s}$" % colorbar_name, y=1.03)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, extend='both', drawedges=cbar_edges)
        # if cbar_edges: # comment when new version of matplotlib fixed bug of drawedges
        #     plt.axhline(max(bounds), color='k', linewidth=1.)
        #     plt.axhline(min(bounds), color='k', linewidth=1.)
        cbar.set_label(label=r"Depolarization Ratio", labelpad=5)
        
        # Colorbar ticks
        # cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
        # cbar.ax.yaxis.set_minor_locator(MultipleLocator(1000.))
        cbar.ax.yaxis.set_minor_locator(FixedLocator(np.array((-999,))))

        # Add bin labels to the colorbar
        if False:
            # for j, mid_bin in enumerate(np.arange(0.05, 0.55+.0001, 0.1)):
            #     cbar.ax.text(1.5, mid_bin, f"{mid_bin:.1f}", va='center', weight='bold', c=palette[j+2], fontsize=8, path_effects=[pe.withStroke(linewidth=1, foreground="k")])
            # outline_colors = ['0.5', '0.2', 'k', 'k', 'k', 'k']
            # outline_colors = ['k', 'k', 'k', 'k', 'k', '0.2']
            # color_names = ['Black', 'Blue', 'Pink', 'Orange', 'Amber', 'Yellow']
            # text_colors = ['w', 'w', 'w', 'k', 'k', 'k']
            color_names = ['White', 'Yellow', 'Orange', 'Red', 'Purple', 'Blue']
            # text_colors = ['k', 'k', 'k', 'k', 'w', 'w']
            for j, color_name in enumerate(color_names):
                # cbar.ax.text(-1.1, j/10+0.05, color_name, va='center', rotation=90, weight='bold', c=palette[j+1], fontsize=5, path_effects=[pe.withStroke(linewidth=0.5, foreground=outline_colors[j])])
                cbar.ax.text(-1.1, j/10+0.05, color_name, va='center', rotation=90, weight='bold', c='k', fontsize=3)
                # cbar.ax.text(0.5, j/10+0.05, color_name, va='center', ha='center', rotation=90, weight='bold', c=text_colors[j], fontsize=5)

        # Save figure
        filename = f"DR_{grid}"
        self.save_fig(filename)

        # # Filename  (for test_colorbar)
        # self.fig_folder = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/test_colorbar/"
        # filename = f"DR_{grid}_{colormap_style}"
        # # filename = f"DR_{grid}_0Current"

        # Save figure
        # self.save_fig(filename)

        # Save data in pickle (for test_colorbar)
        # depol.dump(self.fig_folder+self.head_filename+'.pkl')

        
        # Close figure
        plt.close(fig)
        

    def plot_histo(self, atb, wl, polar, data_type, xmin, xmax):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            atb = remove_edges(atb, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")
        
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        ax0 = plt.subplot(111)

        # Plot
        atb_no_mask = atb.compressed()
        plt.hist(atb_no_mask, 1000, (xmin, xmax))
        plt.xlim(xmin, xmax)
        ax0.set_yscale('log')
        if polar=='par':
            polar_txt = '(parallel)'
        elif polar=='per':
            polar_txt = '(perpendicular)'
        else:
            polar_txt = ''
        if data_type=='AB':
            plt.xlabel(r"$\beta'$ (km$^{-1}$ sr$^{-1}$)")
            plt.title(r'$\mathbf{Attenuated\ backscatter\ at\ %d\ nm\ %s}$'\
                        % (wl, polar_txt), y=1.)
        if data_type=='ASR':
            plt.xlabel("$R'$")
            plt.title(r'$\mathbf{Attenuated\ scattering\ ratio\ at\ %d\ nm\ %s}$'\
                        % (wl, polar_txt), y=1.)
        plt.ylabel("Occurence")

        # Save figure
        filename = f"{data_type}_{wl}_{polar}_histo"
        self.save_fig(filename, adjust=(0.15, 0.2, 0.95, 0.9))

        # Close figure
        plt.close(fig)
        

    def plot_attenuated_backscatter_mol(self, atb, wl, polar, grid):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            atb = remove_edges(atb, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")
        
        my_cmap = takecmap("extviridis")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        
        # Plot
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, atb.T, cmap=my_cmap)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        if polar=='par':
            polar_txt = '(parallel)'
        elif polar=='per':
            polar_txt = '(perpendicular)'
        else:
            polar_txt = ''
        plt.title(r'$\mathbf{Estimated\ molecular\ attenuated\ backscatter\ at\ %d\ nm\ %s}$'\
                    % (wl, polar_txt), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, f"({grid})", ha='left', va='center', transform=fig.transFigure)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation)
        cbar.set_label(label=r"$\beta'$ (km$^{-1}$ sr$^{-1}$)", labelpad=30)

        # Save figure
        filename = f"ABmol{wl:d}{polar}_{grid}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_asr_sigma(self, asr_sigma, wl, polar):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            asr_sigma = remove_edges(asr_sigma, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])
        my_cmap = takecmap("extviridis_r")
        pc = plt.pcolormesh(self.pindexbins, self.altbins, asr_sigma.T, cmap=my_cmap)
        manual_clim = True
        if manual_clim:
            plt.clim(0, 500)
        else:
            plt.clim(0, np.max(asr_sigma))
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        if polar=='par':
            polar_txt = '(parallel)'
        elif polar=='per':
            polar_txt = '(perpendicular)'
        else:
            polar_txt = ''
        plt.title(r'$\mathbf{Estimated\ noise\ standard\ deviation\ at\ %d\ nm\ %s}$' %\
                (wl, polar_txt), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation)
        cbar.ax.yaxis.get_major_formatter()._usetex = False
        cbar.set_label(label=r"$\Delta R'$")

        # Save figure
        filename = f"ASR_sigma{wl:d}{polar}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_asr(self, asr, wl, polar, grid):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            asr = remove_edges(asr, EDGES_REMOVAL)
            
        # Put negative values to 1e-9 in order to plot wiht LogNorm
        asr[asr<0] = 1e-9

        # Figure style
        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])

        # Colormap
        my_cmap = takecmap('extviridis_black_white')
        my_cmap.colorbar_extend = 'both'
        # my_cmap = copy.copy(cm.viridis)
        # my_cmap.set_under([0,0,0,1]) # add black for below min
        # my_cmap.set_over([1,1,1,1]) # add white for over max

        pc = plt.pcolormesh(self.pindexbins, self.altbins, asr.T, cmap=my_cmap, norm=LogNorm())
        if polar=='per':
            plt.clim(1, 1e4)
        elif wl==1064:
            plt.clim(1e2, 1e3)
        else:
            plt.clim(1e-1, 1e2)

        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        if polar == 'par':
            polar_txt = '(parallel)'
        elif polar == 'per':
            polar_txt = '(perpendicular)'
        else:
            polar_txt = ''
        # plt.text(0.02, 0.85, f"({grid})", ha='left', va='center', transform=fig.transFigure)
        plt.title(r'$\mathbf{Attenuated\ scattering\ ratio\ at\ %d\ nm\ %s}$' % \
                (wl, polar_txt), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, extend='both')
        cbar.set_label(label="$R'$")
        # cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))

        # Save figure
        filename = f"ASR{wl:d}{polar}_{grid}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_asr_above_sigma(self, asr, asr_sigma, wl, polar):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            asr = remove_edges(asr, EDGES_REMOVAL)
            asr_sigma = remove_edges(asr_sigma, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")

        k = 1
        asr_sigma_sup = np.zeros(asr.shape)
        asr_sigma_sup[asr>(1+k*asr_sigma)] = 1
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        gs01 =  gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1],
                                                height_ratios=[1, 1, 1])
        ax0 = plt.subplot(gs0[0])
        cmaplist = ['w', 'r']
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(3) - 0.5 # '0=inf', '1=sup'
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, asr_sigma_sup.T, cmap=my_cmap,
                            norm=my_norm)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        if polar=='par':
            polar_txt = '(parallel)'
        elif polar=='per':
            polar_txt = '(perpendicular)'
        else:
            polar_txt = ''
        plt.title(r'$\mathbf{Attenuated\ scattering\ ratio\ %d\ nm\ %s}$' %\
                (wl, polar_txt), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs01[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate([r"$< 1 + %g \Delta R'$" % k,
                                r"$> 1 + %g \Delta R'$" % k]):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=7, transform=ax1.transAxes)

        # Save figure
        filename = f"ASR{wl:d}{polar}_supsigma"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_vfm_type(self, vfm_type):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            vfm_type = remove_edges(vfm_type, EDGES_REMOVAL)
            
        # Figure style
        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])
        if COLORMAP == "LEGACY":
            cmaplist = ['#777777',
                        "#0026FF",
                        "#00DCFF",
                        "#FFA000",
                        "#FAFF00",
                        "#00FF6E",
                        "#C0C0C0",
                        "#000000"]
        elif COLORMAP == "FRIENDLY":
            cmaplist = ["#999999",
                        "#77B3FB",
                        "#FFFFF0",
                        "#F3CF4F",
                        "#FE9F6D",
                        "#733C14",
                        "#322115",
                        "#DC332A"]
        colorbins = np.arange(9) - 0.5 # '0=Invalid', '1=Clear', '2=Cloud',
                                        # '3=Tropo. aerosol', '4=Strato. aerosol',
                                        # '5=Surface', '6=Subsurface',
                                        # '7=Fully Attenuated'
        my_cmap = mpl.colors.ListedColormap(cmaplist)                            
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_type.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ %s}$' % VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Invalid', 'Clear', 'Cloud', 'Tropospheric\nAerosol',
                                'Stratospheric\nAerosol', 'Surface', 'Subsurface',
                                'Fully\nAttenuated']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=7, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM"
        self.save_fig(filename)

        # # Save figure (for test_colorbar)
        # self.fig_folder = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/test_colorbar/"
        # filename = f"VFM"
        # self.save_fig(filename)#, transparent=True, adjust=(0.02, 0.02, 0.98, 0.8))
        
        # Close figure
        plt.close(fig)
        
        
    def plot_vfm_ha(self, vfm_ha):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            vfm_ha = remove_edges(vfm_ha, EDGES_REMOVAL)

        # Figure style
        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])

        # my_cmap = takecmap('extviridis', cdark=0)
        my_cmap = cm.viridis_r
        palette = my_cmap(np.linspace(0, 256, 5).astype(int))
        palette[4] = np.array((0., 0., 0., 1.))
        palette = np.vstack((np.array((0.8, 0.8, 0.8, 1.)), palette))
        my_cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.arange(7) - 0.5 # '0=not applicable', '1=1/3 km', '2=1 km',
                                    # '3=5 km', '4=20 km', '5=80 km'
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_ha.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ (horizontal\ averaging)\ %s}$' %\
                VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Not applicable', '1/3 km', '1 km', '5 km',
                                '20 km', '80 km']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=4, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM_ha"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_vfm_phase(self, vfm_type, vfm_phase):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            vfm_type = remove_edges(vfm_type, EDGES_REMOVAL)
            vfm_phase = remove_edges(vfm_phase, EDGES_REMOVAL)

        # Put "not applicable" (-1) where no cloud
        vfm_phase = np.ma.array(vfm_phase, dtype='f')
        vfm_phase[vfm_type != 2] = -1

        # Figure style
        setstyle("ticks_nogrid")

        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])

        cmaplist = ['#C0FFA8',
                    "#FF0000",
                    "#FFFFFF",
                    "#0000FF",
                    "#A9A9A9"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(6) - 1.5 # '-1=Not applicable', '0=Unknown', '1=Ice',
                                    # '2=Water', '3=Oriented ice crystal',
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_phase.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ (cloud\ phase)\ %s}$' %\
                VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Not applicable', 'Unknown', 'Ice', 'Water',
                                'Oriented ice crystal']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=4, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM_cloud_phase"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)

    
    def plot_vfm_subtype(self, vfm_type, vfm_subtype):
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            vfm_type = remove_edges(vfm_type, EDGES_REMOVAL)
            vfm_subtype = remove_edges(vfm_subtype, EDGES_REMOVAL)
            
        # Keep only cloud subtypes: Put "not applicable" (-1) where no cloud
        vfm_cloud = np.ma.array(vfm_subtype, dtype='f')
        vfm_cloud[vfm_type != 2] = -1

        # Keep only tropo aerosol subtypes: Put "not applicable" (0) where no aerosol
        vfm_tropo_aerosol = np.ma.copy(vfm_subtype)
        vfm_tropo_aerosol[vfm_type != 3] = 0
        
        # Keep only strato aerosol subtypes: Put "not apllicable" (0) where no aerosol
        vfm_strato_aerosol = np.ma.copy(vfm_subtype)
        vfm_strato_aerosol[vfm_type != 4] = 0
        
        # Figure style
        setstyle("ticks_nogrid")
    
        
        # Plot cloud subtypes
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])

        cmaplist = ["#0026FF",
                    "#FF0000",
                    "#00DCFF",
                    "#AE5700",
                    "#FFA000",
                    "#FAFF00",
                    "#00FF6E",
                    "#C0C0C0",
                    "#000000"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(10) - 1.5 # '-1=Not applicable',
                                        # '0=Low overcast, transparent',
                                        # '1=Low overcast, opaque',
                                        # '2=Transition stratocumulus',
                                        # '3=Low, broken cumulus',
                                        # '4=Altocumulus (transparent)',
                                        # '5=Altostratus (opaque)',
                                        # '6=Cirrus (transparent)',
                                        # '7=Deep convective (opaque)'
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_cloud.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ (cloud\ subtype)\ %s}$' %\
                VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Not applicable', 'Low overcast, transparent',
                                'Low overcast, opaque', 'Transition stratocumulus',
                                'Low, broken cumulus', 'Altocumulus (transparent)',
                                'Altostratus (opaque)', 'Cirrus (transparent)',
                                'Deep convective (opaque)']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=4, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM_cloud_subtype"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


        # Plot tropo aerosol subtypes
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])
        
        cmaplist = ["#C6C6C6",
                    "#0000FF",
                    "#FAFF00",
                    "#FFB300",
                    "#00A600",
                    "#994C00",
                    "#000000",
                    "#008EC2",
                    "#F4F4F4",
                    "#8A8A8A",
                    "#515151"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(12) - 0.5 # '0=Not applicable', '1=Marine', '2=Dust',
                                        # '3=Polluted continental/Smoke',
                                        # '4=Clean continental', '5=Polluted dust',
                                        # '6=Elevated smoke', '7=Dusty marine',
                                        # '8=PSC aerosol', '9=Volcanic ash',
                                        # '10=Sulfate/Other'
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_tropo_aerosol.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ (tropospheric\ aerosol\ subtype)\ %s}$' %\
                VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Not applicable', 'Marine', 'Dust',
                                'Polluted continental/Smoke', 'Clean continental',
                                'Polluted dust', 'Elevated smoke', 'Dusty marine',
                                'PSC aerosol', 'Volcanic ash', 'Sulfate/Other']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=4, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM_tropo_aerosol_subtype"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)
        
        
        # Plot strato aerosol subtypes
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        
        ax0 = plt.subplot(gs0[0])

        cmaplist = ["#C6C6C6",
                    "#56B4E9",
                    "#E69F00",
                    "#DB4D3B",
                    "#3F3F3F"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(6) - 0.5 # '0=Not applicable', '1=PSC aerosol', '2=Volcanic Ash',
                                        # '3=Sulfate/Other',
                                        # '4=Elevated smoke'
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, vfm_strato_aerosol.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Vertical\ Feature\ Mask\ (stratospheric\ aerosol\ subtype)\ %s}$' %\
                VERSION_CAL_LID_L2, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, drawedges=True)
        # cbar.ax.set_yticklabels([' ']) # Delete colorbar number label
        cbar.ax.tick_params(which='both', right=False, labelright=False)
        for j, lab in enumerate(['Not applicable', 'PSC aerosol', 'Volcanic Ash',
                                'Sulfate/Other', 'Elevated smoke']):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) +
                                j/float(colorbins.size-1), lab, va='center',
                                fontsize=4, transform=ax1.transAxes)

        # Save figure
        filename = f"VFM_strato_aerosol_subtype"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)
        

    def plot_caliop_params(self, E, C, Ga, rms, nsf, wv, polar):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            E = remove_edges(E, EDGES_REMOVAL)
            C = remove_edges(C, EDGES_REMOVAL)
            Ga = remove_edges(Ga, EDGES_REMOVAL)
            rms = remove_edges(rms, EDGES_REMOVAL)
            nsf = remove_edges(nsf, EDGES_REMOVAL)
            
        # Parameters
        fig_w = cm2in(18) # cm
        fig_h = cm2in(20) # cm
        
        if wv==532:
            wv_txt = r'532\ nm\ '
        elif wv==1064:
            wv_txt = r'1064\ nm\ '
        else:
            wv_txt = ''

        if polar=='par':
            polar_txt_long = r'Parallel\ '
        elif polar=='per':
            polar_txt_long = r'Perpendicular\ '
        else:
            polar_txt = ''
            polar_txt_long = ''

        # Create figure
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs0 = gridspec.GridSpec(5, 1, hspace=0.6)

        # Laser energy
        ax = plt.subplot(gs0[0])
        plt.plot(self.pindex, E*1000.)
        plt.ylabel('Energy (mJ)')
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins,
                           flag_lat_lon_label=False, flag_dist=True, flag_dist_label=True)
        plt.title(r"$\mathbf{%sLaser\ Energy}$" % wv_txt)

        # Calibration constant
        ax = plt.subplot(gs0[1])
        plt.plot(self.pindex, C)
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins,
                           flag_lat_lon_label=False, flag_dist=True, flag_dist_label=False)
        plt.xlabel('')
        plt.title(r"$\mathbf{%sCalibration\ Constant}$" % wv_txt)

        # Gain
        ax = plt.subplot(gs0[2])
        plt.plot(self.pindex, Ga)
        plt.ylabel(r'Gain (V $\mathrm{V^{-1}}$)')
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins,
                           flag_lat_lon_label=False, flag_dist=True, flag_dist_label=False)
        plt.xlabel('')
        plt.title(r"$\mathbf{%s%sAmplifier\ Gain}$" % (wv_txt, polar_txt_long))

        # RMS baseline
        ax = plt.subplot(gs0[3])
        plt.plot(self.pindex, rms)
        plt.ylabel('RMS noise (digit. counts)')
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins,
                           flag_lat_lon_label=False, flag_dist=True, flag_dist_label=False)
        plt.xlabel('')
        plt.title(r"$\mathbf{%s%sRMS\ Baseline}$" % (wv_txt, polar_txt_long))

        # NSF
        ax = plt.subplot(gs0[4])
        plt.plot(self.pindex, nsf)
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins,
                           flag_lat_lon_label=True, flag_dist=True, flag_dist_label=False)
        plt.title(r"$\mathbf{%s%sNoise\ Scale\ Factor}$" % (wv_txt, polar_txt_long))

        # Save figure
        filename = f"param{wv:d}{polar}"
        self.save_fig(filename, adjust=(0.12, 0.06, 0.93, 0.92))
        
        # Close figure
        plt.close(fig)
        

    def plot_nb_bins_shift(self, nb_bins_shift):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            nb_bins_shift = remove_edges(nb_bins_shift, EDGES_REMOVAL)
            
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))

        # Plot
        ax = plt.subplot(111)
        plt.plot(self.pindex, nb_bins_shift)
        lat_lon_dist_xaxis(ax, self.lat, self.lon, self.pindex, self.pindexbins, flag_dist=False)
        plt.title(r"Number_Bins_Shift")

        # Save figure
        filename = "nb_bins_shift"
        self.save_fig(filename, adjust=(0.07, 0.18, 0.95, 0.9))

        # Close figure
        plt.close(fig)

    def plot_layer_feature(self, data_5km, data_1km, data_333m, prop, title):
        
        if EDGES_REMOVAL != 0: # to avoid error with "-0"
            data_5km = remove_edges(data_5km, EDGES_REMOVAL)
            data_1km = remove_edges(data_1km, EDGES_REMOVAL)
            data_333m = remove_edges(data_333m, EDGES_REMOVAL)
            
        # Overlay data of different resolution
        data = np.ma.copy(data_333m)
        data[data.mask] = data_1km[data.mask] # put 1 km only where no 333 m data
        data[data.mask] = data_5km[data.mask] # put 5 km only where no 1km or 333 m data

        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        
        if self.colorbar_position == 'right':
            gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        elif self.colorbar_position == 'bottom':
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
        

        # Colormap
        if (prop == 'AB532') | (prop == 'AB532par') | (prop == 'AB532per') |\
           (prop == 'AB1064'):
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey background
            clim_min = 1e-4
            clim_max = 1e-2
            my_cmap = takecmap('extviridis')
            extend = 'both'
            my_cmap.colorbar_extend = extend
            my_norm = LogNorm()
            cbar_label = r"$\beta'$ $(km^{-1}\ sr^{-1})$"
            cbar_edges = False
        elif prop == 'IAB532':
            mpl.rcParams["axes.facecolor"]='0.8' # dark grey background
            clim_min = 0
            clim_max = 0.03
            my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 0.031, 0.005)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = r"$\gamma'$ $(sr^{-1})$"
            cbar_edges = True
        elif prop == 'ACR':
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey background
            clim_min = 0
            clim_max = 1.6
            bounds = np.arange(0, 1.61, 0.1)
            # YlOrBr_r - Purples
            my_cmap1 = copy.copy(cm.YlOrBr_r)
            nb_colors = 11 # don't use first last (too light)
            colors1 = my_cmap1(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap2 = copy.copy(cm.Purples)
            nb_colors = 7
            colors2 = my_cmap2(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [[0/255., 0/255., 0/255., 1.]],
                            [colors1[0]],
                            [colors1[1]],
                            [colors1[2]],
                            [colors1[3]],
                            [colors1[4]],
                            [colors1[5]],
                            [colors1[6]],
                            [colors1[7]],
                            [colors1[8]],
                            [colors1[9]],
                            [colors2[0]],
                            [colors2[1]],
                            [colors2[2]],
                            [colors2[3]],
                            [colors2[4]],
                            [colors2[5]],
                            [colors2[6]]]
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = "Attenuated Color Ratio"
            cbar_edges = True
        elif prop == 'DR' or prop == 'DR1064':
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey backgroud
            clim_min = 0
            clim_max = 1
            my_cmap = takecmap('cubeh1_r', nb_colors=256, clight=1., cdark=0.15)
            # my_cmap = cm.YlGnBu_r
            # my_cmap = cm.RdPu_r
            # my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 0.50001, 0.05)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = "Depolarization Ratio"
            cbar_edges = True
        elif (prop == 'alt') or (prop == 'alt_base') or (prop == 'alt_top'):
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey backgroud
            clim_min = 0
            clim_max = 20
            my_cmap = takecmap('extviridis')
            bounds = np.arange(0, 20.1, 1)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = r"Altitude $(km)$"
            cbar_edges = True
        elif prop == 'temp':
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey backgroud
            clim_min = -50
            clim_max = 20
            bounds = np.arange(-50, 20.1, 10)
            # YlOrRd_r - Blues
            my_cmap1 = copy.copy(cm.YlOrRd_r)
            nb_colors = 5
            colors1 = my_cmap1(np.linspace(0, 255, nb_colors).astype(int))
            my_cmap2 = copy.copy(cm.Blues)
            nb_colors = 4
            colors2 = my_cmap2(np.linspace(0, 255, nb_colors).astype(int))
            colors = np.r_[ [colors1[0]],
                            [colors1[1]],
                            [colors1[2]],
                            [colors1[3]],
                            [colors1[4]],
                            [colors2[0]],
                            [colors2[1]],
                            [colors2[2]],
                            [colors2[3]]][::-1]
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = r"Temperature $(°C)$"
            cbar_edges = True
        elif prop == 'lat':
            mpl.rcParams["axes.facecolor"]='0.2' # dark grey backgroud
            clim_min = np.min(self.lat)
            clim_max = np.max(self.lat)
            my_cmap = takecmap('extviridis')
            bounds = np.linspace(np.min(self.lat), np.max(self.lat), 10)
            nb_colors = len(bounds) + 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = 'both'
            my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend=extend)
            cbar_label = r"Latitude $($°$)$"
            cbar_edges = True
        elif prop == 'cad_score':
            mpl.rcParams["axes.facecolor"]='0.8' # light grey backgroud
            clim_min = -100
            clim_max = 100
            my_cmap = cm.RdBu
            bounds = np.arange(-100, 101, 20)
            nb_colors = len(bounds) - 1
            colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = None
            my_cmap, my_norm = from_levels_and_colors(bounds, colors)
            cbar_label = r"CAD Score"
            cbar_edges = True
        elif prop == 'surface_type':
            mpl.rcParams["axes.facecolor"]='0.8' # light grey backgroud
            clim_min = 1
            clim_max = 3
            my_cmap = cm.RdBu_r
            # bounds = np.arange(-100, 101, 10)
            # nb_colors = len(bounds) - 1
            # colors = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
            extend = None
            # my_cmap, my_norm = from_levels_and_colors(bounds, colors)
            my_norm = None
            cbar_label = "1=water; 2=desert;\n3=land not desert"
            cbar_edges = False
            
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, data.T, cmap=my_cmap, norm=my_norm,
                            rasterized=True)
        plt.clim(clim_min, clim_max)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title('%s' % title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        mpl.rcParams["axes.facecolor"]='1.' # white backgroud
        ax1 = plt.subplot(gs0[1])
        if self.colorbar_position == 'right':
            cbar_orientation = 'vertical'
        elif self.colorbar_position == 'bottom':
            cbar_orientation = 'horizontal'
        cbar = plt.colorbar(pc, cax=ax1, orientation=cbar_orientation, extend=extend, drawedges=cbar_edges)
        # if cbar_edges: # comment when new version of matplotlib fixed bug of drawedges
        #     plt.axhline(max(bounds), color='k', linewidth=1.)
        #     plt.axhline(min(bounds), color='k', linewidth=1.)
        cbar.set_label(label=cbar_label, labelpad=5)
        if (prop == 'AB532') | (prop == 'AB532par') | (prop == 'AB532per') |\
           (prop == 'AB1064'):
            cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
            minor_locators = np.concatenate((np.arange(2,10)*1e-4,
                                             np.arange(2,10)*1e-3,
                                             np.arange(2,10)*1e-2,
                                             np.arange(2,10)*1e-1))
            cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
        elif prop == 'ACR':
            cbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        elif prop == 'DR':
            cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
            cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        elif prop == 'IAB532':
            cbar.ax.yaxis.set_major_locator(MultipleLocator(0.01))
            cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        elif prop == 'cad_score':
            cbar.ax.yaxis.set_major_locator(MultipleLocator(20))
            cbar.ax.yaxis.set_minor_locator(MultipleLocator(20))
        elif prop == 'temp':
            cbar.ax.yaxis.set_major_locator(MultipleLocator(10))
            cbar.ax.yaxis.set_minor_locator(MultipleLocator(10))
            
        # Save figure
        filename = f"layer_mean_{prop}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)

        
if __name__ == '__main__':
    tic_main_program = print_time()
    args = parse_arguments()
    globals().update(load_yaml_configuration(
        args.configuration,
        {
            "FOLDER_PATH", "VERSION_CAL_LID_L1", "VERSION_CAL_LID_L2",
            "TYPE_CAL_LID_L1", "TYPE_CAL_LID_L2", "REGULAR_GRIDS",
            "APPLY_DECONVOLUTION", "EDGES_REMOVAL", "INVERT_XAXIS", "YMIN",
            "YMAX", "COLORMAP", "PLOT_ASPECT_RATIO", "FIGURES_PATH",
            "FIGURES_FILETYPE", "PLOT_MAP", "PLOT_AB_532", "PLOT_AB_532_HIST",
            "PLOT_AB_532_PAR", "PLOT_AB_532_PAR_HIST", "PLOT_AB_532_PER",
            "PLOT_AB_532_PER_HIST", "PLOT_AB_1064", "PLOT_AB_1064_HIST",
            "PLOT_ACR", "PLOT_DR", "PLOT_AB_MOL_532", "PLOT_AB_MOL_532_PAR",
            "PLOT_AB_MOL_532_PER", "PLOT_AB_MOL_1064", "PLOT_ASR_532_STD",
            "PLOT_ASR_532_PAR_STD", "PLOT_ASR_532_PER_STD", "PLOT_ASR_1064_STD",
            "PLOT_ASR_532", "PLOT_ASR_532_HIST", "PLOT_ASR_532_PAR",
            "PLOT_ASR_532_PAR_HIST", "PLOT_ASR_532_PER", "PLOT_ASR_532_PER_HIST",
            "PLOT_ASR_1064", "PLOT_ASR_1064_HIST", "PLOT_ASR_532_ABOVE_STD",
            "PLOT_ASR_532_PAR_ABOVE_STD", "PLOT_ASR_532_PER_ABOVE_STD",
            "PLOT_ASR_1064_ABOVE_STD", "PLOT_PARAMS_532_PAR", "PLOT_PARAMS_532_PER",
            "PLOT_PARAMS_1064", "PLOT_NB_BINS_SHIFT", "PLOT_VFM_FEATURE_TYPE",
            "PLOT_VFM_HORIZONTAL_AVERAGING", "PLOT_VFM_PHASE", "PLOT_VFM_SUBTYPE",
            "PLOT_FEATURE_DR", "PLOT_FEATURE_DR1064", "PLOT_FEATURE_ACR",
            "PLOT_FEATURE_IAB532", "PLOT_FEATURE_TEMP", "PLOT_EXT532", "PLOT_EXT1064",
        },
    ))
    
    
    # ***************************
    # *** Load CALIOP L1 data ***
    print("\n*****Load CALIOP L1 data...*****")

    # Load data with various regular grid sizes
    cal_l1 = {}
    for grid in REGULAR_GRIDS:
        cal_l1[grid] = CALIOPRegularGridReader(product='L1',
                                               version=VERSION_CAL_LID_L1,
                                               data_type=TYPE_CAL_LID_L1,
                                               granule_date=GRANULE_DATE,
                                               grid=grid,
                                               slice_start=SLICE_START,
                                               slice_end=SLICE_END,
                                               slice_start_end_type=SLICE_START_END_TYPE,
                                               folderpath=FOLDER_PATH,
                                               deconvolution=APPLY_DECONVOLUTION)
    
    # Print filepaths of loading files
    print(f"\tCAL_LID_L1: {cal_l1[REGULAR_GRIDS[0]].filepath}")

    # Print lat/lon of min and max prof indices
    cal_l1[REGULAR_GRIDS[0]].print_lat_lon_min_max()

    # Load L1 parameters
    cal_l1_keys = get_cal_l1_keys()
    data_dict_cal_lid_l1 = {}
    for grid in REGULAR_GRIDS:
        print(f"\t{grid}")
        data_dict_cal_lid_l1[grid] = {}
        for key in cal_l1_keys:
            data_dict_cal_lid_l1[grid][key] = cal_l1[grid].get_data(key)
    
    # *******************************
    # *** Load CALIOP L2 VFM data ***
    print("\n*****Load CALIOP L2 VFM data...*****")
    
    # Load data with various regular grid sizes
    cal_l2_vfm = CALIOPRegularGridReader(product='L2_VFM',
                                         version=VERSION_CAL_LID_L2,
                                         data_type=TYPE_CAL_LID_L2,
                                         granule_date=GRANULE_DATE,
                                         slice_start=SLICE_START,
                                         slice_end=SLICE_END,
                                         slice_start_end_type=SLICE_START_END_TYPE,
                                         version_l1=VERSION_CAL_LID_L1,
                                         data_type_l1=TYPE_CAL_LID_L1)
    
    # Print filepaths of loading files
    print(f"\tCAL_LID_L2_VFM: {cal_l2_vfm.filepath}")
    
    # Print lat/lon of min and max prof indices
    cal_l2_vfm.print_lat_lon_min_max()
        
    # Load VFM parameters
    cal_l2_vfm_keys = [
        "Latitude",
        "Longitude",
        "Lidar_Data_Altitudes",
        "Feature_Classification_Flags"
    ]
    data_dict_cal_lid_l2_vfm = {}
    for key in cal_l2_vfm_keys:
        data_dict_cal_lid_l2_vfm[key] = cal_l2_vfm.get_data(key)


    # ****************************************
    # *** Take features of interest in VFM ***
    print("\n*****Take features of interest in VFM...*****")
    vfm = np.copy(data_dict_cal_lid_l2_vfm["Feature_Classification_Flags"])
    vfm[vfm==FILL_VALUE_FLOAT] = 0
    vfm = vfm.astype('uint16')
    # The VFM (5 km blocks) misses the last L1 profiles of the granule: pad with 0 (invalid)
    nb_missing_profiles = cal_l1[REGULAR_GRIDS[0]].nb_profiles - vfm.shape[0]
    if nb_missing_profiles > 0:
        vfm = np.pad(vfm, ((0, nb_missing_profiles), (0, 0)))
    vfm_type = np.bitwise_and(vfm, 7) # Bits 1-16 & 111 = Bits 1-3
    vfm_ha = vfm >> 13 # Bits 14-16 >> Bits 1-3
    vfm_phase = np.bitwise_and(vfm >> 5, 3) # Bits 6-16 >> Bits 1-11 & 11 = Bits 6-7
    vfm_subtype = np.bitwise_and(vfm >> 9, 7) # Bits 10-16 >> Bits 1-7 & 111 = Bits 10-12


    if PLOT_FEATURE_DR or PLOT_FEATURE_DR1064 or PLOT_FEATURE_ACR or PLOT_FEATURE_IAB532 or PLOT_FEATURE_TEMP:
        # *********************************
        # *** Load CALIOP L2 layer data ***
        print("\n*****Load CALIOP L2 layer data...*****")
        cal_l2_5km = CALIOPRegularGridReader(product='L2_05kmMLay',
                                             version=VERSION_CAL_LID_L2,
                                             data_type=TYPE_CAL_LID_L2,
                                             granule_date=GRANULE_DATE,
                                             grid='333mx30m',
                                             slice_start=SLICE_START,
                                             slice_end=SLICE_END,
                                             slice_start_end_type=SLICE_START_END_TYPE,
                                             version_l1=VERSION_CAL_LID_L1,
                                             data_type_l1=TYPE_CAL_LID_L1)
        
        cal_l2_layer_keys = ["Integrated_Volume_Depolarization_Ratio",
                             "Integrated_Attenuated_Total_Color_Ratio",
                             "Integrated_Attenuated_Backscatter_532",
                             "Layer_Centroid_Temperature"]
        data_dict_cal_lid_l2_5km = {}
        for key in cal_l2_layer_keys:
            data_dict_cal_lid_l2_5km[key] = cal_l2_5km.get_data(key)
        
        # Print filepaths of loading files
        print(f"\tCAL_LID_L2: {cal_l2_5km.filepath}")
        
        # Print lat/lon of min and max prof indices
        cal_l2_5km.print_lat_lon_min_max()
        
    if PLOT_FEATURE_DR or PLOT_FEATURE_DR1064 or PLOT_FEATURE_ACR or PLOT_FEATURE_IAB532 or PLOT_FEATURE_TEMP:
        cal_l2_1km = CALIOPRegularGridReader(product='L2_01kmCLay',
                                             version=VERSION_CAL_LID_L2,
                                             data_type=TYPE_CAL_LID_L2,
                                             granule_date=GRANULE_DATE,
                                             grid='333mx30m',
                                             slice_start=SLICE_START,
                                             slice_end=SLICE_END,
                                             slice_start_end_type=SLICE_START_END_TYPE,
                                             version_l1=VERSION_CAL_LID_L1,
                                             data_type_l1=TYPE_CAL_LID_L1)
        
        cal_l2_layer_keys = ["Integrated_Volume_Depolarization_Ratio",
                             "Integrated_Attenuated_Total_Color_Ratio",
                             "Integrated_Attenuated_Backscatter_532",
                             "Layer_Centroid_Temperature"]
        data_dict_cal_lid_l2_1km = {}
        for key in cal_l2_layer_keys:
            data_dict_cal_lid_l2_1km[key] = cal_l2_1km.get_data(key)

        # Print filepaths of loading files
        print(f"\tCAL_LID_L2: {cal_l2_1km.filepath}")
        
        # Print lat/lon of min and max prof indices
        cal_l2_1km.print_lat_lon_min_max()
        
    if PLOT_FEATURE_DR or PLOT_FEATURE_DR1064 or PLOT_FEATURE_ACR or PLOT_FEATURE_IAB532 or PLOT_FEATURE_TEMP:
        cal_l2_333m = CALIOPRegularGridReader(product='L2_333mMLay',
                                              version=VERSION_CAL_LID_L2,
                                              data_type=TYPE_CAL_LID_L2,
                                              granule_date=GRANULE_DATE,
                                              grid='333mx30m',
                                              slice_start=SLICE_START,
                                              slice_end=SLICE_END,
                                              slice_start_end_type=SLICE_START_END_TYPE)
    
        # Print filepaths of loading files
        print(f"\tCAL_LID_L2: {cal_l2_333m.filepath}")
        
        # Print lat/lon of min and max prof indices
        cal_l2_333m.print_lat_lon_min_max()
            
        cal_l2_layer_keys = ["Integrated_Volume_Depolarization_Ratio",
                             "Integrated_Attenuated_Total_Color_Ratio",
                             "Integrated_Attenuated_Backscatter_532",
                             "Layer_Centroid_Temperature"]
        data_dict_cal_lid_l2_333m = {}
        for key in cal_l2_layer_keys:
            data_dict_cal_lid_l2_333m[key] = cal_l2_333m.get_data(key)
    
    
    if PLOT_EXT532 or PLOT_EXT1064:
        # *********************************
        # *** Load CALIOP L2 layer data ***
        print("\n*****Load CALIOP L2 layer data...*****")

        # slice_start and slice_end from L1 because no ssLongitude in L2_Pro file
        slice_start_pro = cal_l1[REGULAR_GRIDS[0]].prof_min
        slice_end_pro = cal_l1[REGULAR_GRIDS[0]].prof_max
        slice_start_end_pro = 'profindex'

        cal_l2_5km_apro = CALIOPRegularGridReader(product='L2_05kmAPro',
                                                  version=VERSION_CAL_LID_L2,
                                                  data_type=TYPE_CAL_LID_L2,
                                                  granule_date=GRANULE_DATE,
                                                  grid='333mx30m',
                                                  slice_start=slice_start_pro,
                                                  slice_end=slice_end_pro,
                                                  slice_start_end_type=slice_start_end_pro,
                                                  version_l1=VERSION_CAL_LID_L1,
                                                  data_type_l1=TYPE_CAL_LID_L1)

        cal_l2_5km_cpro = CALIOPRegularGridReader(product='L2_05kmCPro',
                                                  version=VERSION_CAL_LID_L2,
                                                  data_type=TYPE_CAL_LID_L2,
                                                  granule_date=GRANULE_DATE,
                                                  grid='333mx30m',
                                                  slice_start=slice_start_pro,
                                                  slice_end=slice_end_pro,
                                                  slice_start_end_type=slice_start_end_pro,
                                                  version_l1=VERSION_CAL_LID_L1,
                                                  data_type_l1=TYPE_CAL_LID_L1)

        # cal_l2_pro_keys = ["Extinction_Coefficient_532",
        #                    "Extinction_Coefficient_1064",
        #                    "Lidar_Data_Altitudes_init"]
        cal_l2_pro_keys = ["Lidar_Data_Altitudes_init",]
        data_dict_cal_lid_l2_apro = {}
        data_dict_cal_lid_l2_cpro = {}
        for key in cal_l2_pro_keys:
            data_dict_cal_lid_l2_apro[key] = cal_l2_5km_apro.get_data(key)
            data_dict_cal_lid_l2_cpro[key] = cal_l2_5km_cpro.get_data(key)
        

    # ************
    # *** Plot ***
    print("\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    if PLOT_ASPECT_RATIO == "browse_colorbar_right":
        plot_fig.fig_w = cm2in(16) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.87
        plot_fig.adj_top = 0.81
        plot_fig.axes_title_pad = 1.15
        plot_fig.clabelpad = 40
        plot_fig.colorbar_position = 'right'
    elif PLOT_ASPECT_RATIO == "browse_colorbar_bottom":
        plot_fig.fig_w = cm2in(13.5) # cm
        plot_fig.fig_h = cm2in(10) # cm
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.94
        plot_fig.adj_top = 0.81
        plot_fig.axes_title_pad = 1.15
        plot_fig.clabelpad = 40
        plot_fig.colorbar_position = 'bottom'
        plot_fig.axes_labelsize = 7
        plot_fig.xtick_labelsize = 7
        plot_fig.ytick_labelsize = 7
    elif PLOT_ASPECT_RATIO == "spec":
        plot_fig.fig_w = cm2in(12) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.1
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.83
        plot_fig.adj_top = 0.81
        plot_fig.axes_title_pad = 1.15
        plot_fig.clabelpad = 40
        plot_fig.y_major_locator = 1
        plot_fig.y_minor_locator = 0.2
    plot_fig.filetype = FIGURES_FILETYPE
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, cal_l1[REGULAR_GRIDS[0]].lon_min, cal_l1[REGULAR_GRIDS[0]].lon_max)
    plot_fig.set_head_filename(GRANULE_DATE, cal_l1[REGULAR_GRIDS[0]].lon_min, cal_l1[REGULAR_GRIDS[0]].lon_max)
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_coordinates(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Latitude'],
                             data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Longitude'],
                             data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Lidar_Data_Altitudes'])

    # Map
    if PLOT_MAP:
        cal_l1_whole_granule = CALIOPRegularGridReader(product='L1',
                                                       version=VERSION_CAL_LID_L1,
                                                       data_type=TYPE_CAL_LID_L1,
                                                       granule_date=GRANULE_DATE,
                                                       grid=REGULAR_GRIDS[0])
        lat_granule = cal_l1_whole_granule.get_data('Latitude')
        lon_granule = cal_l1_whole_granule.get_data('Longitude')
        plot_fig.plot_map(lat_granule, lon_granule, 
                          data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Profile_UTC_Time'])

    # AB 532
    if PLOT_AB_532:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_attenuated_backscatter(data_dict_cal_lid_l1[grid]['Total_Attenuated_Backscatter_532'],
                                                 532, '', grid)
    if PLOT_AB_532_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Total_Attenuated_Backscatter_532'],
                            532, '', 'AB', -0.01, 0.1)

    # AB 532 PARALLEL
    if PLOT_AB_532_PAR:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_attenuated_backscatter(data_dict_cal_lid_l1[grid]['Parallel_Attenuated_Backscatter_532'],
                                                 532, 'par', grid)
    if PLOT_AB_532_PAR_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Parallel_Attenuated_Backscatter_532'],
                            532, 'par', 'AB', -0.01, 0.1)
    
    # AB 532 PERPENDICULAR
    if PLOT_AB_532_PER:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_attenuated_backscatter(data_dict_cal_lid_l1[grid]['Perpendicular_Attenuated_Backscatter_532'],
                                                 532, 'per', grid)
    if PLOT_AB_532_PER_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Perpendicular_Attenuated_Backscatter_532'],
                            532, 'per', 'AB', -0.01, 0.1)
    
    # AB 1064
    if PLOT_AB_1064:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_attenuated_backscatter(data_dict_cal_lid_l1[grid]['Attenuated_Backscatter_1064'],
                                                 1064, '', grid)
    if PLOT_AB_1064_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Backscatter_1064'],
                            1064, '', 'AB', -0.01, 0.1)
        
    # COLOR RATIO
    # use par+per and not tot in order to put fill value in CR if fill value
    # in one of par or per
    if PLOT_ACR:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_attenuated_color_ratio(data_dict_cal_lid_l1[grid]['Attenuated_Backscatter_1064']/\
                                                 data_dict_cal_lid_l1[grid]['Total_Attenuated_Backscatter_532'],
                                                 grid)
            
    # # DEPOLARIZATION
    if PLOT_DR:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_depolarization_ratio(data_dict_cal_lid_l1[grid]['Perpendicular_Attenuated_Backscatter_532']/\
                                               data_dict_cal_lid_l1[grid]['Parallel_Attenuated_Backscatter_532'],
                                               grid)

    # ESTIMATED MOLECULAR AB 532
    if PLOT_AB_MOL_532:
        plot_fig.plot_attenuated_backscatter_mol(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Molecular_Total_Attenuated_Backscatter_532'],
                                                 532, '', REGULAR_GRIDS[0])

    # ESTIMATED MOLECULAR AB 532 PARALLEL
    if PLOT_AB_MOL_532_PAR:
        plot_fig.plot_attenuated_backscatter_mol(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Molecular_Parallel_Attenuated_Backscatter_532'],
                                                 532, 'par', REGULAR_GRIDS[0])

    # ESTIMATED MOLECULAR AB 532 PERPENDICULAR
    if PLOT_AB_MOL_532_PER:
        plot_fig.plot_attenuated_backscatter_mol(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Molecular_Perpendicular_Attenuated_Backscatter_532'],
                                                 532, 'per', REGULAR_GRIDS[0])

    # ESTIMATED MOLECULAR AB 1064
    if PLOT_AB_MOL_1064:
        plot_fig.plot_attenuated_backscatter_mol(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Molecular_Attenuated_Backscatter_1064'],
                                                 1064, '', REGULAR_GRIDS[0])

    # ASR 532 UNCERTAINTY STANDARD DEVIATION
    if PLOT_ASR_532_STD:
        plot_fig.plot_asr_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel'] +\
                                data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular'],
                                532, '')

    # ASR 532 PARALLEL UNCERTAINTY STANDARD DEVIATION
    if PLOT_ASR_532_PAR_STD:
        plot_fig.plot_asr_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel'],
                                532, 'par')

    # ASR 532 PERPENDICULAR UNCERTAINTY STANDARD DEVIATION
    if PLOT_ASR_532_PER_STD:
        plot_fig.plot_asr_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular'],
                                532, 'per')

    # ASR 1064 UNCERTAINTY STANDARD DEVIATION
    if PLOT_ASR_1064_STD:
        plot_fig.plot_asr_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064'],
                                1064, '')

    # ASR 532
    if PLOT_ASR_532:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_asr(data_dict_cal_lid_l1[grid]["Total_Attenuated_Backscatter_532"]/\
                              data_dict_cal_lid_l1[grid]["Molecular_Total_Attenuated_Backscatter_532"],
                              532, '', grid)
    if PLOT_ASR_532_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Total_Attenuated_Backscatter_532"]/
                            data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Total_Attenuated_Backscatter_532"],
                            532, '', 'ASR', -5, 5)

    # # ASR 532 PARALLEL
    if PLOT_ASR_532_PAR:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_asr(data_dict_cal_lid_l1[grid]["Parallel_Attenuated_Backscatter_532"]/\
                              data_dict_cal_lid_l1[grid]["Molecular_Parallel_Attenuated_Backscatter_532"],
                              532, 'par', grid)
    if PLOT_ASR_532_PAR_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Parallel_Attenuated_Backscatter_532"]/\
                            data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Parallel_Attenuated_Backscatter_532"],
                            532, 'par', 'ASR', -5, 5)

    # # ASR 532 PERPENDICULAR
    if PLOT_ASR_532_PER:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_asr(data_dict_cal_lid_l1[grid]["Perpendicular_Attenuated_Backscatter_532"]/\
                              data_dict_cal_lid_l1[grid]["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                              532, 'per', grid)
    if PLOT_ASR_532_PER_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Perpendicular_Attenuated_Backscatter_532"]/\
                            data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                            532, 'per', 'ASR', -100, 100)

    # ASR 1064
    if PLOT_ASR_1064:
        for grid in REGULAR_GRIDS:
            plot_fig.plot_asr(data_dict_cal_lid_l1[grid]["Attenuated_Backscatter_1064"]/\
                              data_dict_cal_lid_l1[grid]["Molecular_Attenuated_Backscatter_1064"],
                              1064, '', grid)
    if PLOT_ASR_1064_HIST:
        plot_fig.plot_histo(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Attenuated_Backscatter_1064"]/\
                            data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Attenuated_Backscatter_1064"],
                            1064, '', 'ASR', -500, 500)

    # ASR ABOVE ASR SIGMA 532
    if PLOT_ASR_532_ABOVE_STD:
        plot_fig.plot_asr_above_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Total_Attenuated_Backscatter_532"]/\
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Total_Attenuated_Backscatter_532"],
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel'] +\
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular'],
                                       532, '')

    # ASR ABOVE ASR SIGMA 532 PARALLEL
    if PLOT_ASR_532_PAR_ABOVE_STD:
        plot_fig.plot_asr_above_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Parallel_Attenuated_Backscatter_532"]/\
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Parallel_Attenuated_Backscatter_532"],
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel'],
                                      532, 'par')

    # ASR ABOVE ASR SIGMA 532 PERPENDICULAR
    if PLOT_ASR_532_PER_ABOVE_STD:
        plot_fig.plot_asr_above_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Perpendicular_Attenuated_Backscatter_532"]/\
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular'],
                                      532, 'per')

    # ASR ABOVE ASR SIGMA 1064
    if PLOT_ASR_1064_ABOVE_STD:
        plot_fig.plot_asr_above_sigma(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Attenuated_Backscatter_1064"]/\
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]["Molecular_Attenuated_Backscatter_1064"],
                                      data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064'],
                                      1064, '')

    # CALIOP VFM Feature Type
    if PLOT_VFM_FEATURE_TYPE:
        plot_fig.plot_vfm_type(vfm_type)

    # CALIOP VFM horizontal averaging
    if PLOT_VFM_HORIZONTAL_AVERAGING:
        plot_fig.plot_vfm_ha(vfm_ha)

    # CALIOP VFM Ice/Water Phase
    if PLOT_VFM_PHASE:
        plot_fig.plot_vfm_phase(vfm_type, vfm_phase)

    # CALIOP VFM Feature Sub-type
    if PLOT_VFM_SUBTYPE:
        plot_fig.plot_vfm_subtype(vfm_type, vfm_subtype)
    
    # Lidar parameters 532 par
    if PLOT_PARAMS_532_PAR:
        plot_fig.plot_caliop_params(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Laser_Energy_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Calibration_Constant_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Parallel_Amplifier_Gain_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Parallel_RMS_Baseline_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Noise_Scale_Factor_532_Parallel'],
                                    532, 'par')
 
    # Lidar parameters 532 per
    if PLOT_PARAMS_532_PER:
        plot_fig.plot_caliop_params(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Laser_Energy_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Calibration_Constant_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Perpendicular_Amplifier_Gain_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Perpendicular_RMS_Baseline_532'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Noise_Scale_Factor_532_Perpendicular'],
                                    532, 'per')

    # Lidar parameters 1064
    if PLOT_PARAMS_1064:
        plot_fig.plot_caliop_params(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Laser_Energy_1064'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Calibration_Constant_1064'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Amplifier_Gain_1064'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['RMS_Baseline_1064'],
                                    data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Noise_Scale_Factor_1064'],
                                    1064, '')
   
    # Number bins shift
    if PLOT_NB_BINS_SHIFT:
        plot_fig.plot_nb_bins_shift(data_dict_cal_lid_l1[REGULAR_GRIDS[0]]['Number_Bins_Shift'])
    
    # Mean layer DR
    if PLOT_FEATURE_DR:
        plot_fig.plot_layer_feature(data_dict_cal_lid_l2_5km["Integrated_Volume_Depolarization_Ratio"],
                                    data_dict_cal_lid_l2_1km["Integrated_Volume_Depolarization_Ratio"],
                                    data_dict_cal_lid_l2_333m["Integrated_Volume_Depolarization_Ratio"],
                                    "DR",
                                    "Layer Mean Depolarization Ratio "
                                    r"$\delta_v=\langle\beta'_{532,\perp}\rangle/\langle\beta'_{532,\parallel}\rangle$")
    
    # Mean layer ACR
    if PLOT_FEATURE_ACR:
        plot_fig.plot_layer_feature(data_dict_cal_lid_l2_5km["Integrated_Attenuated_Total_Color_Ratio"],
                                    data_dict_cal_lid_l2_1km["Integrated_Attenuated_Total_Color_Ratio"],
                                    data_dict_cal_lid_l2_333m["Integrated_Attenuated_Total_Color_Ratio"],
                                    "ACR",
                                    "Layer Mean Attenuated Color Ratio "
                                    r"$\chi'=\langle\beta'_{1064}\rangle/\langle\beta'_{532}\rangle$")
       
    # Mean layer IAB532
    if PLOT_FEATURE_IAB532:
        plot_fig.plot_layer_feature(data_dict_cal_lid_l2_5km["Integrated_Attenuated_Backscatter_532"],
                                    data_dict_cal_lid_l2_1km["Integrated_Attenuated_Backscatter_532"],
                                    data_dict_cal_lid_l2_333m["Integrated_Attenuated_Backscatter_532"],
                                    "IAB532",
                                    "Layer Mean 532 nm Integrated Attenuated Backscatter "
                                    r"$\langle\gamma'_{532}\rangle$")
      
    # Mean layer centroid temperature
    if PLOT_FEATURE_TEMP:
        plot_fig.plot_layer_feature(data_dict_cal_lid_l2_5km["Layer_Centroid_Temperature"],
                                    data_dict_cal_lid_l2_1km["Layer_Centroid_Temperature"],
                                    data_dict_cal_lid_l2_333m["Layer_Centroid_Temperature"],
                                    "temp",
                                    r"Layer Centroid Temperature $\langle T\rangle$ ")

        
    print_time(tic_main_program)
