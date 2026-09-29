#!/usr/bin/env python
# coding: utf8

import sys
import os
import argparse
from collections.abc import Mapping
from pathlib import Path

from datetime import datetime
import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import seaborn as sns
import cmocean
import yaml

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "my_modules"))

from standard_outputs import print_time
from readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from paths import split_granule_date
from figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges


FILL_VALUE_FLOAT = -9999.0


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
    unknown_case_keys = sorted(case.keys() - required_case_keys - {"name", "file_section"})
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

    # A null limit means the first/last profile of the file read.
    slice_start = None if case["start"] is None else float(case["start"])
    slice_end = None if case["end"] is None else float(case["end"])
    if case["mode"] == "profindex":
        if any(limit is not None and not limit.is_integer()
               for limit in (slice_start, slice_end)):
            raise ValueError("Profile-index limits must be integers")
        slice_start = None if slice_start is None else int(slice_start)
        slice_end = None if slice_end is None else int(slice_end)

    # Optional filename suffix of a 2D-McDA section file (e.g. _lon_62.70_61.30);
    # empty to read the complete granule file.
    file_section = str(case.get("file_section") or "").strip()
    if file_section and not file_section.startswith("_"):
        file_section = f"_{file_section}"

    flattened.update(
        GRANULE_DATE=str(case["granule"]),
        GRANULE_SECTION=file_section,
        SLICE_START_END_TYPE=str(case["mode"]),
        SLICE_START=slice_start,
        SLICE_END=slice_end,
        CASE_STUDY_NAME=case.get("name"),
    )
    for key in ("INDATA_FOLDER", "FIGURES_PATH"):
        value = flattened.get(key)
        if value and not Path(value).is_absolute():
            flattened[key] = str((PROJECT_ROOT / value).resolve())
    return flattened


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot 2D-McDA masks")
    parser.add_argument(
        "configuration",
        nargs="?",
        default=Path(__file__).with_name(
            f"{Path(__file__).stem}_single_granule.yaml"),
        help="Complete single-granule YAML configuration.",
    )
    return parser.parse_args()


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.14

    def set_max_detect_level(self, max_detect_level):
        self.max_detect_level = max_detect_level
        
        
    def plot_mask(self, step, mask, channel):
        """Plot mask for each step. If only final mask, put step = None"""

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No detection",] +\
                  ["Detection level %d" % i for i in np.arange(self.max_detect_level)+1] +\
                  ["Low confidence small strips",
                   "Almost fully attenuated",
                   "Fully attenuated",
                   "Likely artifact",
                   "Surface"]

        if step:
            clabels = clabels + ["Potential detection",]

        # Colormap
        nb_colors = self.max_detect_level
        palette = sns.cubehelix_palette(nb_colors, start=2, rot=1, hue=1., gamma=1., light=0.8,
                                        dark=0.2, reverse=True)
        palette.insert(0, [1.0, 1.0, 1.0]) # 0 = Nothing
        palette.append([0.7, 0.0, 0.0])  # 250 = Low confidence small strips
        palette.append([0.8, 0.0, 0.0])  # 251 = Almost fully attenuated
        palette.append([1.0, 0.0, 0.0])  # 252 = Fully attenuated
        palette.append([0.8, 0.8, 0.8])  # 253 = Likely artifact
        palette.append([0.5, 0.0, 0.0])  # 254 = Surface
        if step:
            palette.append([1.0, 0.5, 0.0])  # 255 = Maybe
            last_flag_value = 255
        else:
            last_flag_value = 254
        my_cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.append(np.arange(self.max_detect_level + 1),
                              np.arange(250, last_flag_value + 2)) - 0.5
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
        if step:
            plt.title(r'$\mathbf{Detection\ feature\ (step\ \#%d)\ %s}$' % (step, VERSION_2D_MCDA), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        else:
            if channel == '532_par':
                title = r"532\ nm\ parallel\ detection\ feature\ mask"
            elif channel == '532_per':
                title = r"532\ nm\ perpendicular\ detection\ feature\ mask"
            elif channel == '1064':
                title = r"1064\ nm\ detection\ feature\ mask"
            else:
                raise ValueError(f"Unknown channel = {channel}")
            plt.title(r'$\mathbf{%s\ %s}$' % (title, VERSION_2D_MCDA), fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 4
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        cbar.set_label("Level of detection", labelpad=80)
       
        # Save figure
        if step:
            filename = f"{channel}_step{step:d}"
        else:
            filename = f"mask_{channel}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_atsr(self, step, atsr, channel):
        # sourcery skip: merge-comparisons, merge-duplicate-blocks, remove-redundant-if
        
        # Mask where fill_value
        atsr = np.ma.masked_where(atsr == FILL_VALUE_FLOAT, atsr)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            atsr = remove_edges(atsr, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor((0.75, 0.75, 0.75))
        my_cmap = takecmap("extviridis_r")
        # Put negative values to 1e-9 so they don't appear transparent
        atsr[(atsr<0) & ~atsr.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, atsr.T, cmap=my_cmap, norm=LogNorm())
        if channel == '532_par':
            vmax = 1e1
        elif channel == '532_per':
            vmax = 1e3
        elif channel == '1064':
            vmax = 1e3
        else:
            raise ValueError(f"Unknown channel = {channel}")
        plt.clim(1e-1, vmax)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Detection\ feature\ (step\ \#%d)\ %s}$' % (step, VERSION_2D_MCDA), fontsize=self.axes_titlesize, y=self.axes_title_pad)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both')
        cbar.set_label(label="$R'$",)
        
        ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        
        # Save figure
        filename = f"{channel}_step{step:d}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)
    
    
    def plot_steps(self, mask, atsr, channel):
    
        # Get number of steps
        nb_steps = mask.shape[0]
    
        for step in np.arange(nb_steps):
    
            # Determine if it's a mask or atsr step
            if not np.all(mask[step, :, :] == 0):
                self.plot_mask(step, mask[step, :, :], channel)
            if not atsr[step, :, :].mask.all():
                self.plot_atsr(step, atsr[step, :, :], channel)


    def plot_twoway_transmittance(self, twoway_transmittance, channel):
        # Mask where fill_value
        twoway_transmittance = np.ma.masked_where(twoway_transmittance == FILL_VALUE_FLOAT, twoway_transmittance)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            twoway_transmittance = remove_edges(twoway_transmittance, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor((0.75, 0.75, 0.75))
        my_cmap = cmocean.cm.thermal
        # # Put negative values to 1e-9 so they don't appear transparent
        # twoway_transmittance[(twoway_transmittance<0) & ~twoway_transmittance.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, twoway_transmittance.T, cmap=my_cmap)
        plt.clim(0.1, 1)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Two-way\ transmittance\ %s}$' % channel, fontsize=self.axes_titlesize, y=self.axes_title_pad)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='min')
        cbar.set_label(label="$T^{2}$",)
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        
        # Save figure
        filename = f"{channel}_twoway_transmittance"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask(self, mask):
        """Plot composite mask"""
    
        # Take only Bits 1–3
        mask = np.bitwise_and(mask, 7)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)
        
        # Labels
        clabels = ['Clear air',
                   'Atmospheric feature',
                   'Low confidence',
                   'Surface/Subsurface',
                   'Fully Attenuated']
    
        # Colormap
        cmaplist = ['1',
                    "#00539c",
                    '0.6',
                    (88./255., 41./255., 0./255.),
                    (222./255., 41./255., 22./255.)]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.array((0.5, 1.5, 2.5, 3.5, 5.5, 7.5))
                    # 1: Clear air
                    # 2: Atmospheric feature
                    # 3: Low confidence
                    # 5: Surface/Subsurface
                    # 7: Totally Attenuated
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
        plt.title(r'$\mathbf{Composite\ detection\ feature\ mask\ %s}$' % VERSION_2D_MCDA, fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 4
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = "mask_composite"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask_strong_weak(self, mask):
        """Plot composite mask"""
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)
    
        # Labels
        clabels = ['Clear',
                   'Weak feature',
                   'Strong feature',
                   'Low confidence',
                   'Surface/Subsurface',
                   'Fully Attenuated']
    
        # Colormap
        cmaplist = ['1',
                    "#fdac53",
                    "#34568b",
                    '0.6',
                    (88./255., 41./255., 0./255.),
                    (222./255., 41./255., 22./255.)]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.array((0.5, 1.5, 2.25, 2.75, 3.5, 5.5, 7.5))
                    # 1: Clear air
                    # 2: Weak feature
                    # 2.5: Strong feature
                    # 3: Low confidence
                    # 5: Surface/Subsurface
                    # 7: Totally Attenuated
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
        plt.title(r'$\mathbf{Composite\ detection\ feature\ mask\ %s}$' % VERSION_2D_MCDA, fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 4
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center',fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = "mask_composite_strong_weak"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask_channel(self, mask):
        """Plot composite mask"""
    
        # Change value of mask
        new_mask = np.ma.zeros(mask.shape)
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 1] = 1
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 8)] = 2
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 16)] = 3
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 32)] = 4
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+16))] = 5
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+32))] = 6
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (16+32))] = 7
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+16+32))] = 8
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 3] = 9
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 5] = 10
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 7] = 11
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            new_mask = remove_edges(new_mask, self.edges_removal)
        
        # Labels
        clabels = ['Clear air',
                   '532 par only',
                   '532 per only',
                   '1064 only',
                   '532 par + 532 per',
                   '532 par + 1064',
                   '532 per + 1064',
                   '532 par + 532 per + 1064',
                   'Low confidence',
                   'Surface/Subsurface',
                   'Totally Attenuated']
    
        # Colormap
        palette = ([1.0,           1.0,      1.0], # 1: Clear air
                   [0.9,           0.0,      0.0], # 2: 532 par only (red)
                   [0.9,           0.9,      0.0], # 3: 532 per only (yellow)
                   [135./255, 206./255, 235./255], # 4: 1064 only (blue)
                   [255./255, 140./255,   0./255], # 5: 532 par + 532 per (orange)
                   [102./255,   0./255, 153./255], # 6: 532 par + 1064 (violet)
                   [152./255, 251./255, 152./255], # 7: 532 per + 1064 (green)
                   [0.0,           0.0,      0.0], # 8: 532 par + 532 per + 1064 (black)
                   [0.8,           0.8,      0.8], # 9: Low confidence (light grey)
                   [0.5,           0.0,      0.0], # 10: Surface/Subsurface (brown)
                   [0.5,           0.5,      0.5]) # 11: Totally Attenuated (grey)
        my_cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.arange(len(palette) + 1) + 0.5
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, new_mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS)
        plt.title(r'$\mathbf{Composite\ detection\ feature\ mask\ %s}$' % VERSION_2D_MCDA, fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 4
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        cbar.set_label("Level of detection", labelpad=80)
       
        # Save figure
        filename = "mask_composite_channel"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    tic_main_program = print_time()
    args = parse_arguments()
    globals().update(load_yaml_configuration(
        args.configuration,
        {
            "INDATA_FOLDER", "VERSION_2D_MCDA", "TYPE_2D_MCDA", "INPUT_FILE_FORMAT",
            "EDGES_REMOVAL", "MAX_DETECT_LEVEL", "PLOT_ALL_STEPS", "INVERT_XAXIS",
            "YMIN", "YMAX", "BROWSE_IMAGE_ASPECT_RATIO", "FIGURES_PATH",
        },
    ))

    
    # *******************************
    # *** Load 2D-McDA data file ***
    input_file_format = INPUT_FILE_FORMAT.lower()
    if input_file_format not in ("netcdf", "hdf"):
        raise ValueError("INPUT_FILE_FORMAT must be either 'netCDF' or 'HDF'")

    print(f"\n*****Load 2D-McDA {INPUT_FILE_FORMAT} data file...*****")
    
    # Get filename and filepath
    file_extension = "nc" if input_file_format == "netcdf" else "hdf"
    filename_2d_mcda = f"CAL_LID_L2_2D_McDA-{TYPE_2D_MCDA}-{VERSION_2D_MCDA.replace('.', '-')}." \
                       f"{GRANULE_DATE}{GRANULE_SECTION}.{file_extension}"
    granule_date_dict = split_granule_date(GRANULE_DATE)
    datafile = os.path.join(INDATA_FOLDER, f"2D_McDA.{VERSION_2D_MCDA.replace('V', 'v')}",
                            str(granule_date_dict['year']),
                            f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                            f"{granule_date_dict['day']:02d}",
                            filename_2d_mcda)
    # Also allow INDATA_FOLDER to point directly to the granule directory.
    direct_datafile = os.path.join(INDATA_FOLDER, filename_2d_mcda)
    if os.path.isfile(direct_datafile):
        datafile = direct_datafile

    # Open the selected file format
    print(f"\tGranule path: {datafile}")
    if input_file_format == "hdf":
        cal_2d_mcda = CALIPSOReader(datafile)
        lat_granule = cal_2d_mcda.get_data("Latitude")
        lon_granule = cal_2d_mcda.get_data("Longitude")
    else:
        cal_2d_mcda = Dataset(datafile, mode="r")
        lat_granule = cal_2d_mcda.variables["Latitude"][:]
        lon_granule = cal_2d_mcda.variables["Longitude"][:]

    # Get prof_min and prof_max (null limits = first/last profile of the file)
    if SLICE_START_END_TYPE == 'longitude':
        prof_min, prof_max = get_prof_min_max_indexes_from_lon(lon_granule, SLICE_START, SLICE_END)
    else:
        prof_min = 0 if SLICE_START is None else SLICE_START
        prof_max = lon_granule.size - 1 if SLICE_END is None else SLICE_END
        
    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {prof_min:d} "
          f"(lat = {lat_granule[prof_min]:.2f} / lon = {lon_granule[prof_min]:.2f}) "
          f"to max profile index {prof_max:d} "
          f"(lat = {lat_granule[prof_max]:.2f} / lon = {lon_granule[prof_max]:.2f})")
    
    # Load 2D-McDA parameters
    data_dict_cal_2d_mcda = {}
    cal_2d_mcda_keys = [
        "Latitude",
        "Longitude",
        "Profile_ID",
        "Profile_Time",
        "Profile_UTC_Time",
        "Altitude",
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064",
        "Composite_Detection_Flags"
    ]
    for key in cal_2d_mcda_keys:
        if input_file_format == "hdf":
            data = cal_2d_mcda.get_data(key, prof_min, prof_max, 'profindex')
        else:
            if key not in cal_2d_mcda.variables:
                raise KeyError(f"Variable '{key}' not found in file '{datafile}'")
            variable = cal_2d_mcda.variables[key]
            variable_slice = [slice(None)] * variable.ndim
            if "Profile_ID" in variable.dimensions:
                profile_axis = variable.dimensions.index("Profile_ID")
                variable_slice[profile_axis] = slice(prof_min, prof_max + 1)
            data = variable[tuple(variable_slice)]
        data_dict_cal_2d_mcda[key] = data
    
    data_dict_cal_2d_mcda_steps = {}
    if PLOT_ALL_STEPS:
        cal_2d_mcda_steps_keys = [
            "Parallel_Detection_Flags_532_steps",
            "Perpendicular_Detection_Flags_532_steps",
            "Detection_Flags_1064_steps",
            "Parallel_Attenuated_Scattering_Ratio_532_steps",
            "Perpendicular_Attenuated_Scattering_Ratio_532_steps",
            "Attenuated_Scattering_Ratio_1064_steps",
            "Parallel_CumulativeTwoWayTransmittance_532",
            "Perpendicular_CumulativeTwoWayTransmittance_532",
            "CumulativeTwoWayTransmittance_1064"
        ]
        for key in cal_2d_mcda_steps_keys:
            if input_file_format == "hdf":
                data = cal_2d_mcda.get_data(key, prof_min, prof_max, 'profindex')
            else:
                if key not in cal_2d_mcda.variables:
                    raise KeyError(f"Variable '{key}' not found in file '{datafile}'")
                variable = cal_2d_mcda.variables[key]
                variable_slice = [slice(None)] * variable.ndim
                if "Profile_ID" in variable.dimensions:
                    profile_axis = variable.dimensions.index("Profile_ID")
                    variable_slice[profile_axis] = slice(prof_min, prof_max + 1)
                data = variable[tuple(variable_slice)]
            data_dict_cal_2d_mcda_steps[key] = data

    if input_file_format == "netcdf":
        cal_2d_mcda.close()
    
    # Weak/Strong feature mask
    mask_weak_strong = np.bitwise_and(data_dict_cal_2d_mcda["Composite_Detection_Flags"], 7).astype(float)
    # 2.5 = 'Strong' were detection without averaging at least in one channel,
    # keep 2 = 'Weak' elsewhere
    where_strong = ((data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] >= 1) &
                    (data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] <= 4)) |\
                   ((data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] >= 1) &
                    (data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] <= 4)) |\
                   ((data_dict_cal_2d_mcda["Detection_Flags_1064"] >= 1) &
                    (data_dict_cal_2d_mcda["Detection_Flags_1064"] <= 4))
    mask_weak_strong[where_strong] = 2.5
    
    
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
    plot_fig.set_max_detect_level(MAX_DETECT_LEVEL)
    plot_fig.set_coordinates(data_dict_cal_2d_mcda["Latitude"],data_dict_cal_2d_mcda["Longitude"],
                             data_dict_cal_2d_mcda["Altitude"])
    
    # Plot the 3 channel masks
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"], '532_par')
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"], '532_per')
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Detection_Flags_1064"], '1064')

    # Plot the composite masks
    plot_fig.plot_composite_mask(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
    plot_fig.plot_composite_mask_strong_weak(mask_weak_strong)
    plot_fig.plot_composite_mask_channel(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
    
    # Plot every steps
    if PLOT_ALL_STEPS:
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Parallel_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Parallel_Attenuated_Scattering_Ratio_532_steps"],
                   '532_par')
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Perpendicular_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Perpendicular_Attenuated_Scattering_Ratio_532_steps"],
                   '532_per')
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Detection_Flags_1064_steps"],
                   data_dict_cal_2d_mcda_steps["Attenuated_Scattering_Ratio_1064_steps"],
                   '1064')
    if PLOT_ALL_STEPS:
        plot_fig.plot_twoway_transmittance(data_dict_cal_2d_mcda_steps["Parallel_CumulativeTwoWayTransmittance_532"], '532_par')
        plot_fig.plot_twoway_transmittance(data_dict_cal_2d_mcda_steps["Perpendicular_CumulativeTwoWayTransmittance_532"], '532_per')
        plot_fig.plot_twoway_transmittance(data_dict_cal_2d_mcda_steps["CumulativeTwoWayTransmittance_1064"], '1064')
    
    
    print_time(tic_main_program)
