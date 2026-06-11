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
        
        
    def plot_mask(self, step, mask, channel):
        """Plot mask for each step. If only final mask, put step = None"""

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
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
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
        plt.title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 8
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
            if j == 3:
                cbar.ax.text(4, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), 'Detection level',
                             va='center', rotation=90, fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        # cbar.set_label("Level of detection", labelpad=self.clabelpad)
       
        # Save figure
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_{channel}_mask"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def backscatter_cbar_labels(self, cbar):
        cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        cbar.ax.tick_params(which='both', labelright=False)
        minor_bounds = cmlidar.cm.BACKSCATTER_DISCRETE_BOUNDS
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_bounds))
        cbar_minor_label = ['1.0',
                            '1.0', '3.0', '6.0',
                            '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                            '1.0', '1.5', '2.0', '3.0', '5.0']
        for j, bound in enumerate(minor_bounds):
            cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=6)
        cbar_major_label = [r'$\mathbf{×10^{-5}}$', r'$\mathbf{×10^{-4}}$', r'$\mathbf{×10^{-3}}$', r'$\mathbf{×10^{-2}}$']
        c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
        for j, bound in enumerate(c_bar_major_values):
            cbar.ax.text(3.5, bound, cbar_major_label[j], va='center', fontsize=8)
        cbar.set_label(label=r"$\langle\beta^{\prime}\rangle$ (km$^{-1}$ sr$^{-1}$)")


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
        # palette = ["#000000",
        #             '#000444', '#010846', '#030b48', '#050e4a', '#08104c', '#0b134f', '#0e1551', '#101753', '#131a55', '#151c57', '#171e59', '#1a215c', '#1c235e', '#1e2560', '#202862', '#222a65', '#242d67', '#262f69', '#28316b', '#2a346e', '#2c3670', '#2e3972', '#303b74', '#323e77', '#344079', '#36437b', '#37467d', '#394880', '#3b4b82', '#3d4d84', '#3f5087', '#415389', '#43558b', '#44588e', '#465b90', '#485d92', '#4a6095', '#4c6397', '#4e6599', '#4f689c', '#516b9e', '#536da1', '#5570a3', '#5773a5', '#5876a8', '#5a79aa', '#5c7bad', '#5e7eaf', '#6081b1', '#6284b4', '#6387b6', '#6589b9', '#678cbb', '#698fbe', '#6b92c0', '#6d95c3', '#6e98c5', '#709bc7', '#729eca', '#74a1cc', '#76a4cf', '#78a7d1', '#79a9d4', '#7bacd6', '#7dafd9', '#7fb2db', '#81b5de', '#83b8e0', '#85bbe3', '#86bee5', '#88c1e8', '#8ac4eb', '#8cc7ed', '#8ecbf0', '#90cef2', '#92d1f5', '#93d4f7', '#95d7fa', '#97dafc',
        #             '#99ddff', '#a8e1f7', '#b6e4ef', '#c3e8e6', '#ceecde', '#d9efd5', '#e3f3cd', '#edf7c4', '#f6fbbb', '#ffffb2',
        #             '#fffcb0', '#fff9ae', '#fef6ac', '#fef3aa', '#fdf1a8', '#fdeea6', '#fceba4', '#fce8a1', '#fbe59f', '#fbe29d', '#fae09b', '#fadd99', '#f9da97', '#f9d795', '#f8d493', '#f7d191', '#f7cf8f', '#f6cc8d', '#f6c98b', '#f5c689', '#f4c387', '#f4c085', '#f3be83', '#f2bb81', '#f1b87f', '#f1b57d', '#f0b27b', '#efaf79', '#eead77', '#eeaa76', '#eda774', '#eca472', '#eba170', '#ea9e6e', '#e99c6c', '#e9996a', '#e89668', '#e79366', '#e69064', '#e58d62', '#e48a61', '#e3875f', '#e2845d', '#e1825b', '#e07f59', '#df7c57', '#de7956', '#dd7654', '#dc7352', '#db7050', '#da6d4e', '#d96a4c', '#d8674b', '#d76449', '#d66047', '#d45d45', '#d35a44', '#d25742', '#d15440', '#d0503e', '#cf4d3d', '#ce493b', '#cc4639', '#cb4238', '#ca3e36', '#c93b34', '#c73633', '#c63231', '#c52e2f', '#c4292e', '#c2242c', '#c11e2b', '#c01629', '#be0d28',
        #             '#bd0026', '#b31e2b', '#a82c30', '#9e3635', '#933d3a', '#88433f', '#7c4844', '#704c49', '#62504e', '#535353',
        #             '#555555', '#575757', '#595959', '#5b5b5b', '#5d5d5d', '#5f5f5f', '#616161', '#636363', '#656565', '#676767', '#696969', '#6b6b6b', '#6d6d6d', '#6f6f6f', '#717171', '#737373', '#757575', '#777777', '#797979', '#7b7b7b', '#7d7d7d', '#7f7f7f', '#818181', '#848484', '#868686', '#888888', '#8a8a8a', '#8c8c8c', '#8e8e8e', '#909090', '#929292', '#949494', '#979797', '#999999', '#9b9b9b', '#9d9d9d', '#9f9f9f', '#a1a1a1', '#a4a4a4', '#a6a6a6', '#a8a8a8', '#aaaaaa', '#acacac', '#afafaf', '#b1b1b1', '#b3b3b3', '#b5b5b5', '#b8b8b8', '#bababa', '#bcbcbc', '#bebebe', '#c1c1c1', '#c3c3c3', '#c5c5c5', '#c7c7c7', '#cacaca', '#cccccc', '#cecece', '#d0d0d0', '#d3d3d3', '#d5d5d5', '#d7d7d7', '#dadada', '#dcdcdc', '#dedede', '#e0e0e0', '#e3e3e3', '#e5e5e5', '#e7e7e7', '#eaeaea', '#ececec', '#eeeeee', '#f1f1f1', '#f3f3f3', '#f6f6f6', '#f8f8f8', '#fafafa', '#fdfdfd', '#ffffff'
        #             ]
        # my_cmap = mpl.colors.ListedColormap(palette)
        # b1 = np.logspace(-5, -3+np.log10(1.5), 85)
        # b2 = np.logspace(-3+np.log10(1.5), -3+np.log10(6.5), 85)[1:]
        # b3 = np.logspace(-3+np.log10(6.5), -1, 84)[1:]
        # bounds = np.concatenate((b1, b2, b3))
        # colors = my_cmap(np.arange(len(palette)))
        # my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')

        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9
        # pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=my_norm)
        # plt.clim(1e-5, 1e-1)
        # pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap="backscatter_continuous", norm=cmlidar.cm.backscatter_continuous_norm)
        # self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm(), rasterized=True)
        plt.clim(1e-6, 2e-3)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\beta^{\prime}$ (km$^{-1}$ sr$^{-1}$)", labelpad=5)
        # cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        # cbar.ax.tick_params(which='both', labelright=False)
        # cbar_major_label = [r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$']
        # for j, bound in enumerate(np.array((1e-5, 1e-4, 1e-3, 1e-2))):
        #     cbar.ax.text(2, bound, cbar_major_label[j], va='center', fontsize=9)
        # # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        # minor_locators = np.concatenate((np.arange(2,10)*1e-6,
        #                                  np.arange(2,10)*1e-5,
        #                                  np.arange(2,10)*1e-4,
        #                                  np.arange(2,10)*1e-3,
        #                                  np.arange(2,10)*1e-2))
        # cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
        # cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both')
        # cbar.set_label(label=r"$\beta^{\prime}$ (km$^{-1}$ sr$^{-1}$)", fontsize=8, labelpad=45)
        # self.backscatter_cbar_labels(cbar)

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_sr_signal(self, sr_signal, title, filename):

        # Mask where fill_value
        sr_signal = np.ma.masked_where(sr_signal == FILL_VALUE_FLOAT, sr_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            sr_signal = remove_edges(sr_signal, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        sr_signal[(sr_signal<0) & ~sr_signal.mask] = 1e-9
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, sr_signal.T, cmap=my_cmap, norm=LogNorm(), rasterized=True)
        plt.clim(1, 10)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"Scattering ratio", labelpad=5)

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_temperature(self, temperature, title, filename):

        # Mask where fill_value
        temperature = np.ma.masked_where(temperature == FILL_VALUE_FLOAT, temperature)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            temperature = remove_edges(temperature, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor('0.5')
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        my_cmap = cmocean.cm.thermal
        my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, temperature.T, cmap=my_cmap, rasterized=True)
        plt.clim(180, 220)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"Temperature (K)", labelpad=5)

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask_channel(self, mask):
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

        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T,
                        cmap=cmap, norm=norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(f"2D-McDA-PSC {VERSION_2D_McDA} composite detection mask (by channel)", fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=4 if lab==label_all_channel else 7, transform=cbar.ax.transAxes)

        # Save figure
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_composite_channel_mask"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_psc_composition(self, mask, temp, plot_temp=False, plot_tropopause=False):
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
        mask_plot = np.vectorize(mapping.get)(mask)

        # Create colormap
        my_cmap = mpl.colors.ListedColormap(palette)

        # Define color boundaries for ordered indices
        colorbins = np.arange(len(clabels) + 1) - 0.5

        # Create normalization
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)

    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask_plot.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(f'2D-McDA-PSC {VERSION_2D_McDA} PSC composition', weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        
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
            i_mid = len(x_centers) // 2
            ax0.text(x_centers[i_mid], tropopause_alt[i_mid] + 1.0,
                     "Tropopause", color="white", fontsize=10, ha="center", va="bottom", zorder=11, weight="bold")

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        
        # Save figure
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_PSC_composition"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    INDATA_FOLDER = "/home/vaillant/codes/projects/2D_McDA_PSC/out/data/"
    GRANULE_DATE = "2010-01-18T00-19-57ZN"
    GRANULE_SECTION = "" # void if complete file
    VERSION_2D_McDA = "V2.6.0"
    TYPE_2D_McDA = "Prototype"
    SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the 2D-McDA file) or 'longitude'
    SLICE_START = 170.59 # profindex or longitude
    SLICE_END = 27.95 # profindex or longitude
    EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
    MAX_DETECT_LEVEL = 5
    INVERT_XAXIS = False
    YMIN = 8.4
    YMAX = 30
    PLOT_ASPECT_RATIO = "browse" # "browse", "spec" or None
    FIGURES_FILETYPE = 'png' #'png' 'svg'
    FLAG_GRANULE = True # Write granule name in the plots
    FIGURES_PATH = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
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
        "PSC_Ice_Mixture_Boundary"]
    if True:
        cal_2d_mcda_keys += [
            "Homogeneous_Chunks_Mean_Particulate_Perpendicular_Attenuated_Backscatter_532",
            "Homogeneous_Chunks_Mean_Total_Attenuated_Scattering_Ratio_532",
            "Homogeneous_Chunks_Mean_PSC_Ice_Mixture_Boundary",
            "Homogeneous_Chunks_Mean_Temperature",
            "PSC_Composition"]
        # cal_2d_mcda_keys += [
        #     "Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532",
        #     "Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532",
        #     "Homogeneous_Chunks_Mean_Total_Attenuated_Backscatter_1064"]
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


    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    if PLOT_ASPECT_RATIO == 'browse':
        plot_fig.fig_w = cm2in(16) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.87
        plot_fig.adj_top = 0.81
    elif PLOT_ASPECT_RATIO == 'spec':
        plot_fig.fig_w = cm2in(12) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.1
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.83
        plot_fig.adj_top = 0.81
        plot_fig.axes_title_pad = 1.15
        plot_fig.clabelpad = 40
    plot_fig.filetype = FIGURES_FILETYPE
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, lon_granule[prof_min], lon_granule[prof_max])
    plot_fig.set_head_filename(GRANULE_DATE, lon_granule[prof_min], lon_granule[prof_max])
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_max_detect_level(MAX_DETECT_LEVEL)
    plot_fig.set_coordinates(data_dict_cal_2d_mcda["Latitude"], data_dict_cal_2d_mcda["Longitude"],
                             data_dict_cal_2d_mcda["Altitude"])
    
    # Plot signals
    if True:
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB532par"
        title = r"$\mathbf{532\ nm\ Parallel\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB532per"
        title = r"$\mathbf{532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB1064"
        title = r"$\mathbf{1064\ nm\ Attenuated\ Backscatter}\ \beta^{\prime}_{1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Total_Attenuated_Backscatter_1064"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB532par_part"
        title = r"$\mathbf{532\ nm\ Particulate\ Parallel\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Particulate_Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB532per_part"
        title = r"$\mathbf{532\ nm\ Particulate\ Perpendicular\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Particulate_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_AB1064_part"
        title = r"$\mathbf{1064\ nm\ Particulate\ Attenuated\ Backscatter}\ \beta^{\prime}_{1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Particulate_Total_Attenuated_Backscatter_1064"], title, filename)
        
    # Plot averaged signals on homogeneous chunks
    if False:
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_AB532par"
        title = r"$\mathbf{Mean\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_AB532per"
        title = r"$\mathbf{Mean\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \beta^{\prime}_{532,\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_AB1064"
        title = r"$\mathbf{Mean\ 1064\ nm\ Attenuated\ Backscatter}\ \beta^{\prime}_{1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Total_Attenuated_Backscatter_1064"], title, filename)

    if True:
        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_AB532per_part"
        title = r"$\mathbf{Mean\ 532\ nm\ Particulate\ Perpendicular\ Attenuated\ Backscatter}\ \beta^{\prime}_{p,532,\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Particulate_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_R532"
        title = r"$\mathbf{Mean\ 532\ nm\ Attenuated\ Scattering\ Ratio}\ R^{\prime}_{532}$"
        plot_fig.plot_sr_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Total_Attenuated_Scattering_Ratio_532"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_R532_nat_ice_threshold"
        title = r"$\mathbf{532\ nm\ Attenuated\ Scattering\ Ratio NAT/ice\ threshold}\ R^{\prime}_{532}$"
        plot_fig.plot_sr_signal(data_dict_cal_2d_mcda["PSC_Ice_Mixture_Boundary"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_R532_nat_ice_threshold"
        title = r"$\mathbf{Mean\ 532\ nm\ Attenuated\ Scattering\ Ratio NAT/ice\ threshold}\ R^{\prime}_{532}$"
        plot_fig.plot_sr_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_PSC_Ice_Mixture_Boundary"], title, filename)

        filename = f"2D-McDA-PSC{VERSION_2D_McDA}_chunks_mean_temperature"
        title = r"$\mathbf{Mean\ Temperature}$"
        plot_fig.plot_temperature(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Temperature"], title, filename)


    # Plot the 3 channel masks
    if True:
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"], '532_par')
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"], '532_per')
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Detection_Flags_1064"], '1064')

    # Plot the composite masks
    if True:
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

        plot_fig.plot_composite_mask_channel(composite_mask_channel)
    
    # Plot classification mask
    if True:
        plot_fig.plot_psc_composition(data_dict_cal_2d_mcda["PSC_Composition"], 
                                      data_dict_cal_2d_mcda["Temperature"],
                                      PLOT_TEMPERATURE_CONTOURS_OVER_COMPOSITION,
                                      PLOT_TROPOPAUSE)

    print_time(tic_main_program)
