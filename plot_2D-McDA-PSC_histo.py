#!/usr/bin/env python
# coding: utf8

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from pathlib import Path
import pandas as pd
import re
import cmocean

# =========================
# CONFIGURATION
# =========================
RECOMPUTE_HISTO = True  # <-- True = recalcul, False = reload + plot
VERSION = "2.7.2"
ROOT_DIR = Path(f"/home/vaillant/codes/projects/2D_McDA_PSC/out/data/2D_McDA_PSC.v{VERSION}")
START_DATE = "2009-05-01"
END_DATE   = "2009-10-31"

HISTO_FILE = Path(
    "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
    "plot_2D-McDA-PSC_histo/"
    f"plot_2D-McDA-PSC_histo_v{VERSION}_{START_DATE}_{END_DATE}.nc"
)

PLOT_FILE = Path(
    "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
    "plot_2D-McDA-PSC_histo/"
    f"plot_2D-McDA-PSC_histo_v{VERSION}_{START_DATE}_{END_DATE}.png"
)

# PSC classes (from flag_values)
PSC_CLASSES = np.array([-4, 0, 1, 2, 3, 4, 5, 6])
N_CLASSES = len(PSC_CLASSES)

if RECOMPUTE_HISTO:
        
    # =========================
    # DATE RANGE (daily)
    # =========================
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    day_index = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}

    # =========================
    # FILE SEARCH
    # =========================
    pattern = re.compile(
        r".*(\d{4}-\d{2}-\d{2})T.*ZN\.nc$"
    )

    all_files = sorted(ROOT_DIR.rglob("*.nc"))

    selected_files = []
    for f in all_files:
        m = pattern.match(f.name)
        if m:
            file_date = m.group(1)
            if START_DATE <= file_date <= END_DATE:
                selected_files.append(f)

    print(f"Found {len(selected_files)} files in date range")

    # =========================
    # FIRST FILE: GET ALTITUDE GRID
    # =========================
    with xr.open_dataset(selected_files[0]) as ds0:
        altitudes = ds0["Altitude"].values

    n_days = len(days)
    n_alt = len(altitudes)

    # histogram: [time, altitude, class]
    histo = np.zeros((N_CLASSES, n_alt, n_days), dtype=np.int64)

    # =========================
    # LOOP OVER FILES
    # =========================
    for f in selected_files:
        print("Processing:", f.name)

        # extract date
        m = pattern.match(f.name)
        if not m:
            continue

        date_str = m.group(1)
        if date_str not in day_index:
            continue

        d = day_index[date_str]

        with xr.open_dataset(f) as ds:
            comp = ds["PSC_Composition"].values

            for i_class, val in enumerate(PSC_CLASSES):
                mask = (comp == val)
                histo[i_class, :, d] += mask.sum(axis=0)

    # =========================
    # SAVE OUTPUT
    # =========================
    HISTO_FILE.parent.mkdir(parents=True, exist_ok=True)

    ds_out = xr.Dataset(
        data_vars=dict(
            PSC_histo=(["psc_class", "altitude", "time"], histo)
        ),
        coords=dict(
            psc_class=PSC_CLASSES,
            altitude=altitudes,
            time=[d.strftime("%Y-%m-%d") for d in days]  
        )
    )

    ds_out.to_netcdf(HISTO_FILE)

    print("Saved to:", HISTO_FILE)



# =========================
# LOAD
# =========================
ds = xr.open_dataset(HISTO_FILE)

histo = ds["PSC_histo"].values  

psc_classes = ds["psc_class"].values
altitude = ds["altitude"].values
time = pd.to_datetime(ds["time"].values)

n_class = len(psc_classes)

# =========================
# INDEX HELPERS
# =========================
idx = {val: i for i, val in enumerate(psc_classes)}

def get_class(val):
    return histo[idx[val], :, :]  # (altitude, time)

# =========================
# GROUPS
# =========================
sts = get_class(1)
nat = get_class(2)
sbs = get_class(3)
ice = get_class(4)
enh_nat = get_class(5)
wave = get_class(6)

all_psc = sts + nat + ice + enh_nat + wave
ice_group = ice + wave
nat_group = nat + enh_nat

# =========================
# CREATE GRID EDGES
# =========================
time_num = np.arange(len(time))
alt = altitude

# edges (simple version)
time_edges = np.arange(len(time) + 1)
alt_edges = np.arange(len(altitude) + 1)

# =========================
# FIGURE SETUP
# =========================
fig, axes = plt.subplots(
    5,
    1,
    figsize=(10, 10),
    sharex=True,
    sharey=True
)

# =========================
# PLOT DEFINITIONS
# =========================
plots = [
    ("All PSCs (STS, NAT, enhanced NAT, ice, wave ice)", all_psc),
    ("STS", sts),
    ("Ice + wave ice", ice_group),
    ("NAT + enhanced NAT", nat_group),
    ("SBS", sbs),
]

# =========================
# PLOT LOOP
# =========================
for ax, (title, data) in zip(axes, plots):

    ax.set_facecolor('0.5')
    data_masked = np.ma.masked_where(data == 0, data)
    pm = ax.pcolormesh(time, 
                       altitude, 
                       data_masked, 
                       cmap=cmocean.cm.thermal, 
                       vmin=1, vmax=12000,
                       shading="auto")
    
    ax.set_title(title)
    ax.set_ylabel("Altitude (km)")

# =========================
# FINAL AXIS
# =========================
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())  # ticks au 1er du mois
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(''))  # "May", "June", etc.

# Center month labels
start = pd.to_datetime(time.min()).normalize()
end = pd.to_datetime(time.max()).normalize()

months = pd.date_range(start, end, freq="MS")  # 1er du mois
mid_months = months + pd.Timedelta(days=14)

month_labels = [d.strftime('%B') for d in months]
year_labels = [d.strftime('%Y') for d in months]

for x, mlabel, ylabel in zip(mid_months, month_labels, year_labels):

    # Month (ligne du haut)
    axes[-1].text(
        x,
        -0.06,
        mlabel,
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10
    )

    # Year (ligne du bas)
    axes[-1].text(
        x,
        -0.17,
        ylabel,
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="gray",
        fontweight="bold"
    )

plt.suptitle(f"2D-McDA-PSC v{VERSION}")

plt.tight_layout()

cbar = fig.colorbar(pm, ax=axes, orientation="vertical", label="Occurrences", extend="max")
ticks = [1] + list(np.arange(1000, 12001, 1000))
cbar.set_ticks(ticks)

plt.savefig(PLOT_FILE, dpi=400, transparent=False)
print("\t%s saved" % PLOT_FILE)