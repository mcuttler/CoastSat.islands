#==========================================================#
# CoastSat.islands — full workflow example
#
# Sand polygon extraction always runs.
# Vegetation polygon extraction is optional — set
#     'track_veg': True
# in Section 4 (Settings) to enable it.
#
# Authors:
#   M. Cuttler  (University of Western Australia)
#   K. Vos      (Water Research Laboratory, UNSW)
#==========================================================#

#%% 1. Imports and GEE authentication

import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()
import pandas as pd
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
from coastsat import SDS_islands, SDS_download, SDS_preprocess, SDS_tools, SDS_transects

%matplotlib qt

# Authenticate and initialise Google Earth Engine
project_name = 'ee-coastsat-mc'   # <-- set your GEE Cloud Project name here
SDS_download.authenticate_and_initialize(project_name)


#%% 2. Site definition

# Region of interest — load from KML or enter coordinates manually
polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'example', 'EVA.kml'))
# polygon = [[114.4249504953477, -21.9295184484435],
#            [114.4383556651795, -21.92949300318377],
#            [114.4388731500701, -21.91491228133647],
#            [114.4250081185656, -21.91495393621703],
#            [114.4249504953477, -21.9295184484435]]

polygon = SDS_tools.smallest_rectangle(polygon)

dates    = ['2019-01-01', '2019-01-20']
sat_list = ['L8', 'S2']
sitename = 'EVA'
filepath_data = os.path.join(os.getcwd(), 'data')

inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data,
}

# Check image availability before downloading (optional)
# SDS_download.check_images_available(inputs)


#%% 3. Download or load imagery

# Download from GEE — only needed once; comment out after first run
# inputs['include_T2'] = True  # uncomment to include Landsat Tier 2 (not recommended for time-series)
# metadata = SDS_download.retrieve_images(inputs)

# Load existing metadata
metadata = SDS_download.get_metadata(inputs)


#%% 4. Settings

settings = {
    # ------------------------------------------------------------------ #
    # Main switch — set to True to also extract the vegetation polygon.   #
    # Sand detection and all sand outputs are always produced.            #
    # ------------------------------------------------------------------ #
    'track_veg': True,              # True = extract vegetation polygon alongside sand

    # --- general ---
    'cloud_thresh': 0.5,             # maximum cloud cover fraction (0–1)
    'dist_clouds': 200,              # distance around clouds to mask (m)
    'output_epsg': 28350,            # output CRS as EPSG code

    # --- quality control ---
    'check_detection': True,         # show each detection for manual validation
    'adjust_detection': False,       # allow threshold adjustment per image
    'save_figure': True,             # save a .jpg for each kept detection

    # --- sand detection ---
    'min_beach_area': 1000,          # minimum sand area (m²)
    'buffer_size': 100,              # buffer around sandy pixels (m)
    'min_length_sl': 500,            # minimum shoreline length (m)
    'cloud_mask_issue': True,        # True if sand pixels are being masked in black
    'sand_color': 'default',         # 'default' | 'latest' | 'dark' | 'bright'
    'pan_off': False,                # True to disable Landsat 7/8/9 pansharpening
    's2cloudless_prob': 100,         # s2cloudless cloud probability threshold (0–100)

    # --- vegetation detection (only used when track_veg = True) ---
    'min_veg_area': 10000,           # minimum vegetated area (m²) — same units as min_beach_area
    'veg_thresh': 'auto',            # 'auto' = Otsu thresholding; or fixed float, e.g. 0.25
    'ndvi_thresh_ceiling': 0.25,   # Otsu upper clamp — lower this (e.g. 0.20) for sparse/patchy veg
    'ndvi_floor':          0.10,   # Otsu lower clamp — rarely needs changing

    # --- inputs reference ---
    'inputs': inputs,
}

# [OPTIONAL] create a reference shoreline — helps identify outliers and false detections
# Note: also required by tide_correct_sand_polygon (Section 10)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# maximum distance (m) from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100


#%% 5. Extract sand polygons  (+ vegetation when track_veg = True)

# When track_veg = True, vegetation extraction runs inside the same
# preprocessing loop as sand — no second image pass is needed.
# Output always contains sand fields; veg fields are added automatically
# when track_veg = True, and SDS_tools.merge_output handles them together.

# Note: make sure your working directory is set to your CoastSat directory
# so the NN classifiers in /classification/models/ can be found.
output = SDS_islands.extract_sand_poly(metadata, settings)

# Report what was extracted
print(f'\nExtracted {len(output["dates"])} images')
print('  Sand fields: sand_area, sand_perimeter, sand_centroid, sand_points, '
      'sand_eccentricity, sand_orientation')
if settings.get('track_veg', False):
    print('  Veg fields:  veg_area, veg_perimeter, veg_centroid, veg_points, '
          'veg_eccentricity, veg_orientation, ndvi_threshold')


#%% 6. Sand polygon plots

# --- map: sand polygon outlines through time ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_title(f'{sitename} — sand polygons')
ax.grid(linestyle=':', color='0.5')
for i in range(len(output['dates'])):
    pts = output['sand_points'][i]
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], '-', label=output['dates'][i].strftime('%d-%m-%Y'))
ax.legend(fontsize=8)
plt.tight_layout()

# --- time-series: sand area, orientation, eccentricity ---
fig, axes = plt.subplots(3, 1, figsize=(14, 7), tight_layout=True, sharex=True)
axes[0].plot(output['dates'], output['sand_area'], 'b-x')
axes[0].set(title='Sub-aerial sand area (m²)')
axes[0].grid(True, linestyle=':')
axes[1].plot(output['dates'], np.array(output['sand_orientation']), 'b-x')
axes[1].set(title='Orientation (degrees from horizontal)')
axes[1].grid(True, linestyle=':')
axes[2].plot(output['dates'], output['sand_eccentricity'], 'b-x')
axes[2].set(title='Eccentricity')
axes[2].grid(True, linestyle=':')
plt.suptitle(f'{sitename} — sand metrics', fontweight='bold')


#%% 7. Vegetation plots  [only runs when track_veg = True]

if settings.get('track_veg', False):

    # --- map: sand and vegetation outlines overlaid ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title(f'{sitename} — sand (tan) and vegetation (green) polygons')
    ax.grid(linestyle=':', color='0.5')
    for pts in output.get('sand_points', []):
        if pts is not None:
            ax.fill(pts[:, 0], pts[:, 1], color='tan', alpha=0.3)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='peru', lw=0.8)
    for pts in output.get('veg_points', []):
        if pts is not None:
            ax.fill(pts[:, 0], pts[:, 1], color='forestgreen', alpha=0.3)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='darkgreen', lw=0.8)
    plt.tight_layout()

    # --- time-series: veg area, orientation, eccentricity ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), tight_layout=True, sharex=True)
    axes[0].plot(output['dates'], output['veg_area'], 'g-x')
    axes[0].set(title='Vegetation area (m²)')
    axes[0].grid(True, linestyle=':')
    axes[1].plot(output['dates'], np.array(output['veg_orientation']), 'g-x')
    axes[1].set(title='Orientation (degrees from horizontal)')
    axes[1].grid(True, linestyle=':')
    axes[2].plot(output['dates'], output['veg_eccentricity'], 'g-x')
    axes[2].set(title='Eccentricity')
    axes[2].grid(True, linestyle=':')
    plt.suptitle(f'{sitename} — vegetation metrics', fontweight='bold')

    # --- comparison: sand vs vegetation area, orientation, eccentricity ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), tight_layout=True, sharex=True)
    axes[0].plot(output['dates'], output['sand_area'], 'b-x', label='Sand')
    axes[0].plot(output['dates'], output['veg_area'], 'g-x', label='Vegetation')
    axes[0].set(title='Area (m²)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    axes[1].plot(output['dates'], np.array(output['sand_orientation']), 'b-x', label='Sand')
    axes[1].plot(output['dates'], np.array(output['veg_orientation']), 'g-x', label='Vegetation')
    axes[1].set(title='Orientation (degrees)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    axes[2].plot(output['dates'], output['sand_eccentricity'], 'b-x', label='Sand')
    axes[2].plot(output['dates'], output['veg_eccentricity'], 'g-x', label='Vegetation')
    axes[2].set(title='Eccentricity')
    axes[2].legend()
    axes[2].grid(True, linestyle=':')
    plt.suptitle(f'{sitename} — sand vs vegetation comparison', fontweight='bold')


#%% 8. Reload saved output  (re-run analysis without re-extracting)

filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output.pkl'), 'rb') as f:
    output = pickle.load(f)


#%% 9. Transect analysis

# Auto-generate shore-normal transects radiating from the island centre
ang_start = 0
ang_end   = 360
ang_step  = 10   # degree interval between transects
settings['heading']         = np.array(list(range(ang_start, ang_end, ang_step)))
settings['island_center']   = np.mean(settings['reference_shoreline'], axis=0).tolist()
settings['transect_length'] = 400
transects = SDS_islands.calc_island_transects(settings)

# [alternative] draw transects manually for non-circular islands
# transects = SDS_transects.draw_transects(output, settings)

# plot transects over sand polygons
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.grid(linestyle=':', color='0.5')
for pts in output['sand_points']:
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], '-', lw=0.8, color='peru')
for key in transects.keys():
    ax.plot(transects[key][:, 0], transects[key][:, 1], 'k-', lw=0.6)
ax.plot(settings['island_center'][0], settings['island_center'][1], 'ro')
plt.tight_layout()

# intersect transects with sand polygon boundaries → cross-shore distance time-series
settings['along_dist'] = 10
cross_distance = SDS_islands.compute_intersection_islands(output, transects, settings)

# intersect transects with vegetation boundary → cross-shore distance time-series
if settings.get('track_veg', False):
    cross_distance_veg = SDS_islands.compute_intersection_veg(output, transects, settings)


#%% 10. Tidal correction

# ---- Option 1: FES2022 global tide model (recommended) -------------------------
# Setup: https://github.com/kvos/CoastSat/blob/master/doc/FES2022_setup.md
# from coastsat import SDS_slope
# import pyfes
# centroid = np.mean(polygon[0], axis=0)
# if centroid[0] < 0: centroid[0] += 360
# filepath_fes = os.path.join(os.pardir, 'CoastSat.webgis', 'aviso-fes-main', 'data', 'fes2022b')
# selected_yaml = os.path.join(filepath_fes, 'fes2022.yaml')
# # Option 1b: latitude-clipped model (1–2 GB RAM)
# # fp_band_configs = os.path.join(os.getcwd(), 'examples', 'tide_model_clipping',
# #                                'fes2022_by_latitude', 'band_configs.json')
# # with open(fp_band_configs, 'r') as f: band_configs = json.load(f)
# # selected_yaml = SDS_slope.select_yaml_for_centroid(centroid, band_configs)['yaml']
# if int(pyfes.__version__.split('.')[0]) < 2026:
#     handlers = pyfes.load_config(selected_yaml)
#     ocean_tide, load_tide = handlers['tide'], handlers['radial']
# else:
#     config = pyfes.config.load(selected_yaml)
#     ocean_tide, load_tide = config.models['tide'], config.models['radial']
# date_range = [pytz.utc.localize(datetime(1984, 1, 1)), pytz.utc.localize(datetime(2030, 1, 1))]
# dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, 900, ocean_tide, load_tide)
# tides_sat = SDS_slope.compute_tide_dates(centroid, output['dates'], ocean_tide, load_tide)
# output_corrected = dict(output)
# output_corrected['tide'] = list(tides_sat)

# ---- Option 2: measured tide data — CoastSat standard UTC CSV -------------------
# CSV must have columns: 'dates' (UTC, pandas-parseable) and 'tide' (metres above MSL)
# tide_file = os.path.join(os.getcwd(), 'example', 'tides.csv')
# tides_sat, output_corrected = SDS_islands.process_tide_data(tide_file, output)

# ---- Option 3: measured tide data — legacy tab-separated (WA local time, UTC+8) -
tide_file = os.path.join(os.getcwd(), 'example', 'EvaTide_2019.txt')
tides_sat, output_corrected = SDS_islands.process_tide_data(tide_file, output)

# plot tide levels at satellite acquisition times
fig, ax = plt.subplots(1, 1, figsize=(14, 4), tight_layout=True)
ax.grid(which='major', linestyle=':', color='0.5')
# Option 1 (FES2022) only: uncomment to also show continuous tide record
# ax.plot(dates_ts, tides_ts, '-', color='0.6', label='full record')
ax.plot(output['dates'], tides_sat, '-o', color='k', ms=6, mfc='w', lw=1,
        label='satellite acquisition')
ax.set(ylabel='Tide level (m)', xlim=[output['dates'][0], output['dates'][-1]],
       title='Tide levels at the time of image acquisition')
ax.legend()

# correction parameters
settings['beach_slope'] = 0.1375  # local beach face slope (tan β)
settings['zref'] = 0              # reference datum height (m above MSL)

# tidally correct cross-shore distances and rebuild sand polygons at reference datum
cross_distance_corrected = SDS_islands.tide_correct(
    cross_distance, tides_sat, output, transects, settings)
output_corrected = SDS_islands.tide_correct_sand_polygon(
    cross_distance_corrected, output_corrected, settings)

# %%
