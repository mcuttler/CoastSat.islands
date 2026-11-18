"""
example_islands_veg.py
======================
Example script for running CoastSat.islands with **combined** sand polygon
and vegetation polygon extraction.

Workflow
--------
1. Authenticate with Google Earth Engine.
2. Define the site and download/load imagery metadata.
3. Run sand polygon extraction  (SDS_islands.extract_sand_poly).
4. Run vegetation extraction    (SDS_vegline.extract_veglines_islands).
5. Plot and compare results.

Branch: feature/veg-boundary
"""

#%% 1. Imports and GEE authentication
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from coastsat import SDS_islands, SDS_download, SDS_preprocess, SDS_tools, SDS_transects
from coastsat import SDS_vegline                   # <-- NEW: vegetation module

%matplotlib qt

# ---------------------------------------------------------------------------
# Authenticate Google Earth Engine
# ---------------------------------------------------------------------------
project_name = 'your-gee-project'  # <-- set your GEE Cloud Project name here
SDS_download.authenticate_and_initialize(project_name)


#%% 2. Site definition and inputs

# Region of interest — load from KML or enter coordinates manually
polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'example', 'EVA.kml'))
# polygon = [[114.4249, -21.9295],
#            [114.4384, -21.9295],
#            [114.4389, -21.9149],
#            [114.4250, -21.9149],
#            [114.4249, -21.9295]]

polygon = SDS_tools.smallest_rectangle(polygon)

dates    = ['2019-01-01', '2022-12-31']
sat_list = ['S2']          # Sentinel-2; also supports 'L8', 'L9', etc.
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

# Download imagery from GEE (skip if already downloaded)
# metadata = SDS_download.retrieve_images(inputs)

# Load existing metadata
metadata = SDS_download.get_metadata(inputs)


#%% 3. Settings

settings = {
    # --- general ---
    'cloud_thresh': 0.5,
    'dist_clouds': 0,
    'output_epsg': 28350,           # projected CRS for the site (metres)
    # --- quality control ---
    'check_detection': False,
    'adjust_detection': False,
    'save_figure': True,
    # --- sand detection (SDS_islands) ---
    'min_beach_area': 1000,         # minimum sand area (m²)
    'buffer_size': 100,             # buffer around sand pixels (m)
    'min_length_sl': 500,           # minimum shoreline length (m)
    'cloud_mask_issue': True,
    'sand_color': 'bright',
    'pan_off': False,
    # --- vegetation detection (SDS_vegline) ---
    'min_veg_area': 500,            # minimum vegetation area (pixels)
    'veg_thresh': 'auto',           # 'auto' = Otsu; or set a fixed float e.g. 0.25
    # --- inputs reference ---
    'inputs': inputs,
}

# Optional: create/load a reference shoreline for sand detection
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
settings['max_dist_ref'] = 100


#%% 4. Sand polygon extraction

print('\n=== Sand polygon extraction ===')
output_sand = SDS_islands.extract_sand_poly(metadata, settings)

# Save sand output
filepath_site = os.path.join(inputs['filepath'], sitename)
os.makedirs(filepath_site, exist_ok=True)
with open(os.path.join(filepath_site, sitename + '_output_sand.pkl'), 'wb') as f:
    pickle.dump(output_sand, f)
print('Sand output saved.')


#%% 5. Vegetation polygon extraction

print('\n=== Vegetation polygon extraction ===')
output_veg = SDS_vegline.extract_veglines_islands(metadata, settings)

# Save vegetation output
with open(os.path.join(filepath_site, sitename + '_output_veg.pkl'), 'wb') as f:
    pickle.dump(output_veg, f)
print('Vegetation output saved.')


#%% 6. Quick visualisation — overlay sand and veg polygons

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_title(f'{sitename} — sand (tan) and vegetation (green) polygons')
ax.grid(linestyle=':', color='0.5')

# Plot sand polygons
for pts in output_sand.get('sand_points', []):
    if pts is not None:
        ax.fill(pts[:, 0], pts[:, 1], color='tan', alpha=0.3)
        ax.plot(pts[:, 0], pts[:, 1], '-', color='peru', lw=0.6)

# Plot vegetation polygons (all satellites combined)
for satname in output_veg:
    for pts in output_veg[satname]['veg_polygon_coords']:
        if pts is not None:
            ax.fill(pts[:, 0], pts[:, 1], color='forestgreen', alpha=0.3)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='darkgreen', lw=0.6)

plt.tight_layout()


#%% 7. Time-series comparison — sand area vs vegetation area

# Collect dates and areas from all satellites
dates_veg, area_veg = [], []
for satname in output_veg:
    dates_veg.extend(output_veg[satname]['dates'])
    area_veg.extend(output_veg[satname]['veg_area'])

# Sort by date
sort_idx = np.argsort(dates_veg)
dates_veg = [dates_veg[i] for i in sort_idx]
area_veg   = [area_veg[i]  for i in sort_idx]

fig, axes = plt.subplots(2, 1, figsize=(14, 6), tight_layout=True, sharex=True)

axes[0].plot(output_sand['dates'], output_sand['sand_area'], 'b-o',
             ms=3, label='Sand area')
axes[0].set_ylabel('Area (m²)')
axes[0].set_title('Sub-aerial sand area')
axes[0].grid(True, linestyle=':')

axes[1].plot(dates_veg, area_veg, 'g-o', ms=3, label='Vegetation area')
axes[1].set_ylabel('Area (m²)')
axes[1].set_title('Vegetation area (NDVI-based)')
axes[1].grid(True, linestyle=':')

plt.suptitle(sitename, fontweight='bold')


#%% 8. Load saved results (if re-running)

# with open(os.path.join(filepath_site, sitename + '_output_sand.pkl'), 'rb') as f:
#     output_sand = pickle.load(f)
# with open(os.path.join(filepath_site, sitename + '_output_veg.pkl'), 'rb') as f:
#     output_veg = pickle.load(f)
