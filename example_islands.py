#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings
# load modules
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
#authenticate and initialise Google Earth Engine
project_name = 'ee-coastsat-mc'  # <-- set your GEE Cloud Project name here
SDS_download.authenticate_and_initialize(project_name)

#%% 2.

# region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'example','EVA.kml'))
# or enter the coordinates (first and last pair of coordinates are the same)
# polygon = [[114.4249504953477, -21.9295184484435],
#            [114.4383556651795, -21.92949300318377],
#            [114.4388731500701, -21.91491228133647],
#            [114.4250081185656, -21.91495393621703],
#            [114.4249504953477, -21.9295184484435]]

# convert polygon to a smallest rectangle (sides parallel to coordinate axes)
polygon = SDS_tools.smallest_rectangle(polygon)
# date range
dates = ['2019-01-01', '2019-01-20']

# satellite missions
sat_list = ['L8','S2']
# name of the site
sitename = 'EVA'

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data,
        }

# # before downloading the images, check how many images are available for your inputs
#SDS_download.check_images_available(inputs);
    
#%% 2. Retrieve images
    
# only uncomment this line if you want Landsat Tier 2 images (not suitable for time-series analysis)
# inputs['include_T2'] = True

# retrieve satellite images from GEE
metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs) 

#%% 3. Batch island contour detection

# settings for the sand contour mapping
# settings for the shoreline extraction
settings = {
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'dist_clouds': 200,         # ditance around clouds where shoreline can't be mapped
    'output_epsg': 28350,       # epsg code of spatial reference system desired for the output
    # quality control:
    'check_detection': False,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 1000,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 100,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': True,  # switch this parameter to True if sand pixels are masked (in black) on many images
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    'pan_off': False,           # True to switch pansharpening off for Landsat 7/8/9 imagery
    's2cloudless_prob': 100,    # cloud probability threshold for s2cloudless (0-100)
    # add the inputs defined previously
    'inputs': inputs,
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
#SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections); required if using sand_polygon
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100        


#%%
#Note, you will need to make sure you are in your CoastSat directory to enable detection - this step requires the classification scheme from CoastSat
# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_islands.extract_sand_poly(metadata, settings)

#%%
# plot the sand polygons
fig = plt.figure()
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['sand_points'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '-', label=date.strftime('%d-%m-%Y'))
plt.legend()    

# plot time-series of island metrics (area, orientation, eccentricity)
fig, ax = plt.subplots(3,1, figsize=(15,6), tight_layout=True, sharex=True)
ax[0].plot(output['dates'],output['sand_area'],'b-x')
ax[0].grid('on')
ax[0].set(title='Sub-aerial sand area (m^2)')
ax[1].plot(output['dates'],np.array(output['sand_orientation'])*180/np.pi,'b-x')
ax[1].grid('on')
ax[1].set(title='Orientation (degrees)')
ax[2].plot(output['dates'],output['sand_eccentricity'],'b-x')
ax[2].grid('on')
ax[2].set(title='Eccentricity')

#%% 4. Shoreline/island contour analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 

# option 1: automatically compute transects from single origin point for circular islands
# angles for the transects around the island
ang_start = 0 
ang_end = 360
ang_step = 10 # degree interval for calculating transects 
settings['heading'] = np.array(list(range(ang_start,ang_end,ang_step)))
# origing of all the transects
settings['island_center'] = np.mean(settings['reference_shoreline'], axis=0).tolist()
# To get the correct coordinates run:  np.mean(settings['reference_shoreline'], axis=0)
# length of each transect from the origin
settings['transect_length'] = 400
# generate transects automatically
transects = SDS_islands.calc_island_transects(settings)
    
# plot transects 
fig = plt.figure(figsize=(15,6), tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['sand_points'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '-', label=date.strftime('%d-%m-%Y'))
plt.legend()
for key in transects.keys():
    plt.plot(transects[key][:,0], transects[key][:,1], 'k-')
plt.plot(settings['island_center'][0],settings['island_center'][1],'ro')

# option 2: draw transects manually (if island is not circular)
# transects = SDS_transects.draw_transects(output, settings)

# intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
settings['along_dist'] = 10
cross_distance = SDS_islands.compute_intersection_islands(output, transects, settings)            

#%% 5. Tidal correction

# ---- Option 1: FES2022 global tide model (recommended) -------------------------
# Setup instructions: https://github.com/kvos/CoastSat/blob/master/doc/FES2022_setup.md
# from coastsat import SDS_slope
# import pyfes
# # import json  # needed for Option 1b (latitude-clipped model) below

# # centroid of the ROI polygon in WGS84 (lon, lat); FES uses 0–360 longitudes
# centroid = np.mean(polygon[0], axis=0)
# if centroid[0] < 0: centroid[0] += 360

# # Option 1a: load full global model (requires ~8 GB RAM)
# filepath_fes = os.path.join(os.pardir, 'CoastSat.webgis', 'aviso-fes-main', 'data', 'fes2022b')
# selected_yaml = os.path.join(filepath_fes, 'fes2022.yaml')
# # # Option 1b: latitude-clipped model (1–2 GB RAM)
# # fp_band_configs = os.path.join(os.getcwd(), 'examples', 'tide_model_clipping',
# #                                'fes2022_by_latitude', 'band_configs.json')
# # with open(fp_band_configs, 'r') as f: band_configs = json.load(f)
# # selected_yaml = SDS_slope.select_yaml_for_centroid(centroid, band_configs)['yaml']

# # load pyfes (supports both pre-2026 and 2026+ API)
# if int(pyfes.__version__.split('.')[0]) < 2026:
#     handlers = pyfes.load_config(selected_yaml)
#     ocean_tide, load_tide = handlers['tide'], handlers['radial']
# else:
#     config = pyfes.config.load(selected_yaml)
#     ocean_tide, load_tide = config.models['tide'], config.models['radial']

# # compute tides: full time-series (15-min) and at satellite acquisition times
# date_range = [pytz.utc.localize(datetime(1984, 1, 1)), pytz.utc.localize(datetime(2030, 1, 1))]
# dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, 900, ocean_tide, load_tide)
# tides_sat = SDS_slope.compute_tide_dates(centroid, output['dates'], ocean_tide, load_tide)
# # package for tide_correct_sand_polygon (FES has no gaps so output_corrected == output)
# output_corrected = dict(output)
# output_corrected['tide'] = list(tides_sat)

# ---- Option 2: measured tide data — CoastSat standard UTC CSV -------------------
# CSV must have columns: 'dates' (UTC, pandas-parseable) and 'tide' (metres above MSL)
# tide_file = os.path.join(os.getcwd(), 'example', 'tides.csv')
# tides_sat, output_corrected = SDS_islands.process_tide_data(tide_file, output)

# ---- Option 3: measured tide data — legacy tab-separated (WA local time, UTC+8) -
tide_file = os.path.join(os.getcwd(), 'example', 'EvaTide_2019.txt')
tides_sat, output_corrected = SDS_islands.process_tide_data(tide_file, output)

# plot tide levels at satellite acquisition times (matches v3.3 style)
fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
ax.grid(which='major', linestyle=':', color='0.5')
# Option 1 (FES2022) only: uncomment to also show full continuous tide record
# ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
ax.plot(output['dates'], tides_sat, '-o', color='k', ms=6, mfc='w', lw=1, label='image acquisition')
ax.set(ylabel='tide level [m]', xlim=[output['dates'][0], output['dates'][-1]],
       title='Tide levels at the time of image acquisition')
ax.legend()

# define correction parameters
settings['beach_slope'] = 0.1375  # local beach face slope (tan β)
settings['zref'] = 0              # reference elevation (m above MSL)
# tidally correct the cross-shore distance time-series
cross_distance_corrected = SDS_islands.tide_correct(cross_distance, tides_sat, output, transects, settings)
# tidally correct the sand polygon output
output_corrected = SDS_islands.tide_correct_sand_polygon(cross_distance_corrected, output_corrected, settings)
# %%
