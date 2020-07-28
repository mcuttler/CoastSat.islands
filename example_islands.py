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
from coastsat import SDS_islands, SDS_download, SDS_preprocess, SDS_tools, SDS_transects

# region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'example','EVA.kml'))
# or enter the coordinates (first and last pair of coordinates are the same)
# polygon = [[114.4249504953477, -21.9295184484435],
#            [114.4383556651795, -21.92949300318377],
#            [114.4388731500701, -21.91491228133647],
#            [114.4250081185656, -21.91495393621703],
#            [114.4249504953477, -21.9295184484435]]

# date range
dates = ['2019-01-01', '2019-02-01']

# satellite missions
sat_list = ['S2']

# name of the site
sitename = 'EVA'

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data')

# island file - contains the coordinates of the island centroid and the beach slope
island_file = os.path.join(os.getcwd(), 'example',sitename + '_info.csv')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data,
    'island_file': island_file
        }
    
#%% 2. Retrieve images
    
# retrieve satellite images from GEE
metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs)   

    
#%% 3. Batch island contour detection
    
# settings for the sand contour mapping
settings = { 
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'output_epsg': 3857,        # epsg code of spatial reference system desired for the output
    # quality control:        
    'check_detection_sand_poly': True, # if True, uses sand polygon for detection and shows user for validation 
    'save_figure': True,               # if True, saves a figure showing the mapped shoreline for each image
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 50,       # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 100,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images
    'sand_color': 'default',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
}

#read additional settings for island info - adds:
settings = SDS_islands.read_island_info(inputs['island_file'],settings)

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections); required if using sand_polygon
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100        

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_islands.extract_sand_poly(metadata, settings)

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
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    

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

# option 1: draw transects
transects = SDS_transects.draw_transects(output, settings)

# option 2: automatically compute transects from single origin point for circular islands
ang_start = 0 
ang_end = 360
ang_step = 10 #degree step for calculating transects 
settings['heading'] = np.array(list(range(ang_start,ang_end,ang_step)))
settings['transect_length'] = 400
# transects = SDS_islands.calc_island_transects(settings)
    
# intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
settings['along_dist'] = 10
cross_distance = SDS_islands.compute_intersection_islands(output, transects, settings)            

#%% 5. Tidal correction
    
#process tide data
#input tide data is in local time (Australian West Coast, UTC +8 hrs), but code below converts to UTC
tide_file = os.path.join(os.getcwd(),'example','EvaTide_2019.txt')
tide, output_corrected = SDS_islands.process_tide_data(tide_file, output)    

# define reference height datum for tidal correction
settings['zref'] = 0      
# tidally correct the time-series of shoreline change along the transects              
cross_distance_corrected = SDS_islands.tide_correct(cross_distance,tide, output, transects, settings)
   
# also a function to tidally corrected the sand polygons     
output_corrected = SDS_islands.tide_correct_sand_polygon(cross_distance_corrected, output_corrected, settings)