"""This module contains all the functions for CoastSat.islands

   Author: Mike Cuttler
"""

# load modules
import os
import numpy as np
import pandas as pd
import math
import pickle

# plotting modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from pylab import ginput

# geospatial
from shapely.geometry import LineString, LinearRing, Polygon, MultiPoint
from shapely import ops
from datetime import datetime, timezone
import geopandas as gpd

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

# machine learning modules
from sklearn.externals import joblib
from shapely.geometry import LineString

# coastsat modules (from main toolbox)
from coastsat import SDS_shoreline, SDS_preprocess, SDS_tools


def read_island_info(island_file,settings):
    """
    To be filled
    MC
    
    """
    
    island_info = pd.read_csv(island_file)
    island_info = island_info.values
    
    settings['island_center'] = island_info[0,0:2]
    settings['beach_slope'] = island_info[0,2]
    
    return settings 


def show_detection_sand_poly(im_ms, cloud_mask, im_labels, im_binary_sand, im_binary_sand_closed, sand_contours,
                   settings, date, satname, regions):
    """
    Shows the detected sand polygons and boundary of this polygon (pseudo shoreline) to the user for visual quality control. The user can select "keep"
    if the shoreline detection is correct or "skip" if it is incorrect. 
    
    KV WRL 2019
    MC UWA 2019
    
    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        shoreline: np.array 
            array of points with the X and Y coordinates of the shoreline
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        settings: dict
            contains important parameters for processing the shoreline
        date: string
            date at which the image was taken
        satname: string
            indicates the satname (L5,L7,L8 or S2)
                       
    Returns:    -----------
        skip_image: boolean
            True if the user wants to skip the image, False otherwise.
            
    """  
    
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection 
    filepath_sandpoly = os.path.join(filepath_data, sitename, 'jpg_files', 'sand_polygons')
    if not os.path.exists(filepath_sandpoly):      
        os.makedirs(filepath_sandpoly)
        
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    
    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2,0], sharex=ax1, sharey=ax1)
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
                                         
    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
    ax1.set_title(sitename + '    ' + date + '     ' + satname, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    im_class = np.copy(im_RGB)
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]
        
    ax2.imshow(im_class)    
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               loc='bottom right', fontsize=10)
    ax2.set_title(date, fontweight='bold', fontsize=16)
    
    # create image 3 (closed sand polygon)
    ax3.imshow(im_RGB)
    for props in regions:                       
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5* props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5* props.major_axis_length
        x2 = x0 - math.cos(orientation) * 0.5* props.major_axis_length
        y2 = y0 + math.sin(orientation) * 0.5* props.major_axis_length
        plt.plot((x0, x1), (y0, y1), '-b', linewidth=2.5)
        plt.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
        plt.plot(x0, y0, '.r', markersize=15)
        # plt.text(50,25,'Orientation = ' + str(round(np.degrees(sand_orientation),2)), color = 'white',fontweight = 'bold')
    ax3.set_title('Major Axis Orientation')
    
    #plot sand contours on each sub plot
    for k in range(len(sand_contours)):
#                ax3.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'r-', linewidth=2.5)
                ax2.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'r-', linewidth=2.5)
                ax1.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'k--', linewidth=1.5)                

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection_sand_poly']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath_sandpoly, date + '_' + satname + '.jpg'), dpi=150)

    # Don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()       
   
    return skip_image
    

def extract_sand_poly(metadata, settings):
    """
    Extracts shorelines from satellite images.

    KV WRL 2018

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded

        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        buffer_size: int
            size of the buffer (m) around the sandy beach over which the pixels are considered in the
            thresholding algorithm
        min_beach_area: int
            minimum allowable object area (in metres^2) for the class 'sand'
        cloud_thresh: float
            value between 0 and 1 defining the maximum percentage of cloud cover allowed in the images
        output_epsg: int
            output spatial reference system as EPSG code
        check_detection: boolean
            True to show each invidual detection and let the user validate the mapped shoreline

    Returns:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):      
            os.makedirs(filepath_jpg)
    
    print('Mapping shorelines:')

    # loop through satellite list
    for satname in metadata.keys():

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # initialise some variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = []   # filename of the images from which the shorelines where derived
        output_cloudcover = [] # cloud cover of the images
        output_geoaccuracy = []# georeferencing accuracy of the images
        output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
        # sand fields    
        output_sand_area = []       #area of sandy pixles identified from classification 
        output_sand_perimeter = []  #perimieter of sandy pixels
        output_sand_centroid = []   #coordinates center of mass of sandy pixels
        output_sand_points = []     #coordinates of sandy pixels
        output_sand_eccentricity = [] #measure of sand area eccentricity
        output_sand_orientation = []  #orientation of major axis        

        # load classifiers and convert settings['min_beach_area'] and settings['buffer_size']
        # from metres to pixels
        # load classifiers and
        if satname in ['L5','L7','L8']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark.pkl'))
            elif settings['sand_color'] == 'bright':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright.pkl'))
            else:
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat.pkl'))

        elif satname == 'S2':
            pixel_size = 10
            clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_S2.pkl'))

        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        
        if 'reference_shoreline' in settings.keys():
            max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
            
        # loop through the images
        for i in range(len(filenames)):
            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

            # get image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, imQA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, clf)
            # if the classifier does not detect sand pixels skip this image
            if sum(sum(im_labels[:,:,0])) == 0:
                continue

            #######################################################################################
            # SAND POLYGONS (kilian)
            #######################################################################################
            #######################################################################################

            # create binary image with True where the sand pixels
            im_binary_sand = (im_classif == 1)
            # fill the interior of the ring of sand around the island
            im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
            # vectorise the contours
            if satname == 'S2':
                thresh = 0.5
            else:
                thresh = 0.75
            
            sand_contours = measure.find_contours(im_binary_sand_closed, thresh)
            
            # if several contours, it means there is a gap --> merge sand and non-classified pixels
            if len(sand_contours) > 1:
                im_binary_sand = np.logical_or(im_classif == 1, im_classif == 0)
                im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
                if satname == 'S2':
                    thresh = 0.5
                else: 
                    thresh = 0.75
                    
                sand_contours = measure.find_contours(im_binary_sand_closed, thresh)
                # if there are still more than one contour, only keep the one with more points
                if len(sand_contours) > 1:
                    n_points = []
                    for j in range(len(sand_contours)):
                        n_points.append(sand_contours[j].shape[0])
                    sand_contours = [sand_contours[np.argmax(n_points)]]
                    
                    # convert to world coordinates
                    sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                    sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                    # make a shapely polygon 
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                else:    
                    # convert to world coordinates
                    sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                    sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                    # make a shapely polygon
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                                      
            else:    
                # convert to world coordinates
                sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                # make a shapely polygon
                if len(sand_contours_coords)>=3:
                    linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                    sand_polygon = Polygon(shell=linear_ring, holes=None)
            
            # check if perimeter of polygon matches with reference shoreline
            # if much longer (1.5 times) then also merge sand and non-classified pixels
            if linear_ring.length > 1.5*LineString(settings['reference_shoreline']).length:
                im_binary_sand = np.logical_or(im_classif == 1, im_classif == 0)
                im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
                sand_contours = measure.find_contours(im_binary_sand_closed, 0.5)
                # if there are still more than one contour, only keep the one with more points
                if len(sand_contours) > 1:
                    n_points = []
                    for j in range(len(sand_contours)):
                        n_points.append(sand_contours[j].shape[0])
                    sand_contours = [sand_contours[np.argmax(n_points)]]   
                # convert to world coordinates
                sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                # make a shapely polygon
                if len(sand_contours_coords)>=3:
                    linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                    sand_polygon = Polygon(shell=linear_ring, holes=None)
                
            # calculate the attributes of sand polygon
            sand_area = sand_polygon.area
            sand_perimeter = sand_polygon.exterior.length
            sand_centroid = np.array(sand_polygon.centroid.coords)
            sand_points = np.array(sand_polygon.exterior.coords)
            
            label_img = measure.label(im_binary_sand_closed)
            regions = measure.regionprops(label_img)
            #only keep largest region (there should only be one)
            if len(regions)>1:
                bbox = []
                for j in range(len(regions)):
                    bbox.append(regions[j]['bbox_area'])
                sand_eccentricity = regions[np.argmax(bbox)]['eccentricity']
                sand_orientation = regions[np.argmax(bbox)]['orientation']
            else:                    
                sand_eccentricity = regions[0]['eccentricity']
                sand_orientation = regions[0]['orientation']
            #######################################################################################
            #######################################################################################            
            
            # if a reference shoreline is provided, only map the contours that are within a distance
            # of the reference shoreline. For this, first create a buffer around the ref shoreline
            im_ref_buffer = np.ones(cloud_mask.shape).astype(bool)
            if 'reference_shoreline' in settings.keys():
                ref_sl = settings['reference_shoreline']
                # convert to pixel coordinates
                ref_sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(ref_sl, settings['output_epsg'],
                                                                                image_epsg)[:,:-1], georef)
                ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
                # create binary image of the reference shoreline
                im_binary = np.zeros(cloud_mask.shape)
                for j in range(len(ref_sl_pix_rounded)):
                    im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
                im_binary = im_binary.astype(bool)
                # dilate the binary image to create a buffer around the reference shoreline
                se = morphology.disk(max_dist_ref_pixels)
                im_ref_buffer = morphology.binary_dilation(im_binary, se)

            # extract water line contours
            # if there aren't any sandy pixels, use find_wl_contours1 (traditional method),
            # otherwise use find_wl_contours2 (enhanced method with classification)
            # use try/except structure for long runs
            try:
                if sum(sum(im_labels[:,:,0])) == 0 :
                    # compute MNDWI (SWIR-Green normalized index) grayscale image
                    im_mndwi = SDS_shoreline.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contours on MNDWI grayscale image
                    contours_mwi = SDS_shoreline.find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                else:
                    # use classification to refine threshold and extract sand/water interface
                    contours_wi, contours_mwi = SDS_shoreline.find_wl_contours2(im_ms, im_labels,
                                                cloud_mask, buffer_size_pixels, im_ref_buffer)
            except:
                print('Could not map shoreline for this image: ' + filenames[i])
                continue

            # process water contours into shorelines
            shoreline = SDS_shoreline.process_shoreline(contours_mwi, cloud_mask, georef, image_epsg, settings)
            
            if settings['check_detection_sand_poly']:
                date = filenames[i][:18]
                skip_image = show_detection_sand_poly(im_ms, cloud_mask, im_labels, im_binary_sand, im_binary_sand_closed, 
                                                      sand_contours, settings, date, satname, regions)
                # if user decides to skip the image
                if skip_image:
                    continue
                
            # append to output variables
            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            # sand fields
            output_sand_area.append(sand_area)
            output_sand_perimeter.append(sand_perimeter)
            output_sand_centroid.append(sand_centroid)
            output_sand_points.append(sand_points)
            output_sand_eccentricity.append(sand_eccentricity)
            output_sand_orientation.append(sand_orientation)
            
        # create dictionnary of output
        output[satname] = {
                'dates': output_timestamp,
                'shorelines': output_shoreline,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'geoaccuracy': output_geoaccuracy,
                'idx': output_idxkeep,
                'sand_area': output_sand_area,
                'sand_perimeter': output_sand_perimeter,
                'sand_centroid': output_sand_centroid,
                'sand_points': output_sand_points,
                'sand_eccentricity': output_sand_eccentricity,
                'sand_orientation': output_sand_orientation,                
                'idx': output_idxkeep
                }
        print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)        
    
    # save output into a gdb.GeoDataFrame
    gdf = SDS_tools.output_to_gdf(output)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson
    gdf.to_file(os.path.join(filepath, sitename + '_output.geojson'), driver='GeoJSON', encoding='utf-8')
        
    return output


def calc_island_transects(settings):
    """ 
    This code is for calculating transecs radiating from a single point. It uses the 
    x,y (input) as the origin for the transects and calculates transects of given length
    and heading (clockwise from North)
    
    M Cuttler - 2019
    UWA
    
    Arguments:
    ----------
        x: int or float
            x-coordinate of transect origin
        
        y: int or float
            y-coordinate of transect origin
        
        settings: dict
            contains transect_length
        
        heading: numpy array 
            defines all headings for transects - angles provided as clockwise from North
        
    Return:
    ---------
          transects: dict
            contains the X and Y coordinates of each transect.  
           
    """   
    #create dictionary for output
    transects = dict([])
    x = settings['island_center'][0]
    y = settings['island_center'][1]
                                     
    for i,j in enumerate(settings['heading']):
        
        #calculate x and y --- could just use create_transect above
        xx = (math.sin(math.radians(j))*settings['transect_length'])+x
        yy = (math.cos(math.radians(j))*settings['transect_length'])+y
        key = str(i+1)
        transects[key] = np.array([[x, y], [xx, yy]])
        
    return transects 


def compute_intersection_islands(output, transects, settings):
    """
    Computes the intersection between the 2D mapped shorelines and the transects, to generate
    time-series of cross-shore distance along each transect.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        settings: dict
            contains parameters defining :
                along_dist: alongshore distance to caluclate the intersection (median of points 
                within this distance).      
        
    Returns:    
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally 
            corrected.
        
    """      
    #use sand_points as shoreline when using sand_polygon, else use typical CoastSat shoreline
    if settings['check_detection_sand_poly']:
        shorelines = output['sand_points']
    else:
        shorelines = output['shorelines']
    
    along_dist = settings['along_dist']
    
    # initialise variables
    chainage_mtx = np.zeros((len(shorelines),len(transects),6))
    idx_points = []
    
    for i in range(len(shorelines)):
        
        print('\rCalculating intersections: %d%%' % int((i+1)*100/len(output['dates'])), end='')

        sl = shorelines[i]
        idx_points_all = []
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            
            #####################################################################################
            # Modified KILIAN
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]
            idx_points_all.append(idx_close)        
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                chainage_mtx[i,j,:] = np.tile(np.nan,(1,6))
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                    
                # compute mean, median, max, min and std of chainage position
                n_points = len(xy_rot[0,:])
                mean_cross = np.nanmean(xy_rot[0,:])
                median_cross = np.nanmedian(xy_rot[0,:])
                max_cross = np.nanmax(xy_rot[0,:])
                min_cross = np.nanmin(xy_rot[0,:])
                std_cross = np.nanstd(xy_rot[0,:])
                # store all statistics
                chainage_mtx[i,j,:] = np.array([mean_cross, median_cross, max_cross,
                            min_cross, n_points, std_cross])
    
        # store the indices of the shoreline points that were used
        idx_points.append(idx_points_all)
     
    # format into dictionnary
    chainage = dict([])
    chainage['mean'] = chainage_mtx[:,:,0]
    chainage['median'] = chainage_mtx[:,:,1]
    chainage['max'] = chainage_mtx[:,:,2]
    chainage['min'] = chainage_mtx[:,:,3]
    chainage['npoints'] = chainage_mtx[:,:,4]
    chainage['std'] = chainage_mtx[:,:,5]
    chainage['idx_points'] = idx_points
        
    # only return value of mean, median, or max
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = chainage['median'][:,j]    
    
    #save cross_distance dictionary to CSV
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect '+ key] = cross_dist[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                      'transect_time_series.csv')
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)   
    
    return cross_dist


def process_tide_data(tide_file, output):
    """
    function for processing tide data before performing tidal correction. This code finds tide heights
    at time that correspond to shoreline detections
    
    Arguments: 
    ----------
    output: dict
        data from CoastSat analysis
        
    tide_file: full file path and name
        filepath (including name) for tidal file to process. 
        should be organized as [year month day hour min sec z]
    
    Returns:
    ---------
    tide: np.array
        contains tide height at dates correponding to detected shorelines 
    """
    output_corrected = dict([])
    
    # import data from tide file
    tideraw = pd.read_csv(tide_file, sep='\t')
    tideraw = tideraw.values

    #create tide_data dictionary    
    tide_data = {'dates': [], 'tide': []}

    for i,row in enumerate(tideraw):
        #convert tide time to UTC from WA local time (UTC+8 hrs)
        dumtime = datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
        dumtime = datetime.timestamp(dumtime)
        dumtime = datetime.fromtimestamp(dumtime,tz=timezone.utc)
        tide_data['dates'].append(dumtime)
        tide_data['tide'].append(row[-1])
    
    # extract tide heights corresponding to shoreline detections
    tide_out = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    
    for i,date in enumerate(output['dates']):
        print('\rCalculating tides: %d%%' % int((i+1)*100/len(output['dates'])), end='')
        tide_out.append(tide_data['tide'][find(min(item for item in tide_data['dates'] if item > date), tide_data['dates'])])
          
    tide_out = np.array(tide_out)
    #determine all values where no tidal data exists
    tide_nanidx = np.argwhere(~np.isnan(tide_out))
    
    #remove data from everywhere when no tidal data exists
    cloud_cover = []
    dates = []
    filename = []
    geoaccuracy = []
    idx = []
    sand_area = []
    sand_centroid = []
    sand_eccentricity = []
    sand_orientation = []
    sand_perimeter = []
    sand_points = []
    satname = []
    shorelines = []
    tide = []
    
    for i,j in enumerate(tide_nanidx):                
        cloud_cover.append(output['cloud_cover'][int(tide_nanidx[i])])
        dates.append(output['dates'][int(tide_nanidx[i])])
        filename.append(output['filename'][int(tide_nanidx[i])])
        geoaccuracy.append(output['geoaccuracy'][int(tide_nanidx[i])])
        idx.append(output['idx'][int(tide_nanidx[i])])
        sand_area.append(output['sand_area'][int(tide_nanidx[i])])
        sand_centroid.append(output['sand_centroid'][int(tide_nanidx[i])])
        sand_eccentricity.append(output['sand_eccentricity'][int(tide_nanidx[i])])
        sand_orientation.append(output['sand_orientation'][int(tide_nanidx[i])])
        sand_perimeter.append(output['sand_perimeter'][int(tide_nanidx[i])])
        sand_points.append(output['sand_points'][int(tide_nanidx[i])])
        satname.append(output['satname'][int(tide_nanidx[i])])
        shorelines.append(output['shorelines'][int(tide_nanidx[i])])
        tide.append(tide_out[int(tide_nanidx[i])])
    
    output_corrected = {'cloud_cover': cloud_cover, 
                        'dates': dates,
                        'filename': filename, 
                        'geoaccuracy': geoaccuracy,
                        'idx': idx, 
                        'sand_area': sand_area,
                        'sand_centroid': sand_centroid,
                        'sand_eccentricity': sand_eccentricity,
                        'sand_orientation': sand_orientation,
                        'sand_perimeter': sand_perimeter,
                        'sand_points': sand_points,
                        'satname': satname,
                        'shorelines': shorelines,
                        'tide': tide}
    
    return tide_out, output_corrected


def tide_correct(cross_distance, tide, output, transects, settings):
    """
    Function for tide-correcting shoreline position time series returned by SDS_transects.compute_intersection
    
    Arguments:
    -----------
        cross_distance: dict
            contains the intersection points of satellite-derived shorelines and user-defined transects
        
        tide: numpy array
            timeseries of tidal elevations 
        
        zref: int
            reference level of height datum - e.g. 0 m AHD
        
        beta: int
            beach slope
    
    Returns:
    ----------
        cross_distance_tide_corrected: dict
            contains the tide corrected shoreline-transect intersections
    """
    cross_distance_corrected = dict([])
    #Check that length of tide time series is same as SDS timeseries
    #Cross distance should have at least 1 transect
    if len(cross_distance['1'])==len(tide):
        #Cycle through all transects
        for key,transect in cross_distance.items():
            transect_corrected = []               
            for i,ztide in enumerate(tide):
                if np.isnan(ztide):
                    continue
                else:
                    #calculate horizontal correction, assume negative slope so that
                    #if reference datum is above tidal height, shoreline position is shifted
                    #landwards; if reference dateum is below tidal height, shoreline 
                    # position is shifted seawards 
                    dbeta = settings['beach_slope']
                    if dbeta<0:
                        beta = settings['beach_slope']
                    else:
                        beta = -settings['beach_slope']
                        
                    delX = (settings['zref']-ztide)/beta          
                    transect_corrected.append(transect[i]+delX)
                               
            transect_corrected = np.array(transect_corrected)
            cross_distance_corrected[key] = transect_corrected
            
        #save cross_distance dictionary to CSV
        out_dict = dict([])
        out_dict['dates'] = output['dates']
        
        for key in transects.keys():
            out_dict['Transect '+ key] = cross_distance_corrected[key]
        df = pd.DataFrame(out_dict)
        fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                      'transect_time_series_corrected.csv')
        df.to_csv(fn, sep=',')
        print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)   
    
        
    else:
        print('ERROR - time series not same lenght!')

    return cross_distance_corrected


def tide_correct_sand_polygon(cross_distance_corrected, output_corrected, settings):
    """
    To be filled in 
    MC - 2019
    
    """
    #temporary dummy output variable 
    out = dict([])
    out['xout'] = np.ndarray((len(cross_distance_corrected['1']), len(cross_distance_corrected.keys())))
    out['yout'] = np.ndarray((len(cross_distance_corrected['1']), len(cross_distance_corrected.keys())))
    
    #Calculate distance from origin to tide-corrected shoreline intersection
    x = settings['island_center'][0]
    y = settings['island_center'][1]
    
    for i,transect in enumerate(cross_distance_corrected.keys()):
        key = str(i+1)
        
        for j,sl in enumerate(cross_distance_corrected[transect]):
            out['xout'][j,i] = x+(math.sin(math.radians(settings['heading'][i]))*sl)
        
            out['yout'][j,i] = y+(math.cos(math.radians(settings['heading'][i]))*sl)
        
          
    #go through each row and create numpy array of corrected polygon points
    sand_points_corrected = []
    for i,j in enumerate(out['xout']):  
        sand_points_corrected.append(np.array([out['xout'][i,:], out['yout'][i,:]]).T)
    
    output_corrected['sand_points_corrected']=sand_points_corrected  

    #use corrected points to build a polygon and calculate centroid, perimeter and area
    #organize output 
    output_sand_area = []
    output_sand_perimeter = []
    output_sand_centroid = []
    output_sand_points_poly = []
    output_index_correct = []
 
    for i, coords in enumerate(output_corrected['sand_points_corrected']):
        #get rid of NaN values 
        coords = coords[~np.isnan(coords[:,0]),:]
        
        if len(coords)>=3:
            linear_ring = LinearRing(coordinates=coords)
            sand_polygon = Polygon(shell=linear_ring, holes=None)
    
            output_sand_area.append(sand_polygon.area)
            output_sand_perimeter.append(sand_polygon.exterior.length)
            output_sand_centroid.append(np.array(sand_polygon.centroid.coords))
            output_sand_points_poly.append(np.array(sand_polygon.exterior.coords))
            output_index_correct.append(i)

    for key in output_corrected:
        output_corrected[key] = [output_corrected[key][i] for i in output_index_correct]        
              
    output_corrected['sand_area_corrected'] = output_sand_area
    output_corrected['sand_perimeter_corrected'] = output_sand_perimeter
    output_corrected['sand_centroid_corrected'] = output_sand_centroid
    output_corrected['sand_points_poly_corrected'] = output_sand_points_poly
    
    # save outputput structure as output.pkl
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output_tide_corrected.pkl'), 'wb') as f:
        pickle.dump(output_corrected, f)
    
    # save output into a gdb.GeoDataFrame
    gdf = output_to_gdf_poly(output_corrected)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson    
    gdf.to_file(os.path.join(filepath, sitename + '_output_tide_corrected.geojson'), driver='GeoJSON', encoding='utf-8')
        
    #export output data to csv file  
    csv_path = os.path.join(filepath,sitename + '_output_tide_corrected.csv')
    data_out = pd.DataFrame.from_dict(output_corrected)
    
    data_out.to_csv(csv_path)
    
    return output_corrected


def output_to_gdf_poly(output):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    for i in range(len(output['sand_points_poly_corrected'])):
        # skip if there shoreline is empty 
        if len(output['sand_points_poly_corrected'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            coords = output['sand_points_poly_corrected'][i]
            geom = MultiPoint([(coords[_,0], coords[_,1]) for _ in range(coords.shape[0])])
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'date'] = output['dates'][i].strftime('%Y-%m-%d %H:%M:%S')
            gdf.loc[i,'satname'] = output['satname'][i]
            gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1
            
    return gdf_all