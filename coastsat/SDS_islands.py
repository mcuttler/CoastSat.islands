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
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

# coastsat modules (from main toolbox)
from coastsat import SDS_shoreline, SDS_preprocess, SDS_tools


###################################################################################################
# EXTRACTION OF ISLAND CONTOURS AS SAND POLYGONS
###################################################################################################

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
        # load classifiers (if sklearn version above 0.20, learn the new files)
        str_new = ''
        if not sklearn.__version__[:4] == '0.20':
            str_new = '_new'
        if satname in ['L5','L7','L8','L9']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark%s.pkl'%str_new))
            elif settings['sand_color'] == 'bright':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright%s.pkl'%str_new))
            elif settings['sand_color'] == 'default':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat%s.pkl'%str_new))
            elif settings['sand_color'] == 'latest':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_latest%s.pkl'%str_new))   
        elif satname == 'S2':
            pixel_size = 10
            clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_S2%s.pkl'%str_new))

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
            im_ms, georef, cloud_mask, im_extra, imQA, im_nodata = SDS_preprocess.preprocess_single(fn, satname,
                                                                                                    settings['cloud_mask_issue'],
                                                                                                    settings['pan_off'],
                                                                                                    settings['s2cloudless_prob'])
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]
            # v3.3-style two-step cloud cover (matches extract_shorelines)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                             (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99:
                continue
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            if cloud_cover > settings['cloud_thresh']:
                continue

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, cloud_mask,
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
                    sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])               
                    # make a shapely polygon 
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                else:    
                    # convert to world coordinates
                    sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                    sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])               
                    # make a shapely polygon
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                                      
            else:    
                # convert to world coordinates
                sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
                sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])               
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
                sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])               
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
                                                                                image_epsg), georef)
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
                    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contours on MNDWI grayscale image
                    contours_mwi, t_mwi = SDS_shoreline.find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                else:
                    # use classification to refine threshold and extract sand/water interface
                    # v3.3: find_wl_contours2 returns (contours_mwi, t_mwi), no buffer_size_pixels param
                    contours_mwi, t_mwi = SDS_shoreline.find_wl_contours2(im_ms, im_labels,
                                                cloud_mask, im_ref_buffer)
            except Exception as e:
                print('Could not map shoreline for this image: ' + filenames[i] + '  [' + str(e) + ']')
                continue

            # process water contours into shorelines
            # v3.3: process_shoreline now takes cloud_mask_adv and im_nodata
            shoreline = SDS_shoreline.process_shoreline(contours_mwi, cloud_mask_adv, im_nodata, georef, image_epsg, settings)
            
            # v3.3-aligned: use standard 'check_detection' and 'save_figure' keys.
            # 'check_detection_sand_poly' has been retired — populate settings with
            # check_detection=True (interactive) and/or save_figure=True (auto-save jpg).
            if settings.get('check_detection', False) or settings.get('save_figure', False):
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
    gdf = SDS_tools.output_to_gdf(output, geomtype='lines')
    if gdf is not None:
        # set projection
        gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
        # save as geojson
        gdf.to_file(os.path.join(filepath, sitename + '_output.geojson'), driver='GeoJSON', encoding='utf-8')
        
    return output


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
    # Use plt.colormaps (matplotlib >=3.5); cm.get_cmap removed in 3.9,
    # cm.colormaps not reliably available as a submodule attribute.
    cmap = plt.colormaps['tab20c']
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
               loc='lower right', fontsize=10)
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

    # if check_detection is True, let user manually accept/reject the images.
    # Uses standard CoastSat 'check_detection' key (not deprecated 'check_detection_sand_poly').
    skip_image = False
    if settings.get('check_detection', False):

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
    

###################################################################################################
# SHORE-NORMAL TRANSECTS FOR ISLANDS
###################################################################################################

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
    # Use sand_points (from sand polygon extraction) when available; fall back to
    # standard CoastSat shorelines otherwise.  This is data-driven: no settings flag needed.
    if 'sand_points' in output and len(output['sand_points']) > 0:
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


###################################################################################################
# TIDAL CORRECTION
###################################################################################################

def process_tide_data(tide_file, output):
    """
    Extract tide heights at the times of satellite acquisitions and remove any
    images for which no tidal data is available.

    Accepts two tide file formats:
      - CoastSat standard CSV: columns 'dates' (UTC, parseable by pandas) and 'tide'
      - Legacy tab-separated: columns [year month day hour min sec z] in WA local
        time (UTC+8); detected automatically when the file does not end in '.csv'

    Uses SDS_tools.get_closest_datapoint for the time-matching lookup, consistent
    with the approach used throughout CoastSat.

    KV WRL 2020 (adapted for CoastSat.islands MC 2019)

    Arguments:
    -----------
    tide_file: str
        full path to the tide data file (CSV or legacy tab-separated)
    output: dict
        output dictionary from extract_sand_poly

    Returns:
    -----------
    tide_sat: np.array
        tide height at each satellite acquisition date (NaN where no data)
    output_corrected: dict
        copy of output filtered to only the images that have valid tide data,
        with an additional 'tide' key containing the matched tide heights
    """
    # --- read tide file ---
    if tide_file.endswith('.csv'):
        # CoastSat standard format: 'dates' (UTC) and 'tide' columns
        tide_df = pd.read_csv(tide_file, parse_dates=['dates'])
        dates_ts = [_.to_pydatetime() for _ in tide_df['dates']]
        tides_ts = np.array(tide_df['tide'])
    else:
        # Legacy tab-separated format: [year month day hour min sec z] in WA local time (UTC+8)
        tideraw = pd.read_csv(tide_file, sep='\t').values
        dates_ts = []
        tides_ts = []
        for row in tideraw:
            t = datetime(int(row[0]), int(row[1]), int(row[2]),
                         int(row[3]), int(row[4]), int(row[5]))
            t = datetime.fromtimestamp(datetime.timestamp(t), tz=timezone.utc)
            dates_ts.append(t)
            tides_ts.append(row[-1])
        tides_ts = np.array(tides_ts)

    # --- match tide to each satellite acquisition ---
    tide_sat = SDS_tools.get_closest_datapoint(output['dates'], dates_ts, tides_ts)

    # --- filter output to images that have valid tide data ---
    idx_valid = np.where(~np.isnan(tide_sat))[0]
    output_corrected = {key: [output[key][i] for i in idx_valid]
                        for key in output.keys()}
    output_corrected['tide'] = [tide_sat[i] for i in idx_valid]

    return tide_sat, output_corrected


def tide_correct(cross_distance, tide, output, transects, settings):
    """
    Tidally correct cross-shore distances using the linear slope method:

        correction = (zref - tide) / beach_slope
        cross_distance_corrected = cross_distance + correction

    where zref is the reference datum height (e.g. 0 m AHD) and beach_slope is
    positive (the sign convention is handled internally so that a tide above zref
    shifts the shoreline seaward and vice versa).

    Consistent with CoastSat v3.3 (KV WRL 2020), adapted for CoastSat.islands (MC 2019).

    Arguments:
    -----------
    cross_distance: dict
        cross-shore distances along transects from compute_intersection_islands
    tide: np.array
        tide heights at satellite acquisition times (from process_tide_data)
    output: dict
        output dict (dates used when writing the CSV)
    transects: dict
        transect definitions
    settings: dict
        must contain 'beach_slope' (positive float) and 'zref' (reference datum, m)

    Returns:
    -----------
    cross_distance_corrected: dict
        tidally corrected cross-shore distances, one array per transect key
    """
    beach_slope = abs(settings['beach_slope'])   # enforce positive; sign handled below
    zref = settings['zref']
    tide = np.array(tide)

    # sanity check
    n_obs = len(output['dates'])
    if len(tide) != n_obs:
        raise ValueError(
            f'Length mismatch: {n_obs} dates in output but {len(tide)} tide values. '
            'Make sure to run process_tide_data before tide_correct.')

    # vectorised correction across all transects
    # positive correction = seaward shift (tide below zref moves shoreline seaward)
    correction = (tide - zref) / beach_slope
    cross_distance_corrected = {}
    for key in cross_distance.keys():
        cross_distance_corrected[key] = cross_distance[key] + correction

    # save to CSV
    out_dict = {'dates': output['dates']}
    for key in transects.keys():
        out_dict['Transect ' + key] = cross_distance_corrected[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'],
                      'transect_time_series_corrected.csv')
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s' % fn)

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


###################################################################################################
# UTILITIES
###################################################################################################

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