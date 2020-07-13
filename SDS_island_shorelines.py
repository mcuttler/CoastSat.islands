"""This module contains all the functions needed for extracting satellite-derived shorelines (SDS)

   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

# machine learning modules
from sklearn.externals import joblib
from shapely.geometry import LineString, LinearRing, Polygon
from shapely import ops
# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import gridspec
from pylab import ginput
import pickle
#import simplekml

# own modules
import SDS_island_tools, SDS_preprocess

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans


def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
        im1, im2: np.array
            Images (2D) with which to calculate the ND index
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are

    Returns:    -----------
        im_nd: np.array
            Image (2D) containing the ND index
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates a range of features on the image that are used for the supervised classification.
    The features include spectral normalized-difference indices and standard deviation of the image.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_bool: np.array
            2D array of boolean indicating where on the image to calculate the features

    Returns:    -----------
        features: np.array
            matrix containing each feature (columns) calculated for all
            the pixels (rows) indicated in im_bool
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  SDS_island_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_island_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_island_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_island_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_island_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_island_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features
def classify_image_NN(im_ms, im_extra, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network previously trained.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            Pansharpened RGB + downsampled NIR and SWIR
        im_extra:
            only used for Landsat 7 and 8 where im_extra is the panchromatic band
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        min_beach_area: int
            minimum number of pixels that have to be connected to belong to the SAND class
        clf: classifier

    Returns:    -----------
        im_classif: np.array
            2D image containing labels
        im_labels: np.array of booleans
            3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)

    return im_classif, im_labels

def find_wl_contours1(im_ndwi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shorelien detection.
    Finds the water line by thresholding the Normalized Difference Water Index and applying
    the Marching Squares Algorithm to contour the iso-value corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
        im_ndwi: np.ndarray
            Image (2D) with the NDWI (water index)
        cloud_mask: np.ndarray
            2D cloud mask with True where cloud pixels are
        im_ref_buffer: np.array
            Binary image containing a buffer around the reference shoreline

    Returns:    -----------
        contours_wl: list of np.arrays
            contains the (row,column) coordinates of the contour lines

    """

    # reshape image to vector
    vec_ndwi = im_ndwi.reshape(im_ndwi.shape[0] * im_ndwi.shape[1])
    vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec = vec_ndwi[~vec_mask]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndwi_buffer = np.copy(im_ndwi)
    im_ndwi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndwi_buffer, t_otsu)

    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours = contours_nonans

    return contours

def find_wl_contours2(im_ms, im_labels, cloud_mask, buffer_size, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification component to
    refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        buffer_size: int
            size of the buffer around the sandy beach over which the pixels are considered in the
            thresholding algorithm.
        im_ref_buffer: np.array
            Binary image containing a buffer around the reference shoreline

    Returns:    -----------
        contours_wi: list of np.arrays
            contains the (row,column) coordinates of the contour lines extracted from the
            NDWI (Normalized Difference Water Index) image
        contours_mwi: list of np.arrays
            contains the (row,column) coordinates of the contour lines extracted from the
            MNDWI (Modified Normalized Difference Water Index) image

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)

    # create a buffer around the sandy beach
    se = morphology.disk(buffer_size)
    im_buffer = morphology.binary_dilation(im_labels[:,:,0], se)
    vec_buffer = im_buffer.reshape(nrows*ncols)

    # select water/sand/swash pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
    int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
            int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_water,int_sand, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])

    # find contour with MS algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)

    # remove contour points that are NaNs (around clouds)
    contours = contours_wi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_wi = contours_nonans
    # repeat for MNDWI contours
    contours = contours_mwi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_mwi = contours_nonans

    return contours_wi, contours_mwi

def process_shoreline(contours, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. This function also removes
    the contours that are too small to be a shoreline (based on the parameter
    settings['min_length_sl'])

    KV WRL 2018

    Arguments:
    -----------
        contours: np.array or list of np.array
            image contours as detected by the function find_contours
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        settings: dict
            contains important parameters for processing the shoreline:
                output_epsg: output spatial reference system
                min_length_sl: minimum length of shoreline perimeter to be kept (in meters)

    Returns:    -----------
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline

    """

    # convert pixel coordinates to world coordinates
    contours_world = SDS_island_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_island_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    shoreline = contours_array

    return shoreline

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
               bbox_to_anchor=(1, 0.5), fontsize=10)
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
    

def extract_shorelines(metadata, settings):
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
        filepath = SDS_island_tools.get_filepath(settings['inputs'],satname)
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
        if satname in ['L5','L7','L8']:
            if settings['dark_sand']:
                clf = joblib.load(os.path.join(os.getcwd(), 'classification','models', 'NN_4classes_Landsat_dark.pkl'))
            else:
                clf = joblib.load(os.path.join(os.getcwd(), 'classification','models', 'NN_4classes_Landsat.pkl'))
            pixel_size = 15
        elif satname == 'S2':
            clf = joblib.load(os.path.join(os.getcwd(), 'classification','models', 'NN_4classes_S2.pkl'))
            pixel_size = 10
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        if 'reference_shoreline' in settings.keys():
            max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
        # loop through the images
        for i in range(len(filenames)):
            print(filenames[i])
            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

            # get image filename
            fn = SDS_island_tools.get_filenames(filenames[i],filepath, satname)
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
            im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask,
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
                    sand_contours_world = SDS_island_tools.convert_pix2world(sand_contours[0],georef)
                    sand_contours_coords = SDS_island_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                    # make a shapely polygon 
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                else:    
                    # convert to world coordinates
                    sand_contours_world = SDS_island_tools.convert_pix2world(sand_contours[0],georef)
                    sand_contours_coords = SDS_island_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                    # make a shapely polygon
                    if len(sand_contours_coords)>=3:
                        linear_ring = LinearRing(coordinates=sand_contours_coords[~np.isnan(sand_contours_coords[:,0])])
                        sand_polygon = Polygon(shell=linear_ring, holes=None)
                                      
            else:    
                # convert to world coordinates
                sand_contours_world = SDS_island_tools.convert_pix2world(sand_contours[0],georef)
                sand_contours_coords = SDS_island_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
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
                sand_contours_world = SDS_island_tools.convert_pix2world(sand_contours[0],georef)
                sand_contours_coords = SDS_island_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
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
                ref_sl_pix = SDS_island_tools.convert_world2pix(SDS_island_tools.convert_epsg(ref_sl, settings['output_epsg'],
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
                    im_mndwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contours on MNDWI grayscale image
                    contours_mwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                else:
                    # use classification to refine threshold and extract sand/water interface
                    contours_wi, contours_mwi = find_wl_contours2(im_ms, im_labels,
                                                cloud_mask, buffer_size_pixels, im_ref_buffer)
            except:
                print('Could not map shoreline for this image: ' + filenames[i])
                continue

            # process water contours into shorelines
            shoreline = process_shoreline(contours_mwi, georef, image_epsg, settings)
            
            if settings['check_detection_sand_poly']:
                date = filenames[i][:18]
                skip_image = show_detection_sand_poly(im_ms, cloud_mask, im_labels, im_binary_sand, im_binary_sand_closed, 
                                                      sand_contours, settings, date, satname, regions)
                # if user decides to skip the image
                if skip_image:
                    continue
                
         
            # visualise the mapped shorelines, there are two options:
            # if settings['check_detection'] = True, show the detection to the user for accept/reject
            # if settings['save_figure'] = True, save a figure for each mapped shoreline
            elif settings['check_detection'] or settings['save_figure']:
                date = filenames[i][:19]
                skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                            image_epsg, georef, settings, date, satname)
                # if the user decides to skip the image, continue and do not save the mapped shoreline
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

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_island_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)        
    
    # save output into a gdb.GeoDataFrame
    gdf = SDS_island_tools.output_to_gdf(output)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson
    gdf.to_file(os.path.join(filepath, sitename + '_output.geojson'), driver='GeoJSON', encoding='utf-8')
        
    return output
