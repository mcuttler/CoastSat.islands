"""This module contains utilities to work with satellite images' 
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
   Author2: Michael Cuttler, The University of Western Australia 
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math
import pandas as pd
import pickle
from datetime import tzinfo, timedelta, datetime, timezone


# other modules
from osgeo import gdal, osr
import geopandas as gpd
import skimage.transform as transform
from scipy.ndimage.filters import uniform_filter
from shapely import geometry
from shapely.geometry import LineString, LinearRing, Polygon
from shapely import ops
from astropy.convolution import convolve

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (row,columns) to world projected coordinates
    performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.array
            array with 2 columns (rows first and columns second)
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
            
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates (row,column)
    performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.array
            array with 2 columns (rows first and columns second)
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates, first columns with row and second column with column
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.ndarray
            array with 2 columns (rows first and columns second)
        epsg_in: int
            epsg code of the spatial reference in which the input is
        epsg_out: int
            epsg code of the spatial reference in which the output will be            
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # transform points
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

        
    return points_converted

def coords_from_kml(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    -----------
        polygon: list
            coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def save_kml(coords, epsg):
    """
    Saves coordinates with specified spatial reference system into a .kml file in WGS84. 
    
    KV WRL 2018

    Arguments:
    -----------
    coords: np.array
        coordinates (2 columns) to be converted into a .kml file        
                
    Returns:    
    -----------
    Saves 'coords.kml' in the current folder.
        
    """     
    
    kml = simplekml.Kml()
    coords_wgs84 = convert_epsg(coords, epsg, 4326)
    kml.newlinestring(name='coords', coords=coords_wgs84)
    kml.save('coords.kml')
    
def get_filepath(inputs,satname):
    """
    Create filepath to the different folders containing the satellite images.
    
    KV WRL 2018

    Arguments:
    -----------
        inputs: dict 
            dictionnary that contains the following fields:
        'sitename': str
            String containig the name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted
            longitudes in the first column and latitudes in the second column
        'dates': list of str
            list that contains 2 strings with the initial and final dates in format 'yyyy-mm-dd'
            e.g. ['1987-01-01', '2018-01-01']
        'sat_list': list of str
            list that contains the names of the satellite missions to include 
            e.g. ['L5', 'L7', 'L8', 'S2']
        satname: str
            short name of the satellite mission
                
    Returns:    
    -----------
        filepath: str or list of str
            contains the filepath(s) to the folder(s) containing the satellite images
        
    """     
    
    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
        filenames_pan = os.listdir(filepath_pan)
        filenames_ms = os.listdir(filepath_ms)
        if (not len(filenames_pan) == len(filenames_ms)):
            raise 'error: not the same amount of files for pan and ms'
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 8 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
        filenames_pan = os.listdir(filepath_pan)
        filenames_ms = os.listdir(filepath_ms)
        if (not len(filenames_pan) == len(filenames_ms)):
            raise 'error: not the same amount of files for pan and ms'
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(filepath_data, sitename, satname, '10m')
        filenames10 = os.listdir(filepath10)
        filepath20 = os.path.join(filepath_data, sitename, satname, '20m')
        filenames20 = os.listdir(filepath20)
        filepath60 = os.path.join(filepath_data, sitename, satname, '60m')
        filenames60 = os.listdir(filepath60)
        if (not len(filenames10) == len(filenames20)) or (not len(filenames20) == len(filenames60)):
            raise 'error: not the same amount of files for 10, 20 and 60 m bands'
        filepath = [filepath10, filepath20, filepath60]
            
    return filepath
    
def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.
    
    KV WRL 2018

    Arguments:
    -----------
        filename: str
            name of the downloaded satellite image as found in the metadata
        filepath: str or list of str
            contains the filepath(s) to the folder(s) containing the satellite images
        satname: str
            short name of the satellite mission       
        
    Returns:    
    -----------
        fn: str or list of str
            contains the filepath + filenames to access the satellite image
        
    """     
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        filename_ms = filename.replace('pan','ms')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m','20m')
        filename60 = filename.replace('10m','60m')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
        
    return fn
    
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of specified radius.
    
    Arguments:
    -----------
        image: np.array
            2D array containing the pixel intensities of a single-band image
        radius: int
            radius defining the moving window used to calculate the standard deviation. For example,
            radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
        win_std: np.array
            2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std
    win_mean = uniform_filter(image_padded, (win_rows, win_cols))
    win_sqr_mean = uniform_filter(image_padded**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std

def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.
    
    Arguments:
    -----------
        fn: str
            filepath + filename of the .tif raster
        mask: np.array
            array of boolean where True indicates the pixels that are to be masked
        
    Returns:    
    -----------
    overwrites the .tif file directly
        
    """ 
    
    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i+1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None
    
def merge_output(output):
    """
    Function to merge the output dictionnary, which has one key per satellite mission into a 
    dictionnary containing all the shorelines and dates ordered chronologically.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates, organised by satellite mission
        
    Returns:    
    -----------
        output_all: dict
            contains the extracted shorelines in a single list sorted by date
        
    """     
    
    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['satname'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname,
                  len(output[satname]['dates']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['dates'])), key=output_all['dates'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all

def tide_correct(cross_distance, tide, settings, savefile):
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
        cross_dist_corrected_coords_temp = []
        
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
    gdf = SDS_tools.output_to_gdf(output_corrected)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson    
    gdf.to_file(os.path.join(filepath, sitename + '_output_tide_corrected.geojson'), driver='GeoJSON', encoding='utf-8')
        
    #export output data to csv file  
    csv_path = os.path.join(filepath,sitename + '_output_tide_corrected.csv')
    data_out = pd.DataFrame.from_dict(output_corrected)
    
    data_out.to_csv(csv_path)
    
    return output_corrected
        
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

def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
        
    print('%d transects have been loaded' % len(transects.keys()))

    return transects

def output_to_gdf(output):
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
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            geom = geometry.LineString(output['shorelines'][i])
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

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = gdf_all.append(gdf)
            
    return gdf_all
    
   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
