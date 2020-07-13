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
    import SDS_download, SDS_preprocess
    import SDS_island_tools, SDS_island_shorelines, SDS_island_transects

    
    # region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
    polygon = SDS_island_tools.coords_from_kml(os.path.join(os.getcwd(), 'example','EVA.kml'))
                
    # date range
    dates = ['2019-01-01', '2019-02-01']
    
    # satellite missions
    sat_list = ['S2']
    
    # name of the site
    sitename = 'EVA'
    
    # filepath where data will be stored
    filepath_data = os.path.join(os.getcwd(), 'data')
    
    #island file - info about island slope and center coordinates
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
# metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs)   

    
    #%% 3. Batch shoreline detection
        
    # settings for the shoreline extraction
    settings = { 
        # general parameters:
        'cloud_thresh': 0.5,        # threshold on maximum cloud cover
        'output_epsg': 28350,       # epsg code of spatial reference system desired for the output
        # quality control:        
        'check_detection_sand_poly': True, #if True, uses sand polygon for detection and shows user for validation 
        'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
        # add the inputs defined previously
        'inputs': inputs,
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        'min_beach_area': 50,     # minimum area (in metres^2) for an object to be labelled as a beach
        'buffer_size': 100,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
        'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
        'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images
        'dark_sand': False,         # only switch to True if your site has dark sand (e.g. black sand beach)
        'zref': 0   #reference height datum for tidal correction 
    }
    
    #read additional settings for island info - adds:
    settings = SDS_island_tools.read_island_info(island_file,settings)
    
    # [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
    SDS_preprocess.save_jpg(metadata, settings)
    
    ## [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections); required if using sand_polygon
    settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
    ### set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
    settings['max_dist_ref'] = 100        
    ##
    #%% extract shorelines from all images (also saves output.pkl and shorelines.kml)
    output = SDS_island_shorelines.extract_shorelines(metadata, settings)
    
    #plot time series of beach area
    fig = plt.figure()
    plt.plot(output['dates'],output['sand_area'],'b-x')
    plt.grid('on')
    plt.xlabel('Date')
    plt.ylabel('Sub-aerial sand area (m^2)')
    plt.show()
    fig.set_size_inches([8,  4])        
    
    #%% 4. Shoreline analysis
    
    # if you have already mapped the shorelines, load the output.pkl file
    #filepath = os.path.join(inputs['filepath'], sitename)
    #with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    #    output = pickle.load(f) 
    
    # now we have to define cross-shore transects over which to quantify the shoreline changes
    # each transect is defined by two points, its origin and a second point that defines its orientation
    # the parameter transect length determines how far from the origin the transect will span
    settings['transect_length'] = 400 
    
    # there are 3 options to create the transects:
    # - option 1: draw the shore-normal transects along the beach
    # - option 2: load the transect coordinates from a .kml file
    # - option 3: create the transects manually by providing the coordinates
    # - option 4: load transects from pre-made pickle file
    # - option 5: calculate transects emanating from single origin point (e.g. for islands)
    
    # option 1: draw origin of transect first and then a second point to define the orientation - example from main CoastSat toolbox
    #transects = SDS_island_transects.draw_transects(output, settings)
        
    # option 2: load the transects from a KML file - example from main CoastSat toolbox
    #kml_file = 'NARRA_transects.kml'
    #transects = SDS_transects.load_transects_from_kml(kml_file)
    
    # option 3: create the transects by manually providing the coordinates of two points - example from main CoastSat toolbox
    #transects = dict([])
    #transects['Transect 1'] = np.array([[342836, 6269215], [343315, 6269071]])
    #transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
    #transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
    
    # option 4: load transects from pre-made pickle file -example from main CoastSat toolbox
    #filepath = os.path.join(inputs['filepath'], sitename)
    #with open(os.path.join(filepath, sitename + '_transects' + '.pkl'), 'rb') as f:
    #    transects = pickle.load(f)
    
    #option 5: transects from single origin point
    ang_start = 0 
    ang_end = 360
    ang_step = 10 #degree step for calculating transects 
    settings['heading'] = np.array(list(range(ang_start,ang_end,ang_step)))
           
    transects = SDS_island_transects.calc_island_transects(settings)
        
    # intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
    settings['along_dist'] = 10
    
    #add some print out to show percentage of shorelines processed 
    cross_distance = SDS_island_transects.compute_intersection(output, transects, settings)            
    
    
    #%% 5. tide correction for transects and sand polygon
        
    #process tide data
    #input tide data is in local time (Australian West Coast, UTC +8 hrs), but code below converts to UTC
    tide_file = os.path.join(os.getcwd(),'example','EvaTide_2019.txt')
    tide, output_corrected = SDS_island_tools.process_tide_data(tide_file, output)    

    cross_distance_corrected = SDS_island_tools.tide_correct(cross_distance,tide,settings['zref'],settings['beach_slope'])
       
    #Calculate tidally corrected sand_polygon     
    output_corrected = SDS_island_tools.tide_correct_sand_polygon(cross_distance_corrected, output_corrected, settings)
    
    
