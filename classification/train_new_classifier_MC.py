# -*- coding: utf-8 -*-
"""
Copy of 'Train a new classifier for CoastSat'

Converted from Jupyter notebook to .py script
M Cuttler - 20 April 2020
"""
#%%
# load modules
# load_ext autoreload
# autoreload 2

import os, sys
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

os.chdir(r"D:\CUTTLER_GitHub\CoastSat")
# coastsat modules
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_classify

# plotting params
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12

# filepaths 
filepath_images = os.path.join(os.getcwd(), 'classification','data')
filepath_train = os.path.join(os.getcwd(), 'classification','training_data')
filepath_models = os.path.join(os.getcwd(), 'classification','models')

# settings
settings ={'filepath_train':filepath_train, # folder where the labelled images will be stored
           'cloud_thresh':0.9, # percentage of cloudy pixels accepted on the image
           'cloud_mask_issue':True, # set to True if problems with the default cloud mask 
           'inputs':{'filepath':filepath_images}, # folder where the images are stored
           'labels':{'sand':1,'white-water':2,'water':3,'other land features':4}, # labels for the classifier
           'colors':{'sand':[1, 0.65, 0],'white-water':[1,0,1],'water':[0.1,0.1,0.7],'other land features':[0.8,0.8,0.1]},
           'tolerance':0.01, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
            }
        
# read kml files for the training sites
# region = 'PerthMetro'
filepath_sites = os.path.join(os.getcwd(), 'classification','training_sites')
train_sites = os.listdir(filepath_sites)
print('Sites for training:\n%s\n'%train_sites)

#%% 1. Download images
# dowload images at the sites
dates = ['2019-01-01', '2019-07-01']
sat_list = ['L8']

#loop through multiple sites
# for site in train_sites:
#     polygon = SDS_tools.polygon_from_kml(os.path.join(filepath_sites,site))
#     sitename = site[:site.find('.')]  
#     inputs = {'polygon':polygon, 'dates':dates, 'sat_list':sat_list,
#              'sitename':sitename, 'filepath':filepath_images}
#     print(sitename)
#     metadata = SDS_download.retrieve_images(inputs)

# site = 'BUSSELTON_JETTY.kml'
# polygon = SDS_tools.polygon_from_kml(os.path.join(filepath_sites,site))
# sitename = site[:site.find('.')]  
# inputs = {'polygon':polygon, 'dates':dates, 'sat_list':sat_list,
#             'sitename':sitename, 'filepath':filepath_images}
# print(sitename)
# metadata = SDS_download.retrieve_images(inputs)

#%% 2. Label images
# label the images with an interactive annotator
# matplotlib qt
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    settings['inputs']['sat_list']=sat_list
    metadata = SDS_download.get_metadata(settings['inputs'])
    # label images
    SDS_classify.label_images(metadata,settings)

# site = 'BUSSELTON_JETTY.kml'
# settings['inputs']['sitename'] = site[:site.find('.')] 
# # load metadata
# settings['inputs']['sat_list']=sat_list
# metadata = SDS_download.get_metadata(settings['inputs'])
# # label images
# SDS_classify.label_images(metadata,settings)
    
#%% 3. Train classifier
# load labelled images
features = SDS_classify.load_labels(train_sites, settings)

# you can also load the original CoastSat training data (and optionally merge it with your labelled data)
with open(os.path.join(settings['filepath_train'], 'CoastSat_training_set_L8.pkl'), 'rb') as f:
    features_original = pickle.load(f)
for key in features_original.keys():
    print('%s : %d pixels'%(key,len(features_original[key])))

# # add the white-water data from the original training data
features['white-water'] = np.append(features['white-water'], features_original['white-water'], axis=0)
features['other land features'] = np.append(features['other land features'], features_original['other land features'], axis=0)
# # or merge all the classes
# for key in features.keys():
#     features[key] = np.append(features[key], features_original[key], axis=0)
# features = features_original 
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))

#[OPTIONAL] As the classes do not have the same number of pixels, it is good practice to subsample
#the very large classes (in this case 'water' and 'other land features')
# subsample randomly the land and water classes
# as the most important class is 'sand', the number of samples should be close to the number of sand pixels
n_samples = 10000
for key in ['water', 'other land features']:
    features[key] =  features[key][np.random.choice(features[key].shape[0], n_samples, replace=False),:]
# print classes again
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))


# format into X (features) and y (labels) 
classes = ['sand','white-water','water','other land features']
labels = [1,2,3,0]
X,y = SDS_classify.format_training_data(features, classes, labels)


#check for Nan, infinite, etc
if ~np.all(np.isfinite(X)):
    checkX = np.where(~np.isfinite(X))
    row_remove = np.unique(checkX[0])
    Xclean = np.empty((0,X.shape[1]),'float64')
    yclean = np.array([],dtype='float64')
    for i,row in enumerate(X):
        if ~np.isin(i,row_remove):
            Xclean=np.append(Xclean, np.reshape(row,(1,20)),axis=0)
            yclean=np.append(yclean, y[i])
    X = Xclean
    y = yclean        

         

# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X_train,y_train)             
print('Accuracy: %0.4f' % classifier.score(X_test,y_test))

#[OPTIONAL] A more robust evaluation is 10-fold cross-validation (may take a few minutes to run)
# cross-validation
scores = cross_val_score(classifier, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))

# plot confusion matrix
# %matplotlib inline
y_pred = classifier.predict(X_test)
SDS_classify.plot_confusion_matrix(y_test, y_pred,
                                   classes=['other land features','sand','white-water','water'],
                                   normalize=False);

# train with all the data and save the final classifier
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X,y)
joblib.dump(classifier, os.path.join(filepath_models, 'NN_4classes_Landsat_WA_metro.pkl'))

#%% 4. Evaluate the classifier 

# load and evaluate a classifier
os.chdir(r"D:\CUTTLER_GitHub\CoastSat\classification")
classifier = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_WA_metro.pkl'))
settings['output_epsg'] = 3857
settings['min_beach_area'] = 4500
settings['buffer_size'] = 200
settings['min_length_sl'] = 200
settings['cloud_thresh'] = 0.5
# visualise the classified images
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    metadata = SDS_download.get_metadata(settings['inputs'])
    # plot the classified images
    SDS_classify.evaluate_classifier(classifier,metadata,settings)