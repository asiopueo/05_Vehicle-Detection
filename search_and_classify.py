import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split








def bin_spatial(img, color_space='RGB', size=(32,32)):
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		feature_image = np.copy(img)

	features = cv2.resize(feature_image, size).ravel()

	return features


# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features



def img_features(img, settings):
	img_features=[]

	if settings.color_space != 'RGB':
		if settings.color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img) 

	if settings.spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=settings.spatial_size)
		img_features.append(spatial_features)

	if settings.hist_feat == True:
		hist_features = color_hist(feature_image, nbins=settings.hist_bins)
		img_features.append(hist_features)

	if settings.hog_feat == True:
		if settings.hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend( get_hog_features(feature_image[:,:,channel], settings.orient, settings.pix_per_cell, settings.cell_per_block, vis=False, feature_vec=True) )
			hog_features = np.ravel(hog_features) 
		else:
			hog_features = get_hog_features(feature_image[:,:,settings.hog_channel], settings.orient, settings.pix_per_cell, settings.cell_per_block, vis=False, feature_vec=True)

		img_features.append(hog_features)

	return np.concatenate(img_features)


def subsample(img_feats, window, settings):
	
	window_feat = img_feats[:,:,:,:,:]

	return




def search_windows(img, windows, clf, scaler, settings):

	img_feats = img_features(img, settings)
	img_feats = scaler.transform(img_feats.reshape(1, -1))

	hot_windows = []
	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
		
		window_features = subsample(img_feats, window)
		window_features = scaler.transform(window_features.reshape(1, -1))

		prediction = clf.predict(window_features)

		if prediction == 1:
			hot_windows.append(window)

	return hot_windows





def extract_features(imgs, settings):

	features = []

	for file in imgs:
		image = mpimg.imread(file)
		single_features = img_features(image, settings)
		features.append(single_features)

	return features










