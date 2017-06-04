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



def derive_hog_features(img, settings):
	if settings.hog_channel == 'ALL':
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.extend( hog(img[:, :, channel], orientations=settings.orient, pixels_per_cell=(settings.pix_per_cell, settings.pix_per_cell), cells_per_block=(settings.cell_per_block, settings.cell_per_block), transform_sqrt=True, visualise=False, feature_vector=False) )
			
		hog_features = np.ravel(hog_features)

	else:
		hog_features = hog(img[:,:,settings.hog_channel], orientations=settings.orient, pixels_per_cell=(settings.pix_per_cell, settings.pix_per_cell), cells_per_block=(settings.cell_per_block, settings.cell_per_block), transform_sqrt=True, visualise=False, feature_vector=False)

	return hog_features



# Probleme mit x<->y beachten!
def subsample_hog_features(image_hog_feats, window, settings):
	((sx, sy), (ex, ey)) = window
	cell_start_x = sx//settings.pix_per_cell
	cell_end_x = ex//settings.pix_per_cell-1
	cell_start_y = sy//settings.pix_per_cell 
	cell_end_y = ey//settings.pix_per_cell-1

	#print (image_hog_feats.shape)
	if settings.hog_channel == 'ALL':
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.extend( image_hog_feats[channel, cell_start_x:cell_end_x , cell_start_y:cell_end_y , :, :, :] )
			
		hog_features = np.ravel(hog_features)

	else:
		hog_features = image_hog_feats[cell_start_y:cell_end_y , cell_start_x:cell_end_x , :, :, :]
		#print(hog_features.shape)


	return hog_features




def get_all_features(img, settings, image_hog_feats=None, window=None):

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
		spatial_features = cv2.resize(feature_image, settings.spatial_size).ravel()
		img_features.append(spatial_features)

	if settings.hist_feat == True:
		hist_features = color_hist(feature_image, nbins=settings.hist_bins)
		img_features.append(hist_features)

	if settings.hog_feat == True:
		if (image_hog_feats!=None) and (window!=None):
			hog_features = subsample_hog_features(image_hog_feats, window, settings)
			hog_features = hog_features.reshape(-1)
		else:
			hog_features = derive_hog_features(img, settings)
			hog_features = hog_features.reshape(-1)
			
		img_features.append(hog_features)

	return np.concatenate(img_features)




def search_windows(img, windows, clf, scaler, settings):

	image_hog_feats = hog(img[:,:,2], orientations=settings.orient, pixels_per_cell=(settings.pix_per_cell, settings.pix_per_cell), cells_per_block=(settings.cell_per_block, settings.cell_per_block), transform_sqrt=True, visualise=False, feature_vector=False)
	
	hot_windows = []

	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
		
		window_features = get_all_features(test_img, settings, image_hog_feats, window)
		window_features = scaler.transform(window_features.reshape(1, -1))

		prediction = clf.predict(window_features)
		
		if prediction == 1:
			hot_windows.append(window)

	return hot_windows




def extract_features(imgs, settings):
	features = []

	for file in imgs:
		img = mpimg.imread(file)
		single_features = get_all_features(img, settings)
		features.append(single_features)

	return features










