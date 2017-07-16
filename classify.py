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


###############################
# Import configuration module #
###############################
from settings import settings





def color_conversion(img):
	if settings.color_space != 'RGB':
		if settings.color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif settings.color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif settings.color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif settings.color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif settings.color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img) 

	return feature_image



	
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



def extract_hog_features(img):
	if settings.hog_channel == 'ALL':
		hog_features = []
		for channel in range(img.shape[2]):
			# Returns a short list of hog-ndarrays
			hog_features.append( hog(img[:, :, channel], orientations=settings.orient, pixels_per_cell=(settings.pix_per_cell, settings.pix_per_cell), cells_per_block=(settings.cell_per_block, settings.cell_per_block), transform_sqrt=True, visualise=False, feature_vector=False) )		
	else:
		# Returns a hog-ndarray
		hog_features = hog(img[:,:,settings.hog_channel], orientations=settings.orient, pixels_per_cell=(settings.pix_per_cell, settings.pix_per_cell), cells_per_block=(settings.cell_per_block, settings.cell_per_block), transform_sqrt=True, visualise=False, feature_vector=False)

	return hog_features




# Probleme mit x<->y beachten!
def subsample_hog_features(global_hog_feats, window):
	((sx, sy), (ex, ey)) = window
	cell_start_x = sx // settings.pix_per_cell
	cell_end_x = ex // settings.pix_per_cell - 1
	cell_start_y = sy // settings.pix_per_cell 
	cell_end_y = ey // settings.pix_per_cell - 1

	if settings.hog_channel == 'ALL':
		hog_features = []
		for channel in range(3):	# Ausbessern!
			hog_tmp = global_hog_feats[channel]
			hog_features.extend( hog_tmp[cell_start_y:cell_end_y, cell_start_x:cell_end_x, :, :, :] )
			
	else:
		hog_features = global_hog_feats[cell_start_y:cell_end_y, cell_start_x:cell_end_x, :, :, :]

	return hog_features




def extract_search_window_features(img, windows, boxscale, clf, scaler):

	img = color_conversion(img)
	img = cv2.resize(img[:,:,:], (int(img.shape[1]//boxscale),int(img.shape[0]//boxscale)) )
	global_hog_feats = extract_hog_features(img) 
	

	hot_windows = []

	for window in windows:
		window_tmp = ((int(window[0][0]//boxscale), int(window[0][1]//boxscale)), (int(window[1][0]//boxscale), int(window[1][1]//boxscale)) )
		test_img = img[window_tmp[0][1]:window_tmp[1][1], window_tmp[0][0]:window_tmp[1][0]]
		test_img = color_conversion(test_img)

		img_features=[]

		if settings.spatial_feat == True:
			spatial_features = cv2.resize(test_img, settings.spatial_size).ravel()
			img_features.append(spatial_features)

		if settings.hist_feat == True:
			hist_features = color_hist(test_img, nbins=settings.hist_bins)
			img_features.append(hist_features)

		if settings.hog_feat == True:
			hog_features = subsample_hog_features(global_hog_feats, window_tmp)
			hog_features = np.ravel(hog_features)
			img_features.append(hog_features)


		window_features = np.concatenate(img_features)
		window_features = scaler.transform(window_features)

		prediction = clf.predict(window_features)
		
		if prediction == 1:
			hot_windows.append(window)

	return hot_windows




def extract_features(imgs):
	features = []

	for file in imgs:
		img = mpimg.imread(file)

		feature_list = []
		feature_image = color_conversion(img)*255#Only for PNGs


		if settings.spatial_feat == True:
			spatial_features = cv2.resize(feature_image, settings.spatial_size).ravel()
			feature_list.append(spatial_features)

		if settings.hist_feat == True:
			hist_features = color_hist(feature_image, nbins=settings.hist_bins)
			feature_list.append(hist_features)

		if settings.hog_feat == True:
			hog_features = extract_hog_features(img)
			hog_features = np.ravel(hog_features)
			feature_list.append(hog_features)

		feature_array = np.concatenate(feature_list)
		features.append(feature_array)

	return features










