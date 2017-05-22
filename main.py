import sys
import getopt
import glob
import time

import numpy as np
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split

from sliding_window import *
from search_and_classify import search_windows, extract_features

#from moviepy.editor import VideoFileClip




def add_heat(heat, bbox_list):
	for box in bbox_list:
		heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	return heat


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap


def draw_labeled_boxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array
		nonzerox = np.array

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	return img


def pipeline(img, clf, scaler):
	# We will use three different sizes of sliding windows:
	# Their start and stop poitions will be determined at the later stage of pipeline development.

	pipeline.windows1 = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(64,64), xy_overlap=(0.75,0.75))
	pipeline.windows2 = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(128,128), xy_overlap=(0.75,0.75))
	pipeline.windows3 = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(256,256), xy_overlap=(0.75,0.75))

	pipeline.window1_img = draw_boxes(img, pipeline.windows1, color=(255,0,0), thick=3)
	pipeline.window2_img = draw_boxes(img, pipeline.windows2, color=(0,255,0), thick=3)
	pipeline.window3_img = draw_boxes(img, pipeline.windows3, color=(0,0,255), thick=3)

	# We need to apply matching algorithm here:
	hot_boxes = []
	hot_boxes = search_windows(img, pipeline.windows2, clf=clf, scaler=scaler)

	pipeline.heatmap = np.zeros_like(img)
	pipeline.heatmap = add_heat(pipeline.heatmap, hot_boxes)
	pipeline.heatmap = apply_threshold(pipeline.heatmap, 0)

	# Draw and count labeled boxes here:
	pipeline.labels = label(pipeline.heatmap)
	#draw_labeled_boxes()

	return pipeline.window2_img



# Use the pipeline to process only a single image.  Useful for tweaking parameters.
def imageProcessing():
	height = 2
	width = 3

	image = mpimg.imread('test_images/test1.jpg')
	
	result = pipeline(image)
	plt.subplot(height,width,1)
	plt.imshow(image)
	plt.title('Original Input Image')
	plt.subplot(height,width,height*width)
	plt.imshow(result)
	plt.title('Output Image')
	plt.tight_layout()
	plt.show()


# Use the pipeline to compile a video.
def videoProcessing():
	clip = VideoFileClip('./test_video.mp4')
	output_handel = './output.mp4'
	output_stream = clip.fl_image(pipeline)
	output_stream.write_videofile(output_handel, audio=False)


def usage():
	pass


def main(argv):
	pass




if __name__=='__main__':

	# Only using three small set. 
	# Could be an idea to implement the option to use the larger set via a command line option.
	images = glob.glob('./vehicles_smallset/cars[1-3]/*.jpeg') + glob.glob('./non-vehicles_smallset/notcars[1-3]/*.jpeg')

	cars = []
	notcars = []

	for image in images:
		if 'image' in image or 'extra' in image:
			notcars.append(image)
		else:
			cars.append(image)

	# All the parameters have to be harmonized!!!
	
	color_space = 'RGB' 		# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9  				# HOG orientations
	pix_per_cell = 8 			# HOG pixels per cell
	cell_per_block = 2 			# HOG cells per block
	hog_channel = 0 			# Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32) 	# Spatial binning dimensions
	hist_bins = 32    			# Number of histogram bins
	spatial_feat = True 		# Spatial features on or off
	hist_feat = True 			# Histogram features on or off
	hog_feat = True 			# HOG features on or off
	y_start_stop = [None, None]	# Min and max in y to search in slide_window()


	car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
	notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

	# Preprocessing
	X = np.vstack((car_features, notcar_features)).astype(np.float64)

	print('TEST:', X.shape)

	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
 
	svc = LinearSVC()
	t_start = time.time()
	svc.fit(X_train, y_train)
	t_end = time.time()
	print(round(t_start-t_end, 2), 'Seconds to train SVC...')
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


	image = mpimg.imread('test_images/test1.jpg')
	# We first train a HOG classifier.  We are doing this on-the-fly, without saving the results.
	# After we have trained it, we can use the classifier by applying the sliding window search.
	# This is done in the pipeline.


	result = pipeline(image, svc, X_scaler)
	print(pipeline.labels[1], ' cars found.')

	plt.imshow(result)
	plt.title('Output of Pipeline')
	plt.show()

	sys.exit()










