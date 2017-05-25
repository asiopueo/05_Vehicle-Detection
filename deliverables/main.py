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

from moviepy.editor import VideoFileClip


class Settings:
	def __init__(self):
		self.color_space = 'RGB' 		# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		self.spatial_size = (32, 32) 	# Spatial binning dimensions
		self.hist_bins = 32    			# Number of histogram bins
		self.orient = 9  				# HOG orientations
		self.pix_per_cell = 8 			# HOG pixels per cell
		self.cell_per_block = 2 		# HOG cells per block
		self.hog_channel = 0 			# Can be 0, 1, 2, or "ALL"
		self.spatial_feat = True 		# Spatial features on or off
		self.hist_feat = True 			# Histogram features on or off
		self.hog_feat = True 			# HOG features on or off




def add_heat(heat, bbox_list):
	for box in bbox_list:
		heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 50
	return heat


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap


def draw_labeled_boxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	return img


def pipeline(img, clf, scaler, settings):
	# We will use three different sizes of sliding windows:
	# Their start and stop poitions will be determined at the later stage of pipeline development.

	pipeline.windows_S = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(64,64), xy_overlap=(0.5,0.5))
	pipeline.windows_M = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(128,128), xy_overlap=(0.75,0.75))
	pipeline.windows_L = sliding_window(img, x_start_stop=[None,None], y_start_stop=[360,None], xy_window=(256,256), xy_overlap=(0.75,0.75))

	pipeline.window_S_img = draw_boxes(img, pipeline.windows_S, color=(255,0,0), thick=3)
	pipeline.window_M_img = draw_boxes(img, pipeline.windows_M, color=(0,255,0), thick=3)
	pipeline.window_L_img = draw_boxes(img, pipeline.windows_L, color=(0,0,255), thick=3)

	# We need to apply matching algorithm here:
	hot_boxes = []
	pipeline.heatmap = np.zeros_like(img)
	
	hot_boxes = search_windows(img, pipeline.windows_L, clf, scaler, settings)
	pipeline.heatmap = add_heat(pipeline.heatmap, hot_boxes)
	hot_boxes = search_windows(img, pipeline.windows_M, clf, scaler, settings)
	pipeline.heatmap = add_heat(pipeline.heatmap, hot_boxes)
	hot_boxes = search_windows(img, pipeline.windows_S, clf, scaler, settings)
	pipeline.heatmap = add_heat(pipeline.heatmap, hot_boxes)
		
	pipeline.heatmap = apply_threshold(pipeline.heatmap, 75)

	# Draw and count labeled boxes here:
	pipeline.labels = label(pipeline.heatmap)
	return draw_labeled_boxes(img, pipeline.labels)



# Use the pipeline to process only a single image.  Useful for tweaking parameters.
def imageProcessing(image, svc, X_scaler, settings):
	height = 3
	width = 2
		
	image = mpimg.imread('test_images/test6.jpg')

	result = pipeline(image, svc, X_scaler, settings)

	print(pipeline.labels[1], ' cars found.')
	
	plt.subplot(height,width,1)
	plt.imshow(image)
	plt.title('Original Input Image')
	
	plt.subplot(height,width,2)
	plt.title('Heatmap')
	plt.imshow(pipeline.heatmap)
	plt.imsave('./output/heatmap.png', pipeline.heatmap)

	plt.subplot(height,width,3)
	plt.title('Large-sized Boxes')
	plt.imshow(pipeline.window_L_img)
	plt.imsave('./output/large_boxes.png', pipeline.window_L_img)

	plt.subplot(height,width,4)
	plt.title('Medium-sized Boxes')
	plt.imshow(pipeline.window_M_img)
	plt.imsave('./output/medium_boxes.png', pipeline.window_M_img)

	plt.subplot(height,width,5)
	plt.title('Small-sized Boxes')
	plt.imshow(pipeline.window_S_img)
	plt.imsave('./output/small_boxes.png', pipeline.window_S_img)

	plt.subplot(height,width,6)
	plt.title('Resulting Image')
	plt.imshow(result)
	plt.imsave('./output/resulting_image.png', result)


	plt.tight_layout()

	plt.show()
	sys.exit()


# Use the pipeline to compile a video.
def videoProcessing(clip, svc, X_scaler, settings):
	clip = VideoFileClip('./test_videos/project_video.mp4')
	output_handel = './output/heatmap.mp4'
	#output_handel = './output/bboxes.mp4'
	output_stream = clip.fl_image(lambda frame: pipeline(frame, svc, X_scaler, settings))
	output_stream.write_videofile(output_handel, audio=False)
	sys.exit()


def usage():
	pass



def main(argv):
	# Only using the small set. 
	# Could be an idea to implement the option to use the larger set via a command line option.
	images = glob.glob('./vehicles_smallset/cars[1-3]/*.jpeg') + glob.glob('./non-vehicles_smallset/notcars[1-3]/*.jpeg')

	cars = []
	notcars = []

	for image in images:
		if 'image' in image or 'extra' in image:
			notcars.append(image)
		else:
			cars.append(image)

	# Initialize parameters
	settings = Settings()


	car_features = extract_features(cars, settings)
	notcar_features = extract_features(notcars, settings)

	# Preprocessing
	X = np.vstack((car_features, notcar_features)).astype(np.float64)

	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:', settings.orient, 'orientations', settings.pix_per_cell, 'pixels per cell and', settings.cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
 
	# We first train a HOG classifier.  We are doing this on-the-fly, without saving the results.
	# After we have trained it, we can use the classifier by applying the sliding window search.
	# This is done in the pipeline.

	svc = LinearSVC()
	t_start = time.time()
	svc.fit(X_train, y_train)
	t_end = time.time()
	print(round(t_start-t_end, 2), 'Seconds to train SVC...')
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))



	try:
		opts, args = getopt.getopt(argv, 'vih', ['Image=', 'Video=', 'help'])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-i', '--Image'):
			print('Option: ' + '\'' + arg + '\'')
			imageProcessing(arg, svc, X_scaler, settings)
			
		elif opt in ('-v', '--Video'):
			videoProcessing(arg, svc, X_scaler, settings)
			
		elif opt in ('-h', '--help'):
			usage()
			sys.exit()
			
		elif opt == '-c':
			global _tweak
			_tweak = 1

		else:
			usage()
			sys.exit()





if __name__=='__main__':
	main(sys.argv[1:])
	sys.exit()





