import sys, os
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

from moviepy.editor import VideoFileClip
from collections import deque

from functools import partial
import multiprocessing

import pickle



##############
# Own module #
##############

from sliding_window import *
from classify import extract_search_window_features, extract_features
from saver import image_saver, video_saver

###############################
# Import configuration module #
###############################
from settings import settings







def add_heat(heat, bbox_list):
	for box in bbox_list:
		heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	return heat


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap


def screenWriter(img, max, position):
	font = cv2.FONT_HERSHEY_SIMPLEX
	string = 'Max: '+ str(max)
	textedImg = cv2.putText(img, text=string, org=position, fontFace=font, fontScale=1., color=(255,255,255), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
	return textedImg


def draw_labeled_boxes(img, labels):
	for car_number in range(1, labels[1]+1):
		A = (labels[0] == car_number)
		B = A*pipeline.heatmap
		
		print('Box number {}:'.format(car_number), B.max())		
		
		nonzero = A.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
		screenWriter(img,B.max(), bbox[0])

	return img


def pipeline(img, clf, scaler):

	if pipeline.flag == 0:
		pipeline.buffer = deque(12*[],12)
		pipeline.flag = 1
	

	# We will use three different sizes of sliding windows:.
	windows_L = sliding_window(img, x_start_stop=[None,None], y_start_stop=[336,592], xy_window=(128,128), xy_overlap=(0.75,0.75))
	windows_M = sliding_window(img, x_start_stop=[None,None], y_start_stop=[336,592], xy_window=(96,96), xy_overlap=(0.75,0.75))
	windows_S = sliding_window(img, x_start_stop=[None,None], y_start_stop=[336,592], xy_window=(64,64), xy_overlap=(0.75,0.75), pix_per_cell=8)

	pipeline.window_L_img = draw_boxes(img, windows_L, color=(0,0,255), thick=3)
	pipeline.window_M_img = draw_boxes(img, windows_M, color=(0,255,0), thick=3)
	pipeline.window_S_img = draw_boxes(img, windows_S, color=(255,0,0), thick=3)

	# We need to apply the matching algorithm here:
	hot_boxes = []
	search = partial(extract_search_window_features, img, clf=clf, scaler=scaler)
	
	t_start = time.time()
	pool = multiprocessing.Pool()
	hot_boxes = pool.starmap(search, [(windows_S, 1), (windows_M, 1.5)])#, (windows_L, 2)] )
	pool.close()
	hot_boxes = [item for sublist in hot_boxes for item in sublist]
	

	"""
	#Function successive calls without multiprocessing
	hot_boxes = search(windows_S, boxscale=1)
	#hot_boxes = search(windows_M, boxscale=1.5)
	#hot_boxes = search(windows_L, boxscale=2)
	"""
	
	t_end = time.time()
	print(round(t_end-t_start, 2), 'Seconds to determine hot boxes...')
	

	heatmap = np.zeros_like(img)
	heatmap = add_heat(heatmap, hot_boxes)
	
	pipeline.buffer.appendleft(heatmap)
	heatmap = sum(pipeline.buffer)
	print('Absolute maximum value of heatmap:', heatmap.max())
	pipeline.heatmap = apply_threshold(heatmap, 50)

	# Draw and count labeled boxes here:
	pipeline.labels = label(pipeline.heatmap)
	return draw_labeled_boxes(pipeline.heatmap, pipeline.labels)

pipeline.flag = 0



# Use the pipeline to process only a single image.  Useful for tweaking parameters.
def imageProcessing(image, svc, X_scaler):
	height = 2
	width = 3
		
	image = mpimg.imread('test_images/test1.jpg')
	# Might need to scale the image because mpimg scales jpegs from 0 to 255 in contrast to pngs (0 to 1)!
	# The latter were used as training base for the SVM.
	
	result = pipeline(image, svc, X_scaler)

	print(pipeline.labels[1], ' cars found.')
	
	plt.subplot(height,width,1)
	plt.imshow(image)
	plt.title('Original Input Image')
	
	plt.subplot(height,width,5)
	plt.title('Heatmap')
	heatmap_scaled = pipeline.buffer[0]*10
	plt.imshow(heatmap_scaled)
	plt.imsave('./images/heatmap1.png', heatmap_scaled)

	plt.subplot(height,width,2)
	plt.title('Large-sized Boxes')
	plt.imshow(pipeline.window_L_img)
	#plt.imsave('./images/large_boxes.png', pipeline.window_L_img)

	plt.subplot(height,width,3)
	plt.title('Medium-sized Boxes')
	plt.imshow(pipeline.window_M_img)
	#plt.imsave('./images/medium_boxes.png', pipeline.window_M_img)

	plt.subplot(height,width,4)
	plt.title('Small-sized Boxes')
	plt.imshow(pipeline.window_S_img)
	#plt.imsave('./images/small_boxes.png', pipeline.window_S_img)

	plt.subplot(height,width,6)
	plt.title('Resulting Image')
	plt.imshow(result)
	plt.imsave('./images/resulting_image1.png', result)
	image_saver(result)

	plt.tight_layout()

	plt.show()
	sys.exit()




# Use the pipeline to compile a video.
def videoProcessing(clip, svc, X_scaler):
	clip = VideoFileClip('./test_videos/project_video.mp4')#.subclip(5,15)
	output_stream = clip.fl_image(lambda frame: pipeline(frame, svc, X_scaler))
	video_saver(output_stream)
	sys.exit()


def usage():
	print('Usage:')
	print('\t python main.py')
	print('\t python main.py -v [<input_video.mp4>]')
	print('\t python main.py -i [<input_image.jpg>]')
	return


def init(argv):

	init.svc = LinearSVC()
	
	with open('classifier.pkl', 'rb') as file_handle:
		init.svc = pickle.load(file_handle)

	init.X_scaler = StandardScaler()

	with open('scaler.pkl', 'rb') as file_handle:
		init.X_scaler = pickle.load(file_handle)

	return







if __name__=='__main__':

	try:
		opts, args = getopt.getopt(sys.argv[1:], 'vih', ['help'])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-i', '--Image'):
			init(sys.argv[1:])
			imageProcessing(arg, init.svc, init.X_scaler)
			
		elif opt in ('-v', '--Video'):
			init(sys.argv[1:])
			videoProcessing(arg, init.svc, init.X_scaler)

		elif opt in ('-h', '--help'):
			usage()
			sys.exit()

		else:
			usage()
			sys.exit()
	
	sys.exit()





