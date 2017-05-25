import sys

import numpy as np
import cv2
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt




def sliding_window(img, x_start_stop=[None,None], y_start_stop=[None,None], xy_window=(64,64), xy_overlap=(0.5,0.5)):
	 
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]

	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

	window_list=[]

	for ys in range(ny_windows):
		for xs in range(nx_windows):
  
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			window_list.append(((startx, starty), (endx, endy)))

	return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	imcopy = np.copy(img)

	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

	return imcopy



if __name__=='__main__':

	image = mpimg.imread('test_images/test1.jpg')
	windows = sliding_window(image, x_start_stop=[None,None], y_start_stop=[None,None], xy_window=(128,128), xy_overlap=(0.25,0.25))
	window_img = draw_boxes(image, windows, color=(0,0,255), thick=3)


	plt.title('test1.jpg')
	plt.imshow(window_img)
	plt.show()

	sys.exit()





