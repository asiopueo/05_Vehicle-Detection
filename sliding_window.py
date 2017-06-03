import sys

import numpy as np
import cv2
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



# Not quite finished!
def sliding_window(img, x_start_stop=[None,None], y_start_stop=[None,None], xy_window=(64,64), xy_overlap=(0.5,0.5), pix_per_cell=8):
	 
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
 


	x_start_stop[0] = x_start_stop[0] // pix_per_cell
	x_start_stop[1] = x_start_stop[1] // pix_per_cell
	y_start_stop[0] = y_start_stop[0] // pix_per_cell
	y_start_stop[1] = y_start_stop[1] // pix_per_cell

	nx_cells_span = x_start_stop[1] - x_start_stop[0]
	ny_cells_span = y_start_stop[1] - y_start_stop[0]

	nx_cells_per_step = np.int( (xy_window[0]*(1 - xy_overlap[0]))//pix_per_cell )
	ny_cells_per_step = np.int( (xy_window[1]*(1 - xy_overlap[1]))//pix_per_cell )

	# nx_buffer not fully defined yet  
	nx_buffer = np.int( (xy_window[0]*(xy_overlap[0]))//pix_per_cell )
	ny_buffer = np.int( (xy_window[1]*(xy_overlap[1]))//pix_per_cell )

	nx_windows = np.int( (nx_cells_span-nx_buffer)/nx_cells_per_step ) 
	ny_windows = np.int( (ny_cells_span-ny_buffer)/ny_cells_per_step ) 
	
	window_list=[]

	for ys in range(ny_windows):
		for xs in range(nx_windows):
  
			startx = (xs*nx_cells_per_step + x_start_stop[0]) * pix_per_cell
			endx = startx + xy_window[0]
			starty = (ys*ny_cells_per_step + y_start_stop[0]) * pix_per_cell
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





