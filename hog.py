import sys

import numpy as np
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog




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



# Initialize parameters
set = Settings()

image = mpimg.imread('./vehicles_smallset/cars1/243.jpeg')

features, hog_image = hog(image[:,:,0], orientations=set.orient, pixels_per_cell=(set.pix_per_cell, set.pix_per_cell), cells_per_block=(set.cell_per_block, set.cell_per_block), transform_sqrt=True, visualise=True, feature_vector=False)

height = 1
width = 2
			
fig = plt.figure()
fig.add_subplot(height,width,1)
plt.imshow(image)
plt.title('Original Image')
	
fig.add_subplot(height,width,2)
plt.title('HOG Image')
plt.imshow(hog_image)

plt.tight_layout()
plt.show()

fig.savefig('./output/hog_image.png')
		
sys.exit()











