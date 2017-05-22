import numpy as np

import cv2 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt





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

	#features = img.ravel()
	return features



image = mpimg.imread('cutout1.jpg')

feature_vec = bin_spatial(image, color_space='RGB', size=(32,32))

plt.plot(feature_vec)
plt.show()






