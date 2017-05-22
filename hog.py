
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.feature import hog


pix_per_cell = 8
cell_per_block = 2
orient = 9




def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixel_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block,cell_per_block), visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=False, visualise=False, feature_vector=feature_vec) 
		return features







images = glob.glob('*.jpeg')
cars = []
notcars = []


for image in images:
	if 'image' in image or 'extra' in image:
		notcars.append(image)
	else:
		cars.append(image)



features, hog_image = hog(image, orientations=orient, pixel_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block,cell_per_block), visualise=True, feature_vector=False)

plt.subplot(2,1,1)
plt.imshow(image)

plt.subplot(2,1,2)
plt.imshow(hog_image, cmap='gray')

plt.show()

