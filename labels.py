import sys

import numpy as np
import cv2
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label



def draw_labeled_boxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array
		nonzerox = np.array

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	return img


if __name__ == '__main__':

	image = mpimg.imread('test_images/test1.jpg')

	#labels = label(heatmap)
	#print(labels[1], 'cars found')

	draw_img = np.copy(image)
	#draw_img = draw_labeled_boxes(image, labels)

	plt.subplot(2,1,1)
	plt.imshow(draw_img)
	plt.title('Example Test Image')
	plt.show()

	sys.exit()


