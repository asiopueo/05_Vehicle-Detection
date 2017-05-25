import sys

import numpy as np
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2





files = ['./vehicles_smallset/cars1/243.jpeg',
		 './non-vehicles_smallset/notcars2/image0467.jpeg']



car_image = plt.imread(files[0])
notcar_image = plt.imread(files[1])

height = 1
width = 2

fig = plt.figure()

fig.add_subplot(height,width,1)
plt.imshow(car_image)
plt.title('Car')
	
fig.add_subplot(height,width,2)
plt.title('Not a Car')
plt.imshow(notcar_image)

plt.tight_layout()
plt.show()

fig.savefig('./output/car_not_car.png')
		
sys.exit()







