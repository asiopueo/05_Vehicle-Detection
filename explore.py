import numpy as np
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob



def data_look(car_list, notcar_list):
	data_dict = {}
	data_dict['n_cars'] = 0
	data_dict['n_notcars'] = 0
	data_dict['image_shape'] = (0,0,0)
	data_dict['data_type'] = None
	return data_dict





images = glob.glob('./vehicles_smallset/cars[1-3]/*.jpeg') + glob.glob('./non-vehicles_smallset/notcars[1-3]/*.jpeg')


cars = []
notcars = []


for image in images:
	if 'image' in image or 'extra' in image:
		notcars.append(image)
	else:
		cars.append(image)





data_info = data_look(cars, notcars)




print('Your function returned a count of', data_info['n_cars'], ' cars and', data_info['n_notcars'], 'non_cars')
print('of size: ', data_info['image_shape'], 'and data type: ', data_info['data_type'])

car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


plt.subplot(211)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(212)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.tight_layout()
plt.show()






