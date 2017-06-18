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

import pickle

from classify import extract_features







# Training classifier and scaler with the large set. 
# Could be an idea to implement the option to use the larger set via a command line option.

car_images = glob.glob('./vehicles/*/*.png')
notcar_images = glob.glob('./non-vehicles/*/*.png')


print('Extracting car features...')
car_features = extract_features(car_images)
print('Extracting non-car features...')
notcar_features = extract_features(notcar_images)

#
#	Feature missing: rescaling of PNGs
#

# Preprocessing

X = np.vstack((car_features, notcar_features)).astype(np.float64)

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))

# We first train a HOG classifier.  We are doing this on-the-fly, without saving the results.
# After we have trained it, we can use the classifier by applying the sliding window search.
# This is done in the pipeline.

svc = LinearSVC()

print('Start training the classifier...')
t_start = time.time()
svc.fit(X_train, y_train)
t_end = time.time()
print('It took', round(t_end-t_start, 2), 'seconds to train the SVC...')

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


#
#	Savong the classifier and scaler:
#

with open('classifier.pkl', 'wb') as file_handle:
	pickle.dump(svc, file_handle)
	
with open('scaler.pkl', 'wb') as file_handle:
	pickle.dump(X_scaler, file_handle)
	







