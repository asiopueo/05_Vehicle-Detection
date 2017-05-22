
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler




def extract_features(imgs, cspace='RGB', spatial_size=(32,32), hist_bins=32, hist_range=(0,256)):
	features = []

	for file in imgs:
		img = mpimg.imread(file)

		if cspace != 'RGB':
			if cspace == 'HSV':
				pass
			elif cpace == 'LUV':
				pass
			elif cspace == 'HLS':
				pass
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor()
		else:
			feature_image = np.copy(image)

		spatial_features = bin_spatial(feature_image, size=spatial_size)
		hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

		features.append(np.concatenate((spatial_features, hist_features)))

	return features





if __name__ == '__main__':
	
	feature_list = [feature_vec1, feature_vec2, ...]

	X = np.vstack(feature_list).astype(np.float64)

	X_Scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	sys.exit()







