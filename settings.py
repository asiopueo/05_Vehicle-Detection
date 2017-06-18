


class Settings:
	def __init__(self):
		self.color_space = 'RGB' 		# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		self.spatial_size = (32, 32) 	# Spatial binning dimensions
		self.hist_bins = 32    			# Number of histogram bins
		self.orient = 9  				# HOG orientations
		self.pix_per_cell = 8 			# HOG pixels per cell
		self.cell_per_block = 2 		# HOG cells per block
		self.hog_channel = 'ALL'		# Can be 0, 1, 2, or "ALL"
		self.spatial_feat = True 		# Spatial features on or off
		self.hist_feat = True 			# Histogram features on or off
		self.hog_feat = True 			# HOG features on or off




settings = Settings()







