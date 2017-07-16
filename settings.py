


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

	def save(self, file_name):
		with open(file_name, 'w+') as f:
			f.write('color_space = ' 	+ str(self.color_space) + '\n')
			f.write('spatial_size = ' 	+ str(self.spatial_size) + '\n')
			f.write('hist_bins = ' 		+ str(self.hist_bins) + '\n')
			f.write('orient = ' 		+ str(self.orient) + '\n')
			f.write('pix_per_cell = ' 	+ str(self.pix_per_cell) + '\n')
			f.write('cell_per_block = ' + str(self.cell_per_block) + '\n')
			f.write('hog_channel = ' 	+ self.hog_channel + '\n')
			f.write('spatial_feat = ' 	+ str(self.spatial_feat) + '\n')
			f.write('hist_feat = ' 		+ str(self.hist_feat) + '\n')
			f.write('hog_feat = ' 		+ str(self.hog_feat) + '\n')

	def load(self, file_name):
		with open(file_name, 'r') as f:
			self.color_space = f.readline().split()[-1]				# str
			
			tuple_as_str = f.readline().split(' = ')[-1]
			self.spatial_size = tuple(map(int, tuple_as_str[1:-2].split(',')))	# (int, int)
			
			self.hist_bins = int(f.readline().split()[-1])			# int
			self.orient = int(f.readline().split()[-1])				# int
			self.pix_per_cell = int(f.readline().split()[-1])		# int
			self.cell_per_block = int(f.readline().split()[-1])		# int
			self.hog_channel = f.readline().split()[-1]				# str
			self.spatial_feat = bool(f.readline().split()[-1])		# bool
			self.hist_feat = bool(f.readline().split()[-1])			# bool
			self.hog_feat = bool(f.readline().split()[-1])			# bool

	def print(self):
		print('color_space = {}'.format(self.color_space) )
		print('spatial_size = {}'.format(self.spatial_size) )
		print('hist_bins = {}'.format(self.hist_bins) )
		print('orient = {}'.format(self.orient) )
		print('pix_per_cell = {}'.format(self.pix_per_cell) )
		print('cell_per_block = {}'.format(self.cell_per_block) )
		print('hog_channel = {}'.format(self.hog_channel) )
		print('spatial_feat = {}'.format(self.spatial_feat) )
		print('hist_feat = {}'.format(self.hist_feat) )
		print('hog_feat = {}'.format(self.hog_feat) )



settings = Settings()







