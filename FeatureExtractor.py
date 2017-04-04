from skimage.feature import hog
import Standardizer
import numpy as np
import cv2

class FeatureExtractor():
	def __init__(self):
		self.hog_orient = 9
		self.hog_pix_per_cell = 4
		self.hog_cell_per_block = 2

	def extract_features(self, image):
		image = Standardizer.convert_color(image, cv2.COLOR_RGB2YCrCb)

		ch1_hog, ch2_hog, ch3_hog = self.extract_hog_features(image)
		hog_features = np.concatenate((ch1_hog.ravel(), ch2_hog.ravel(), ch3_hog.ravel()))

		spatial_features = self.bin_spatial(image)
		
		color_hist_features = self.color_histogram(image)

		canny_features = self.canny(image)

		return np.concatenate((hog_features, spatial_features, color_hist_features, canny_features))

	def canny(self, image, threshold=(80, 100)):
		gray_image = image[:, :, 0]
		canny = cv2.Canny(gray_image, threshold[0], threshold[1])
		canny_features = canny.ravel()
		return canny_features

	def bin_spatial(self, image, size=(24, 24)):
		return cv2.resize(image, size).ravel()

	def color_histogram(self, image, nbins=32, bins_range=(0, 256)):
		ch1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
		ch2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
		ch3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

		hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
		return hist_features

	def extract_hog_features(self, image):		
		channel_1 = image[:, :, 0]
		channel_2 = image[:, :, 1]
		channel_3 = image[:, :, 2]

		ch1_hog = hog(channel_1, orientations=self.hog_orient, pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell), 
					cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block), visualise=False, feature_vector=False)
		ch2_hog = hog(channel_2, orientations=self.hog_orient, pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell), 
					cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block), visualise=False, feature_vector=False)
		ch3_hog = hog(channel_3, orientations=self.hog_orient, pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell), 
					cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block), visualise=False, feature_vector=False)

		return ch1_hog, ch2_hog, ch3_hog

################
# Unit Testing #
################
# Test by running 'python FeatureExtractor.py'
if __name__ == '__main__':
	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt

	# Load an image
	image = Standardizer.read_image('data/vehicles/GTI_Far/image0000.png')

	cimage = Standardizer.convert_color(image, cv2.COLOR_RGB2YCrCb)

	featureExtractor = FeatureExtractor()
	# testimg = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
	image = cimage[:, :, 0]
	testimg1 = cv2.Canny(image, 80, 100)

	plt.figure()
	plt.imshow(image)
	plt.figure('Canny')
	plt.imshow(testimg1)
	plt.show()
	# # Run extract_features()
	# features = featureExtractor.extract_features(image)

	# render original image

	# render features image