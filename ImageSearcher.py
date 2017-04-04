from scipy.ndimage.measurements import label
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import Standardizer
import numpy as np
import cv2

class ImageSearcher():
	# Draw the bounding boxes unto the given image
	# Bounding boxes are in the format [((x1, y1), (x2, y2))]
	def draw_bounding_boxes(self, image, boxes, color=(1, 0, 0), thickness=6):
		imcopy = np.copy(image)
		for box in boxes:
			cv2.rectangle(imcopy, box[0], box[1], color, thickness)
		return imcopy

	# Produces a heat map for an image, given a list of bounding boxes
	# Bounding boxes are in the format [((x1, y1), (x2, y2))]
	def build_heat_map(self, image, boxes):
		heat_map = np.zeros_like(image, dtype='float64')
		for box in boxes:
			x1, y1 = box[0]
			x2, y2 = box[1]
			heat_map[y1:y2, x1:x2] += 1
		return heat_map

	# Given a heat map, returns a list of detected bounding boxes
	# Bounding boxes are in the format [((x1, y1), (x2, y2))]
	def find_boxes_from_heatmap(self, heat_map, threshold=0.5):
		boxes = []
		labels = []

		# Apply thresholding to heatmap
		heat_map = np.copy(heat_map)
		heat_map[heat_map <= threshold] = 0

		labels = label(heat_map)

		labeled_image = labels[0]
		num_labels_found = labels[1]

		# Cycle through all the labels found
		# In the image, labels start from 1, not 0
		# so our range starts from 1 as well
		for label_num in range(1, num_labels_found + 1):
			nonzero_indices = (labeled_image == label_num).nonzero()

			nonzero_x = np.array(nonzero_indices[1])
			nonzero_y = np.array(nonzero_indices[0])

			x1 = np.min(nonzero_x)
			y1 = np.min(nonzero_y)
			x2 = np.max(nonzero_x)
			y2 = np.max(nonzero_y)

			boxes.append(((x1, y1), (x2, y2)))

		return boxes

	# Runs a sliding window across a region of the image
	# at different scales, and feeds the image slices to the classifier
	def find_cars(self, image, scaler, classifier, feature_extractor, confidence_threshold):
		# Define our search region (for all windows regardless of scale)
		region_y_start = 350
		region_y_stop = 690

		# Slice out only the region we want to search
		region_image = image[region_y_start:region_y_stop, :, :]
		region_image = Standardizer.convert_color(region_image, cv2.COLOR_RGB2YCrCb)

		# Find get the bounding boxes for possible matches
		# Here we search twice at different scales
		possible_matches = self.__find_cars_in_region(region_image, region_y_start=region_y_start, scale_factor=1, scaler=scaler, classifier=classifier, feature_extractor=feature_extractor, confidence_threshold=confidence_threshold)
		possible_matches2 = self.__find_cars_in_region(region_image, region_y_start=region_y_start, scale_factor=2, scaler=scaler, classifier=classifier, feature_extractor=feature_extractor, confidence_threshold=confidence_threshold)
		possible_matches.extend(possible_matches2)

		return possible_matches

	def __find_cars_in_region(self, region_image, region_y_start, scale_factor, scaler, classifier, feature_extractor, confidence_threshold):

		# Get image width and height
		region_image_height, region_image_width = region_image.shape[:2]

		# Resize according to scale factor
		image = cv2.resize(region_image, (np.int(region_image_width / scale_factor), np.int(region_image_height / scale_factor)))
		image_height, image_width = image.shape[:2]

		# Extract the image channels
		ch1_hog, ch2_hog, ch3_hog = feature_extractor.extract_hog_features(image)

		nxblocks = (image_width // feature_extractor.hog_pix_per_cell) - 1
		nyblocks = (image_height // feature_extractor.hog_pix_per_cell) - 1 
		nfeat_per_block = feature_extractor.hog_orient * feature_extractor.hog_cell_per_block ** 2

		window_size = 64
		cells_per_step = 2

		nblocks_per_window = (window_size // feature_extractor.hog_pix_per_cell) - 1
		nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
		nysteps = (nyblocks - nblocks_per_window) // cells_per_step

		possible_matches = []		
		for x in range(nxsteps):
			for y in range(nysteps):

				# Working at the cell level
				ypos = y * cells_per_step
				xpos = x * cells_per_step

				# Working at the pixel level
				xleft = xpos * feature_extractor.hog_pix_per_cell
				ytop = ypos * feature_extractor.hog_pix_per_cell

				# Get hog features for 3 channels of a given patch
				ch1_hog_block = ch1_hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				ch2_hog_block = ch2_hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				ch3_hog_block = ch3_hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				hog_features = np.concatenate((ch1_hog_block, ch2_hog_block, ch3_hog_block))


				# Extract the image patch
				image_patch = cv2.resize(image[ytop:ytop + window_size, xleft:xleft + window_size], (64,64))


				# Get other features
				spatial_features = feature_extractor.bin_spatial(image_patch)
				color_hist_features = feature_extractor.color_histogram(image_patch)
				canny_features = feature_extractor.canny(image_patch)

				# Scale features and make a prediction
				test_features = np.concatenate((hog_features, spatial_features, color_hist_features, canny_features)).reshape(1, -1)
				test_features = scaler.transform(test_features)
				
				test_prediction_confidence = classifier.predict_proba(test_features)[:, 1]

				if test_prediction_confidence >= confidence_threshold:
					xbox_left = np.int(xleft * scale_factor)
					ytop_draw = np.int(ytop * scale_factor)
					win_draw = np.int(window_size * scale_factor)

					x1 = xbox_left
					y1 = ytop_draw+region_y_start
					x2 = xbox_left+win_draw
					y2 = ytop_draw+win_draw+region_y_start

					possible_matches.append(((x1, y1), (x2, y2)))					

		return possible_matches

################
# Unit Testing #
################
# Run 'python ImageSearch.py' to test
if __name__ == '__main__':
	import matplotlib.image as mpimage
	import matplotlib.pyplot as plt

	testimage = Standardizer.read_image('test_images/test1.png')

	image_searcher = ImageSearcher()
	image_searcher.search_image(testimage, None)
