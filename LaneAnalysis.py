import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


#################
# Lane Analysis #
#################
class LaneAnalysis():
	def __init__(self):
		self.left_fit = []
		self.right_fit = []

	# Identify the most likely X position for our lane lines from our image
	def __find_base_x_positions(self, binary_image, debug_mode=False):

		# Get a histogram of the bottom half of the image
		midY = np.int(binary_image.shape[0] / 2)

		# Sum from midY onwards
		histogram = np.sum(binary_image[midY:, :], axis=0)
		
		# Split the histogram into left and right halves, and identify the peaks
		midpoint = np.int(histogram.shape[0] / 2)
		left_base_pos_x = np.argmax(histogram[:midpoint])
		right_base_pos_x = np.argmax(histogram[midpoint:]) + midpoint

		if (debug_mode == True):
			plt.figure('Histogram')
			plt.title('Histogram')
			plt.plot(histogram)
			plt.show()

		return left_base_pos_x, right_base_pos_x

	# Perform a full (more expensive) search and calculation to find the left and right polynomial fits
	def __find_lines_full(self, binary_image, left_base_pos_x, right_base_pos_x, num_sliding_windows=10, debug_mode=False):
		image_height, image_width = binary_image.shape[:2]

		# window height should fit all num_sliding_windows
		window_height = np.int(image_height / num_sliding_windows)

		nonzero_pixel_positions = binary_image.nonzero()
		nonzero_pixel_positions_y = nonzero_pixel_positions[0]
		nonzero_pixel_positions_x = nonzero_pixel_positions[1]

		# Set our starting positions to the base left and right positions we identified
		left_x = left_base_pos_x
		right_x = right_base_pos_x

		window_x_margin = 100
		recenter_threshold = 100

		if (debug_mode == True):
			debug_image = np.dstack((binary_image, binary_image, binary_image)) * 255

		final_left_line_indices = []
		final_right_line_indices = []

		for window in range(num_sliding_windows):
			y1 = image_height - ((window + 1) * window_height)
			y2 = image_height - (window * window_height)
			left_x1 = left_x - window_x_margin
			left_x2 = left_x + window_x_margin
			right_x1 = right_x - window_x_margin
			right_x2 = right_x + window_x_margin

			if (debug_mode == True):
				cv2.rectangle(debug_image, (left_x1, y1), (left_x2, y2), (0, 255, 0), 2)
				cv2.rectangle(debug_image, (right_x1, y1), (right_x2, y2), (0, 255, 0), 2)

			# We want the indices within the current window
			left_indices_within_window = (nonzero_pixel_positions_y >= y1) & (nonzero_pixel_positions_y < y2) & \
										 (nonzero_pixel_positions_x >= left_x1) & (nonzero_pixel_positions_x < left_x2)

			right_indices_within_window = (nonzero_pixel_positions_y >= y1) & (nonzero_pixel_positions_y < y2) & \
										  (nonzero_pixel_positions_x >= right_x1) & (nonzero_pixel_positions_x < right_x2)
			
			# Make sure they are non zero? (Not sure if this is necessary)				  
			left_indices_within_window = left_indices_within_window.nonzero()[0]
			right_indices_within_window = right_indices_within_window.nonzero()[0]

			final_left_line_indices.append(left_indices_within_window)
			final_right_line_indices.append(right_indices_within_window)

			# Recenter the left_x and right_x of the next sliding window if there are enough pixels
			if ((len(nonzero_pixel_positions_x[left_indices_within_window]) + 
				len(nonzero_pixel_positions_y[left_indices_within_window])) > recenter_threshold):
				left_x = np.int(np.mean(nonzero_pixel_positions_x[left_indices_within_window]))

			if ((len(nonzero_pixel_positions_x[right_indices_within_window]) + 
				len(nonzero_pixel_positions_y[right_indices_within_window])) > recenter_threshold):
				right_x = np.int(np.mean(nonzero_pixel_positions_x[right_indices_within_window]))

		final_left_line_indices = np.concatenate(final_left_line_indices)
		final_right_line_indices = np.concatenate(final_right_line_indices)

		final_left_line_pixels_x = nonzero_pixel_positions_x[final_left_line_indices]
		final_left_line_pixels_y = nonzero_pixel_positions_y[final_left_line_indices]

		final_right_line_pixels_x = nonzero_pixel_positions_x[final_right_line_indices]
		final_right_line_pixels_y = nonzero_pixel_positions_y[final_right_line_indices]

		# Fit a second order polynomial through the pixels for each line
		left_fit = np.polyfit(final_left_line_pixels_y, final_left_line_pixels_x, 2)
		right_fit = np.polyfit(final_right_line_pixels_y, final_right_line_pixels_x, 2)

		# Display debug information
		if (debug_mode == True):
			print("XXXX")
			ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
			left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
			right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

			debug_image[nonzero_pixel_positions_y[final_left_line_indices], nonzero_pixel_positions_x[final_left_line_indices]] = [255, 0, 0]
			debug_image[nonzero_pixel_positions_y[final_right_line_indices], nonzero_pixel_positions_x[final_right_line_indices]] = [0, 0, 255]

			plt.figure('Identified Lane Lines')
			plt.title('Identified Lane Lines')
			plt.plot(left_fitx, ploty, color='yellow')
			plt.plot(right_fitx, ploty, color='yellow')
			plt.imshow(debug_image)
			plt.xlim(0, 1280)
			plt.ylim(720, 0)
			plt.show()

		return left_fit, right_fit

	# Perform a lighter search for the left and right fits. This is only possible
	# if the previous left and right fits have already been calculated
	def __find_lines_lite(self, binary_image, left_fit, right_fit, debug_mode=False):
		# If there's no previous fit, we cannot do anything
		if len(left_fit) == 0 or len(right_fit) == 0:
			return [], []

		nonzero_pixel_positions = binary_image.nonzero()
		nonzero_pixel_positions_y = np.array(nonzero_pixel_positions[0])
		nonzero_pixel_positions_x = np.array(nonzero_pixel_positions[1])

		window_x_margin = 100

		# We just search around our last polynomial fit plus minus window_x_margin
		final_left_line_indices = ((nonzero_pixel_positions_x > (left_fit[0] * (nonzero_pixel_positions_y**2) + \
										left_fit[1] * nonzero_pixel_positions_y + left_fit[2] - window_x_margin)) &
									(nonzero_pixel_positions_x < (left_fit[0] * (nonzero_pixel_positions_y ** 2) + \
										left_fit[1] * nonzero_pixel_positions_y + left_fit[2] + window_x_margin)))

		final_right_line_indices = ((nonzero_pixel_positions_x > (right_fit[0] * (nonzero_pixel_positions_y**2) + \
										right_fit[1] * nonzero_pixel_positions_y + right_fit[2] - window_x_margin)) &
									(nonzero_pixel_positions_x < (right_fit[0] * (nonzero_pixel_positions_y ** 2) + \
										right_fit[1] * nonzero_pixel_positions_y + right_fit[2] + window_x_margin)))

		final_left_line_pixels_x = nonzero_pixel_positions_x[final_left_line_indices]
		final_left_line_pixels_y = nonzero_pixel_positions_y[final_left_line_indices]
		final_right_line_pixels_x = nonzero_pixel_positions_x[final_right_line_indices]
		final_right_line_pixels_y = nonzero_pixel_positions_y[final_right_line_indices]

		left_fit = np.polyfit(final_left_line_pixels_y, final_left_line_pixels_x, 2)
		right_fit = np.polyfit(final_right_line_pixels_y, final_right_line_pixels_x, 2)

		# Display debug information
		if (debug_mode == True):
			debug_image = np.dstack((binary_image, binary_image, binary_image)) * 255
			debug_overlay_image = np.zeros_like(debug_image)

			ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
			left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
			right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

			debug_image[nonzero_pixel_positions_y[final_left_line_indices], nonzero_pixel_positions_x[final_left_line_indices]] = [255, 0, 0]
			debug_image[nonzero_pixel_positions_y[final_right_line_indices], nonzero_pixel_positions_x[final_right_line_indices]] = [0, 0, 255]

			left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - window_x_margin, ploty]))])
			left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + window_x_margin, ploty])))])
			left_line_pts = np.hstack((left_line_window1, left_line_window2))

			right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - window_x_margin, ploty]))])
			right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + window_x_margin, ploty])))])
			right_line_pts = np.hstack((right_line_window1, right_line_window2))

			cv2.fillPoly(debug_overlay_image, np.int_([left_line_pts]), (0, 255, 0))
			cv2.fillPoly(debug_overlay_image, np.int_([right_line_pts]), (0, 255, 0))
			result = cv2.addWeighted(debug_image, 1, debug_overlay_image, 0.3, 0)

			
			plt.figure('Refined Search Area')
			plt.title('Refined Search Area')
			plt.imshow(result)
			plt.plot(left_fitx, ploty, color='yellow')
			plt.plot(right_fitx, ploty, color='yellow')
			plt.xlim(0, 1280)
			plt.ylim(720, 0)
			plt.show()

		return left_fit, right_fit

	# Uses find_lines_full when left and right fits are not available or are no longer valid.
	# If not, it uses find_lines_lite
	def find_lines(self, binary_top_down_image, debug_mode=False):
		self.left_fit, self.right_fit = self.__find_lines_lite(binary_top_down_image, self.left_fit, self.right_fit, debug_mode=debug_mode)
		
		# If we still don't have a fit, we need to do the full recalculation
		if (len(self.left_fit) == 0 or len(self.right_fit) == 0):			
			left_base_pos_x, right_base_pos_x = self.__find_base_x_positions(binary_top_down_image, debug_mode=debug_mode)
			self.left_fit, self.right_fit = self.__find_lines_full(binary_top_down_image, left_base_pos_x, right_base_pos_x, debug_mode=debug_mode)
		
		return self.left_fit, self.right_fit


	def generate_top_down_lane_overlay(self, binary_image, left_fit, right_fit):
		overlay_image = np.zeros_like(binary_image).astype(np.uint8)

		ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
		left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
		right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		cv2.fillPoly(overlay_image, np.int_([pts]), (0, 255, 0))
		return overlay_image

	def display_stats(self, image, left_fit, right_fit):
		# Get our x's and y's in pixel space
		ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
		left_fit_in_pixels = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
		right_fit_in_pixels = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
		left_x = left_fit_in_pixels[-1]
		right_x = right_fit_in_pixels[-1]

		lane_width_in_pixels = right_x - left_x
		y = image.shape[0]

		meters_per_pixel_x = 3.7 / lane_width_in_pixels
		meters_per_pixel_y = 30 / y

		# scale them up using our meter-to-pixel ratios for x and y,
		# then fit a new polynomial through the scaled up points
		left_fit_in_meters = np.polyfit(ploty * meters_per_pixel_y, left_fit_in_pixels * meters_per_pixel_x, 2)
		right_fit_in_meters = np.polyfit(ploty * meters_per_pixel_y, right_fit_in_pixels * meters_per_pixel_x, 2)

		# Calculate the distance from center
		# Sample at the lowest point in the image (y = image_height)
		lane_center = (left_x + right_x) / 2
		image_center = image.shape[1] / 2
		dist_from_center_in_pixels = image_center - lane_center
		dist_from_center_in_meters = dist_from_center_in_pixels * meters_per_pixel_x

		# Calculate the radius of curvature for left and right lines
		# Making sure we convert our y value to meters as well
		left_curvature_radius_in_meters = ((1 + (2 * left_fit_in_meters[0] * y * meters_per_pixel_y + left_fit_in_meters[1])**2)**1.5) / np.absolute(2 * left_fit_in_meters[0])
		right_curvature_radius_in_meters = ((1 + (2 * right_fit_in_meters[0] * y * meters_per_pixel_y + right_fit_in_meters[1])**2)**1.5) / np.absolute(2 * right_fit_in_meters[0])

		cv2.putText(image, 'Distance from lane center: {:3f}m'.format(dist_from_center_in_meters), (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
		cv2.putText(image, 'Radius of curvature: Left {:3f}m, Right {:3f}m'.format(left_curvature_radius_in_meters, right_curvature_radius_in_meters), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
		return image

################
# Unit Testing #
################
# Test by running 'python LaneAnalysis.py'
if __name__ == '__main__':
	
	from Preprocessing import combined_threshold, get_transform_matrices, transform_perspective
	
	image = mpimg.imread('test_images/test5.jpg')

	laneAnalysis = LaneAnalysis()

	binary_image = combined_threshold(image)

	transform_matrix, transform_matrix_inverse = get_transform_matrices(image)
	binary_top_down_image = transform_perspective(binary_image, transform_matrix)

	left_fit, right_fit = laneAnalysis.find_lines(binary_top_down_image, debug_mode=True)

	top_down_lane_overlay = laneAnalysis.generate_top_down_lane_overlay(image, left_fit, right_fit)
	lane_overlay = transform_perspective(top_down_lane_overlay, transform_matrix_inverse)

	output = cv2.addWeighted(image, 1.0, lane_overlay, 0.3, 0)

	output = laneAnalysis.display_stats(output, left_fit, right_fit)

	# Perform a second call, this tests the find_lines_lite() function
	left_fit, right_fit = laneAnalysis.find_lines(binary_top_down_image, debug_mode=True)

	plt.figure('Original image')
	plt.title('Original image')
	plt.imshow(image)

	plt.figure('Final output')
	plt.title('Final output')
	plt.imshow(output)

	plt.show()