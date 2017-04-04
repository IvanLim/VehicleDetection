import numpy as np
import cv2

###################################
# Preprocessing Support functions #
###################################
def sobel_threshold(gray_image, orientation='x', sobel_kernel=3, threshold=(0, 255)):
	sobel = []
	if orientation == 'y':
		sobel = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	elif orientation == 'x':
		sobel = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	else:
		print('Invalid orientation. Expecting values "x" or "y"')
		return None

	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

	binary_output = np.zeros_like(gray_image)
	binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1	
	return binary_output

def gradient_magnitude_threshold(gray_image, sobel_kernel=3, threshold=(0, 255)):
	sobelX = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobelY = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	magnitude = np.sqrt(sobelX ** 2 + sobelY ** 2)	
	scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))

	binary_output = np.zeros_like(gray_image)
	binary_output[((scaled_magnitude >= threshold[0]) & (scaled_magnitude <= threshold[1]))] = 1
	return binary_output

def extract_S_channel(image):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	S = hls[:, :, 2]
	return S

def color_threshold(s_channel_image, threshold=[0, 255]):
	binary_output = np.zeros_like(s_channel_image)
	binary_output[(s_channel_image >= threshold[0]) & (s_channel_image <= threshold[1])] = 1
	return binary_output

def get_source_points(image):
	h = image.shape[0]
	w = image.shape[1]

	sx1 = int(np.round(w / 2.15))
	sx2 = w - sx1
	sx4 = w // 7
	sx3 = w - sx4
	sy1 = sy2 = int(np.round(h / 1.6))
	sy3 = sy4 = h

	dx1 = dx4 = int(np.round(w / 4))
	dx2 = dx3 = w - dx1
	dy1 = dy2 = 0
	dy3 = dy4 = h

	src_points = np.float32([[sx1, sy1],[sx2, sy2], [sx3, sy3], [sx4, sy4]])
	dst_points = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])

	return src_points, dst_points

def get_transform_matrices(image):
	src_points, dst_points = get_source_points(image)
	transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
	transform_matrix_inverse = cv2.getPerspectiveTransform(dst_points, src_points)
	return transform_matrix, transform_matrix_inverse

def transform_perspective(image, transform_matrix):
	img_size = (image.shape[1], image.shape[0])
	warped = cv2.warpPerspective(image, transform_matrix, img_size, flags=cv2.INTER_LINEAR)
	return warped

def equalize_histogram_color(image):
	yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
	output = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
	return output

def combined_threshold(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	s_channel = extract_S_channel(image)

	equalized_hist_image = equalize_histogram_color(image)
	equalized_hist_image_s_channel = extract_S_channel(equalized_hist_image)

	sobelX = sobel_threshold(gray, orientation='x', sobel_kernel=3, threshold=(10, 100))
	sobelY = sobel_threshold(gray, orientation='y', sobel_kernel=3, threshold=(10, 100))
	magnitude = gradient_magnitude_threshold(gray, sobel_kernel=3, threshold=(30, 100))

	color = color_threshold(s_channel, threshold=(100, 255))
	equalized_hist_color = color_threshold(equalized_hist_image_s_channel, threshold=(250, 255))

	binary_output = np.zeros_like(gray)
	binary_output[((sobelX == 1) & (sobelY == 1)) & ((magnitude == 1)) | (color == 1) & (equalized_hist_color == 1)] = 1
	return binary_output


################
# Unit testing #
################
# Test by executing 'python Preprocessing.py'
if __name__ == '__main__':

	import matplotlib.pyplot as plt
	import matplotlib.image as mpimage

	testimg = mpimage.imread('test_images/test5.jpg')
	equalized_hist = equalize_histogram_color(testimg)
	
	S = extract_S_channel(testimg)
	equalized_hist_S = extract_S_channel(equalized_hist)

	gray = cv2.cvtColor(testimg, cv2.COLOR_RGB2GRAY)

	sobel_x_thresholded_image = sobel_threshold(gray, orientation='x', sobel_kernel=3, threshold=(10, 100))
	sobel_y_thresholded_image = sobel_threshold(gray, orientation='y', sobel_kernel=3, threshold=(10, 100))
	gradient_magnitude_thresholded_image = gradient_magnitude_threshold(gray, sobel_kernel=3, threshold=(30, 100))
	color_thresholded_image = color_threshold(S, threshold=(100, 255))
	combined_thresholded_image = combined_threshold(testimg)
	transform_matrix, transform_matrix_inverse = get_transform_matrices(testimg)
	
	# Test preprocessing steps
	plt.figure('Original image')
	plt.title('Original image')
	plt.imshow(testimg)

	plt.figure('Original: S Channel')
	plt.title('Original: S Channel')
	plt.imshow(S)

	plt.figure('Equalized histogram')
	plt.title('Equalized histogram')
	plt.imshow(equalized_hist)

	plt.figure('Equalized histogram: S Channel')
	plt.title('Equalized histogram: S Channel')
	plt.imshow(equalized_hist_S)

	plt.figure('Sobel x thresholded image')
	plt.title('Sobel x thresholded image')
	plt.imshow(sobel_x_thresholded_image)

	plt.figure('Sobel y thresholded image')
	plt.title('Sobel y thresholded image')
	plt.imshow(sobel_y_thresholded_image)

	plt.figure('Gradient magnitude thresholded image')
	plt.title('Gradient magnitude thresholded image')
	plt.imshow(gradient_magnitude_thresholded_image)

	plt.figure('Color thresholded image')
	plt.title('Color thresholded image')
	plt.imshow(color_thresholded_image)

	plt.figure('Combined thresholded image')
	plt.title('Combined thresholded image')
	plt.imshow(combined_thresholded_image)

	# Test capturing source and destination points for perspective transform
	scolor = [255, 0, 0]
	dcolor = [0, 0, 255]
	thickness = 5
	w = testimg.shape[1]
	h = testimg.shape[0]

	lineimage = np.copy(testimg)
	sx1 = int(np.round(w / 2.15))
	sx2 = w - sx1
	sx4 = w // 7
	sx3 = w - sx4
	sy1 = sy2 = int(np.round(h / 1.6))
	sy3 = sy4 = h
	cv2.line(lineimage, (sx1, sy1), (sx2, sy2), scolor, thickness)
	cv2.line(lineimage, (sx2, sy2), (sx3, sy3), scolor, thickness)
	cv2.line(lineimage, (sx3, sy3), (sx4, sy4), scolor, thickness)
	cv2.line(lineimage, (sx4, sy4), (sx1, sy1), scolor, thickness)
	plt.figure('Perspective transform: Source points')
	plt.title('Perspective transform: Source points')
	plt.imshow(lineimage)


	transformed_perspective = transform_perspective(testimg, transform_matrix)
	lineimage = np.copy(transformed_perspective)
	dx1 = dx4 = int(np.round(w / 4))
	dx2 = dx3 = w - dx1
	dy1 = dy2 = 0
	dy3 = dy4 = h
	cv2.line(lineimage, (dx1, dy1), (dx2, dy2), dcolor, thickness)
	cv2.line(lineimage, (dx2, dy2), (dx3, dy3), dcolor, thickness)
	cv2.line(lineimage, (dx3, dy3), (dx4, dy4), dcolor, thickness)
	cv2.line(lineimage, (dx4, dy4), (dx1, dy1), dcolor, thickness)
	plt.figure('Perspective transform: Output')
	plt.title('Perspective transform: Output')
	plt.imshow(lineimage)

	plt.show()