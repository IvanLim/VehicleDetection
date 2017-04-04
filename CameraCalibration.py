import numpy as np
import glob
import cv2

############################
# Camera Calibration Class #
############################
# Calculates and stores the camera matrix and the distortion coefficients upon instantiation.
# Once calculated, images can be undistorted by calling instance.undistort(image)
class CameraCalibration():
	# Run through the calculations upon instantiation
	def __init__(self, image_path, num_corners_x, num_corners_y):
		print('Calibrating...')
		image_filenames_glob = glob.glob(image_path)
		print('\t-> Getting image and object points...')
		obj_points, img_points, img_size = self.__get_points(image_filenames_glob, num_corners_x, num_corners_y)
		print('\t-> Calculating camera matrix and distortion coefficients...')
		self.camera_matrix, self.distortion_coefficients = self.__get_calibrated_matrix_and_coefficients(obj_points, img_points, img_size)
		print('\t-> Done')
		print()

	# Gets the image and object points
	def __get_points(self, image_filenames_glob, num_corners_x, num_corners_y):
		correct_points = np.zeros((num_corners_x * num_corners_y,3), np.float32)
		correct_points[:,:2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1,2)

		obj_points = []
		img_points = []
		img_size = (0, 0)

		for filename in image_filenames_glob:
			# Read image and convert to grayscale
			img = cv2.imread(filename)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_size = gray.shape[::-1]

			# Find corners from checkered pattern
			ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

			if (ret == True):
				obj_points.append(correct_points)
				img_points.append(corners)

		return obj_points, img_points, img_size

	# Support function to calculate the camera matrix, and its distortion coefficients
	# We will need these to correct future camera images
	def __get_calibrated_matrix_and_coefficients(self, obj_points, img_points, img_size):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
		return mtx, dist

	# Undistort an image using the matrix and coefficients we calculated before
	def undistort(self, image):
		undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, self.camera_matrix)
		return undistorted_image


################
# Unit testing #
################
# Test by executing 'python CameraCalibration.py'
if __name__ == '__main__':

	import matplotlib.pyplot as plt

	# Instantiate class for testing
	calibration = CameraCalibration('camera_cal/calibration*.jpg', num_corners_x=9, num_corners_y=6)

	# Read image and undistort
	testimg = cv2.imread('camera_cal/calibration1.jpg')
	newimg = calibration.undistort(testimg)

	# Display output
	plt.figure('Original image')
	plt.title('Original image')
	plt.imshow(testimg)	

	plt.figure('Undistorted image')
	plt.title('Undistorted image')
	plt.imshow(newimg)

	plt.show()