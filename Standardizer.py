# Standardizes image loading and normalization etc
# This is because matplotlib.image's imread function
# reads png and jpg files differently. This messes with
# our classification and calculations if not handled properly
import matplotlib.image as mpimage
import numpy as np
import cv2

def read_image(path):
	image = mpimage.imread(path)
	
	if path.endswith('.png'):
		image = scale_image_to_255(image)
	
	return image

def convert_color(image, cv2_conversion_constant):
	image = cv2.cvtColor(image, cv2_conversion_constant)
	return image

def scale_image_to_zero_one(image):
	image = np.float32(image / 255)
	return image

def scale_image_to_255(image):
	image = np.uint8(image * 255)
	return image



################
# Unit testing #
################
# Test by running 'python Standardizer.py'
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import Standardizer

	testimg1 = Standardizer.read_image('test_images/test_png.png')
	testimg2 = Standardizer.read_image('test_images/test_jpg.jpg')

	plt.figure()
	plt.title('PNG Image')
	plt.imshow(testimg1)

	plt.figure()
	plt.title('JPG Image')
	plt.imshow(testimg2)

	plt.show()