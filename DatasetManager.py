from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.ndimage as ndimage
import Standardizer
import numpy as np
import cv2
import glob


# No point labeling with a string when a boolean will do
IS_NON_VEHICLE = False
IS_VEHICLE = True

DATASETS = [('data/vehicles/**/*', IS_VEHICLE),
			('data/non-vehicles/**/*', IS_NON_VEHICLE)]

TRAIN_TEST_RATIO = (0.95, 0.05)

BATCH_SIZE = 1024

class DatasetManager():

	# Returns generators for the training and test sets
	def get_dataset_generators(self):
		train_files, train_labels, test_files, test_labels = self.__get_datasets(DATASETS, TRAIN_TEST_RATIO)

		train_generator = self.__batch_generator(train_files, train_labels, BATCH_SIZE)
		test_generator = self.__batch_generator(test_files, test_labels, BATCH_SIZE)

		return train_generator, test_generator

	# Gets the list of filenames in the dataset, and splits them
	def __get_datasets(self, dataset_path_and_label_list, train_test_ratio):		
		# Build a list of all filenames and labels and splits them into training and test sets
		train_files = []
		train_labels = []

		test_files = []
		test_labels = []

		for dataset_path, dataset_label in dataset_path_and_label_list:
			image_filenames_list = glob.glob(dataset_path)
			num_files = len(image_filenames_list)

			if dataset_label == IS_VEHICLE:
				image_labels = np.ones(num_files, dtype=bool)
			else:
				image_labels = np.zeros(num_files, dtype=bool)

			temp_train_files, temp_test_files, temp_train_labels, temp_test_labels = train_test_split(image_filenames_list, image_labels, test_size=train_test_ratio[1])

			train_files.extend(temp_train_files)
			train_labels.extend(temp_train_labels)

			test_files.extend(temp_test_files)
			test_labels.extend(temp_test_labels)

			print('\t{} -> {} files ({})'.format(dataset_path, num_files, 'Vehicles' if dataset_label == True else 'Non-vehicles'))

		# Give it a good shuffle before we're done
		train_files, train_labels = shuffle(train_files, train_labels)
		test_files, test_labels = shuffle(test_files, test_labels)

		print()
		print('\t{} samples in training set'.format(len(train_files)))
		print('\t{} samples in test set'.format(len(test_files)))
		print()

		return train_files, train_labels, test_files, test_labels

	# Returns a generator that returns a list of images and labels in batches
	def __batch_generator(self, sample_files, labels, batch_size=512):
		num_samples = len(sample_files)

		batch_sample_files = []
		batch_labels = []

		for offset in range(0, num_samples, batch_size):
			end = offset + batch_size			
			batch_sample_files = sample_files[offset:end]			
			batch_labels = labels[offset:end]

			# Load the images for this batch
			batch_images = []
			# batch_augmented_images = []
			# batch_augmented_labels = []

			for sample_file, label in list(zip(batch_sample_files, batch_labels)):
				image = Standardizer.read_image(sample_file)
				batch_images.append(image)

				# augmented_images, augmented_labels = self.__generate_augmented_data(image, label)
				# batch_augmented_images.extend(augmented_images)
				# batch_augmented_labels.extend(augmented_labels)

			# batch_images.extend(batch_augmented_images)
			# batch_labels.extend(batch_augmented_labels)

			yield batch_images, batch_labels

		return

	def get_features_and_labels(self, generator, feature_extractor, description):
		print('Building up {} features...'.format(description))
		features = []
		labels = []
		for batch in generator:
			batch_features = []
			batch_labels = []

			for image, label in list(zip(batch[0], batch[1])):
				image_features = feature_extractor.extract_features(image)
				batch_features.append(image_features)
				batch_labels.append(label)

			features.extend(batch_features)
			labels.extend(batch_labels)
		print('\t-> Done')
		print()
		return features, labels


	def __generate_augmented_data(self, image, label):
		augmented_images = []
		augmented_labels = []

		# Shifting all directions
		for offset in range(4, 8, 4):
			# shifted_image_u = ndimage.interpolation.shift(image, (0, -1 * offset, 0))
			# shifted_image_d = ndimage.interpolation.shift(image, (0, offset, 0))
			shifted_image_l = ndimage.interpolation.shift(image, (offset, 0, 0))
			shifted_image_r = ndimage.interpolation.shift(image, (-1 * offset, 0, 0))
			augmented_images.extend([shifted_image_l, shifted_image_r])
			augmented_labels.extend([label, label])

		# Flipping
		# Flip original image
		# original_flipped = np.fliplr(image)
		# augmented_images.extend([original_flipped])
		# augmented_labels.extend([label])

		# # Flip augmented images
		# for aug_image in augmented_images:
		# 	flipped = np.fliplr(aug_image)
		# 	augmented_images.extend([flipped])
		# 	augmented_labels.extend([label])

		return augmented_images, augmented_labels

##############
# Unit tests #
##############
# Test functions by running 'python DatasetManager.py'
if __name__ == '__main__':
	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt
	import random

	datasetManager = DatasetManager()
	train_generator, test_generator = datasetManager.get_dataset_generators()

	for train_batch_images, train_batch_labels in train_generator:	
		if (random.randint(0, 1000) == 3):
			plt.figure()
			plt.title('Vehicle' if train_batch_labels[0] == True else 'Non vehicle')
			plt.imshow(train_batch_images[0])
			plt.show()
			break