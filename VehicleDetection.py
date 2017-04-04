# General framework imports
from moviepy.editor import VideoFileClip

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from skimage.feature import hog

import matplotlib.image as mpimage
import matplotlib.pyplot as plt

import numpy as np
import cv2

from pathlib import Path
import pickle
import time

# Vehicle Detection Project module imports
from DatasetManager import DatasetManager
from FeatureExtractor import FeatureExtractor
from ImageSearcher import ImageSearcher
from InterFrameLogic import InterFrameLogic
import Standardizer

# Advanced Lane Line Project module imports
from Preprocessing import combined_threshold, get_transform_matrices, transform_perspective
from CameraCalibration import CameraCalibration
from LaneAnalysis import LaneAnalysis

CALIBRATION_FILES = 'camera_cal/calibration*.jpg'
SAVE_FILENAME = 'save.p'


#############################
# Image processing pipeline #
#############################
def process_image(image):

	# Undistort the image
	image = calibration.undistort(image)

	# Since the InterFrameLogic is a subclass of the Borg class,
	# instantiating it just gives us access to the shared dictionary 
	# without having to declare variables as global.
	inter_frame_logic = InterFrameLogic()

	#####################
	# Vehicle detection #
	#####################
	scaler = inter_frame_logic.scaler
	classifier = inter_frame_logic.classifier

	# First pass detection:
	# Run a sliding window across a region of the image and test it against the classifier we trained
	first_pass_boxes = image_searcher.find_cars(image, scaler, classifier, feature_extractor, confidence_threshold=0.9)

	# Build a heat map from the initial list of boxes
	current_frame_heat_map = image_searcher.build_heat_map(image, first_pass_boxes)

	# If we have a heat map from the previous frame, average it with this one to smooth things out
	if inter_frame_logic.last_frame_heat_map != None:
		current_frame_heat_map = (current_frame_heat_map + inter_frame_logic.last_frame_heat_map) / 2

	# Save a copy of the new heat map, to be used in the next frame	
	inter_frame_logic.last_frame_heat_map = current_frame_heat_map

	# Identify bounding boxes based on the combined information on the heat map
	second_pass_boxes = image_searcher.find_boxes_from_heatmap(current_frame_heat_map)

	# Smooth out the detected boxes over several frames
	final_boxes = inter_frame_logic.average_boxes_across_frames(second_pass_boxes)


	##################
	# Lane detection #
	##################

	# Preprocess the image with colour and gradient thresholding
	binary_image = combined_threshold(image)

	# Calculate our transform matrix and its inverse
	transform_matrix, transform_matrix_inverse = get_transform_matrices(image)

	# Transform the image into a top down view for analysis
	binary_top_down_image = transform_perspective(binary_image, transform_matrix)

	left_fit, right_fit = lane_analysis.find_lines(binary_top_down_image)

	# Now that the analysis has produced the best polynomial fits 
	# for the left and right lane lines, we generate an overlay to
	# show our results
	top_down_lane_overlay = lane_analysis.generate_top_down_lane_overlay(image, left_fit, right_fit)

	# Transform the top down overlay to the same perspective as the original image
	lane_overlay = transform_perspective(top_down_lane_overlay, transform_matrix_inverse)

	# Combine the overlay with the original image
	output = cv2.addWeighted(image, 1.0, lane_overlay, 0.3, 0)
	output = lane_analysis.display_stats(output, left_fit, right_fit)

	# Draw bounding boxes over the detected vehicles
	output = image_searcher.draw_bounding_boxes(output, final_boxes, color=(0, 255, 0))	

	return output



##############
# Main logic #
##############
# Set up camera calibration
calibration = CameraCalibration(image_path=CALIBRATION_FILES, num_corners_x=9, num_corners_y=6)

# Instantiate our objects
lane_analysis = LaneAnalysis()
dataset_manager = DatasetManager()
feature_extractor = FeatureExtractor()
image_searcher = ImageSearcher()
inter_frame_logic = InterFrameLogic(first_init=True)
scaler = StandardScaler()

# If there's no pre-trained classifier, train one based on our available images
classifier_file = Path(SAVE_FILENAME)
if not classifier_file.is_file():
	print('Building Training and Test Datasets...')
	training_batch_generator, test_batch_generator = dataset_manager.get_dataset_generators()
	train_features, train_labels = dataset_manager.get_features_and_labels(training_batch_generator, feature_extractor, 'training')
	test_features, test_labels = dataset_manager.get_features_and_labels(test_batch_generator, feature_extractor, 'test')

	print('Scaling features...')
	scaler.fit(train_features)
	train_features = scaler.transform(train_features)
	test_features = scaler.transform(test_features)
	print('\t-> Done')
	print()

	print('Training Classifier...')
	classifier = CalibratedClassifierCV(LinearSVC())
	t1 = time.time()
	classifier.fit(train_features, train_labels)
	t2 = time.time()
	print('\tTraining time: {:.3f} secs'.format(t2 - t1))
	print('\tTest accuracy: {:.4f}'.format(classifier.score(test_features, test_labels)))
	print()

	print('Saving classifier...')
	with open(SAVE_FILENAME, 'wb') as file:
		saveItem = dict()
		saveItem['classifier'] = classifier
		saveItem['scaler'] = scaler
		pickle.dump(saveItem, file)
	print('\t-> Done')
	print()

else:
	# If there's a pre-trained classifier, load it to save time
	print('Pre-trained classifier exists. Loading...')
	with open(SAVE_FILENAME, 'rb') as file:
		loadItem = pickle.load(file)
		classifier = loadItem['classifier']
		scaler = loadItem['scaler']
	print('\t-> Done')
	print()

# Set up our loaded/trained classifier and scaler
inter_frame_logic.scaler = scaler
inter_frame_logic.classifier = classifier

# Process project video, using our image processing pipeline
print('Processing video...')
video_clip = VideoFileClip('project_video.mp4')
processed_clip = video_clip.fl_image(process_image)
processed_clip.write_videofile('project_video_output.mp4', audio=False)


# video_clip = VideoFileClip('project_video.mp4')
# sub_clip = video_clip.subclip(0, 2)
# processed_clip = sub_clip.fl_image(process_image)
# processed_clip.write_videofile('slice_video_output.mp4', audio=False)

