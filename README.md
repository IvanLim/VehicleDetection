# Vehicle Detection Project Report

## The goals / steps of this project are the following:

- To use **classical** machine learning techniques to detect vehicles in a video
- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Apply a color transform and add additional features (binned color features, histograms of color, canny edges) to the feature vector
- Implement a sliding-window technique and use a trained classifier to search for vehicles in images.
- Run the pipeline on a video stream **project\_video.mp4** and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate bounding boxes for vehicles detected and draw them on the image.

This project is graded according to the criteria in the [project rubric](https://review.udacity.com/#!/rubrics/513/view).

## Project files

The project has two separate functions, and contain the following files:

**Vehicle Detection**

- **VehicleDetection.py** contains the main program logic and the image processing pipeline
- **DatasetManager.py** builds and preps the data for classifier training
- **FeatureExtractor.py** contains feature extraction functions
- **ImageSearcher.py** searches for cars in an image
- **InterFrameLogic.py** handles inter frame logic, and smoothing
- **Standardizer.py** standardizes image handling between .png and .jpg images

**Lane Finding**

- **CameraCalibration.py** contains camera calibration / distortion correction support functions
- **Preprocessing.py** contains image pre-processing support functions
- **LaneAnalysis.py** contains the analysis support functions to find / fit the lanes
- **project\_video.mp4** is a video identifying the lanes, distance from center, and curvature radii
- **writeup\_report.pdf** is a report summarizing the project results

### Running the program

To run the main program (which processes and builds the project video), you can run:

**python VehicleDetection.py**

### Testing support functions

The support functions can also be tested by running:

- **python CameraCalibration.py**
- **python Preprocessing.py**
- **python LaneAnalysis.py**
- **python DatasetManager.py**
- **python FeatureExtractor.py**
- **python ImageSearcher.py**
- **python InterFrameLogic.py**
- **python Standardizer.py**

## Lane Detection Image Processing Pipeline

### Quick note on Lane Detection

This project reuses camera calibration and lane detection/analysis code from the **Advanced Lane Line Project**. For a detailed walkthrough of the lane detection pipeline, do take a look at its [project repository](https://github.com/IvanLim/advanced-lane-line). The lane detection pipeline will not be discussed here.

## Vehicle Detection Image Processing Pipeline

### Pipeline Overview

There are 3 key parts to the pipeline: **Training** , **Detection** , and **Smoothing**. A classifier is trained to detect cars from different angles. For each frame of a video, a selected region is sliced up and each slice is fed to the classifier. The classifier then determines if that slice is a car. The combined classification results are then filtered and smoothed across frames to reduce multiple detections and false positives.

## Training

The training code only runs when it is needed. If a trained classifier already exists (the **save.p** file), then the training process is skipped. I used a Linear Support Vector Classifier, **LinearSVC()** with a **CalibratedClassifierCV()** wrapped around it so that I can get confidence values together with the classification.

### Data Preparation

This is handled in **DatasetManager.py.**

Training data consists of a collection of **64x64**** Vehicle **and** Non-Vehicle **images, taken from the GTI and KITTI datasets. Given the number of images, and the possibility that the datasets might reach several GB in size, I wrote generators to load the data in batches. A portion of the KITTI dataset is also kept as a** test set** to make sure we can catch the classifier if it overfits.

### Feature Extraction

This is handled in **FeatureExtractor.py.**

The feature vector is extracted via the following steps:

We start with a **64x64** image, and convert it from **RGB** to the **YCrCb colour space**. This seems to be best suited for feature detection.

Next, we calculate a **Histogram of Oriented Gradients** from the image using sklearn&#39;s **hog()** function. The hog calculation is done with **9 orientation bins** , **4 pixels per cell** , and **2 cells per block**. These numbers were selected because they produced better performance (the accuracy did not vary much when changing these numbers by small amounts). The HOGs were also calculated across **all 3 channels**.

![Original image](https://github.com/IvanLim/vehicle-detection/report/original.png "Original image")

![HOG image](https://github.com/IvanLim/vehicle-detection/report/hog_image.png "HOG image")


Another useful feature to extract was the **spatial binning of colour** , which is a fancy term for downsampling the image and using the pixels as a feature (since each pixel now represents a group of pixels in the higher resolution image).

[Normal image], [Downsampled image]

**Colour histograms** were also calculated and used as a feature. Each channel is fed into numpy&#39;s **histogram()** function, and the final results are concatenated into a single vector.

[Y Histogram][Cr Histogram][Cb Histogram]

The final feature was edges detected using **Canny edge detection**. OpenCV&#39;s **canny()** function was used on the Y (Greyscale) channel of the image and thresholded between **80** and **100**.

[Normal image][Canny edges]

The features were then concatenated into a final feature vector, and scaled using **sklearn&#39;s StandardScaler()**. The sklearn documentation says it &quot;Standardizes features by removing the mean and scaling to unit variance&quot;. Which sounds fancy, but really we&#39;re only doing a few things:

- calculating the **mean** , and the **standard deviation** from the data
- **subtracting the mean** from every element so each value now represents the **distance from the average**
- **dividing by the standard deviation**

so after scaling, a value X in the feature vector just says **I am X standard deviations from the my average**. The scale of the data (whether it&#39;s in the 10&#39;s or the 1000&#39;s) no longer matters. Our feature vector is now standardized.

### Note on Previous (failed) Attempts

While trying to speed things up, I tried both the **OpenCV version of HOG** , and **Haar Cascades**. For the OpenCV HOG, there was a significant speed boost for HOG feature extraction (about 30x over the SKLearn&#39;s HOG implementation). However, it produced a collapsed vector, which means that I could not do subsampling, and had to call the function for every window. This was ultimately less efficient that extracting the HOG features once, and subsampling the results.

I also gave **Haar Cascades** a try, but the training time was too long. I tried training my own, and after a DAY, I still had not finished training a single stage, and the full training process would need 20 stages, which means I would need almost a month of training, which is too long. As a workaround, I used the pre-trained haar cascade classifiers from [this repository](https://github.com/abhi-kumar/CAR-DETECTION). Despite being blazingly fast, it could not detect the black car in the project video, and the detection accuracy on the white car was somewhat inconsistent. I decided to drop it, unless OpenCV provides a GPU accelerated Haar cascade training in the future to cut down training time (currently it&#39;s CPU based).

## Detection

### Sliding Window

A simple 64x64 sliding window was used to slide across the bottom region of the image (where we expect cars to be), progressing 16 pixels every step. At each point, an image patch is obtained, and features are extracted and a feature vector is built (see previous section on Feature Extraction for details). This is then fed into the classifier, which determines if the image is a car or not, and the confidence level associated with it.

### HOG Subsampling

Because sklearn&#39;s hog() function is an expensive operation, one minor optimization would be to only calculate the HOG for the entire search region **once**. After that, the hog output is subsampled to get the HOGs for a given 64x64 image patch. This is a lot more efficient than recalculating the hog for every window.

## Smoothing and Filtering

To help reduce false positives, the following techniques were applied:

### Filtering by Classification Confidence

Wrapping the LinearSVC() classifier inside a CalibratedClassifierCV() gives me access to the **predict\_proba()** function, which gives me the classfication probabilities. These probabilities are then filtered. Anything with less than 90% confidence is removed from the results.

### Heatmaps

Initial detections are drawn onto a blank image, forming a heat map over time. To smooth things out, I average the latest heat map with the heat map from the previous frame (if available). A threshold is applied to weed out weak detections. A list of bounding boxes covering the &#39;hot&#39; areas are then extracted from this heat map.

[Heat map example]

### Filtering and averaging

The detected bounding boxes are then collected over 8 frames, and if a bounding box only shows up in 3 out of the 8 frames, then it&#39;s likely to be a false positive, so we drop it. The remaining bounding boxes are then averaged to get the final results.



## Discussion

### Thoughts and Observations

If you look at the project video, the bounding boxes around the cars can sometimes be much larger compared to the car. I spent time tweaking the parameters for better results, but the alternative would have been stronger filtering and smaller bounding boxes. Between a larger bounding box (causing the car the slow down earlier), and a smaller bounding box (an accident might occur), I chose the safer option.

The Haar Cascade Classifier was really fast, it is a pity that it takes such a long time to train it. OpenCV&#39;s hog function was also about 30x faster than the sklearn version as well, but the fact that it produced a collapsed vector made it impossible to subsample so the performance gain isn&#39;t there.

The current solution also does not detect overlapping cars properly. They just blend into a giant blob. This would be really bad news in the event that the driver is stuck in a bumper-to-bumper traffic jam across multiple lanes.
