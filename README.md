# Vehicle safety system using OpenCV and Deep Learning

Goal: Detect pedestrians, animals and other vehicles in images taken from infrared camera mounted at the front of a car using pre-trained MobileNets Single Shot Detector model and alert the driver about the same to avoid accidents.


Pre-trained model source: https://github.com/chuanqi305/MobileNet-SSD

Required packages: python, opencv, numpy, argparse

Steps:

Step 1: Importing all the required arguments using argument parser\n
Step 2: Define classes for object detection and colours for bounding box
Step 3: Loading the pre-trained model
Step 4: Loading and Preparing image for object detection
Step 5: Detecting objects in the image using the pre-trained model (single forward pass through the neural network)
Step 6: Retrieveing the confidence, postion and other details of the objects by looping over each of the detections
Step 7: Drawing a bounding box and class of object on the image
Step 8: Print class and accuracy of the detection
Step 9: Show the output.


