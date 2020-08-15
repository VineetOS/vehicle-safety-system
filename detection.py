import cv2
import numpy as np
import argparse

#Step 1: Importing all the required arguments using argument parser

par = argparse.ArgumentParser()
par.add_argument("-m", "--model", required=True,
help="path to Caffe pre-trained model")
par.add_argument("-i", "--input", required=True,
help="path to input image")
par.add_argument("-p", "--prototxt", required=True,
help="path to Caffe 'deploy' prototxt file")
pars = vars(par.parse_args())


#Classes for object detection (taken from MobileNets)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


#Step 2: Loading the pre-trained model
train = cv2.dnn.readNetFromCaffe(pars["prototxt"], pars["model"])

#Step 3: Preparing image for object detection
f = cv2.imread(pars["input"])
(h, w) = f.shape[:2]
b = cv2.dnn.blobFromImage(cv2.resize(f, (300, 300)), 0.007843, (300, 300), 127.5)

#Step 4: Detecting objects in the image using the pre-trained model
train.setInput(b)
objects = train.forward()

#Step 5: Final processing for all the objects detected in the image
for i in np.arange(0, objects.shape[2]):

	con = objects[0, 0, i, 2]

	if con > 0.2:
		s = int(objects[0, 0, i, 1])
		mark = objects[0, 0, i, 3:7] * np.array([w, h, w, h])
		(sX, sY, eX, eY) = mark.astype("int")

		l = "{}: {:.2f}% ".format(CLASSES[s],con * 100)
		
		cv2.rectangle(f, (sX, sY), (eX, eY),
		COLORS[s], 2)
		y = sY - 15 if sY - 15 > 15 else sY + 15
		cv2.putText(f, l, (sX, y),
		cv2.FONT_HERSHEY_TRIPLEX, 0.5, COLORS[s], 2)
		print("Object Class: Model Accuracy :: {} ".format(l))

#Display the final results
cv2.imshow("Results", f)
cv2.waitKey(0)