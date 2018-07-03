#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
from pyimagesearch.facedetector import FaceDetector
from pyimagesearch import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
args = vars(ap.parse_args())

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

# construct the face detector and allow the camera to warm
# up
fd = FaceDetector(args["face"])
faceCascade = cv2.CascadeClassifier(args["face"])
time.sleep(0.1)

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = cv2.imread(image_path)
        resized = imutils.resize(image_pil, width = 300)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        image = np.array(gray, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './newfaces'
#path2 = './Test'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))
print "Training done" 

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
print "Training done"
# initialize the camera and grab a reference to the raw camera
# capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image
	frame = f.array

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the image and then clone the frame
	# so that we can draw on it
	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
		minSize = (30, 30))
	frameClone = frame.copy()

	# loop over the face bounding boxes and draw them
	for (fX, fY, fW, fH) in faceRects:
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        predict_image = np.array(gray, 'uint8')
        for (x, y, w, h) in faceRects:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            nbr_actual = 1
            if conf < 100:
                if nbr_predicted == 1:
                    print "{} is Correctly Recognized as Rishabh with confidence {}".format(nbr_actual, conf)
                elif nbr_predicted == 2:
                    print "{} is correctly recognized as Shivank with confidence {}".format(nbr_actual, conf)
                elif nbr_predicted == 3:
                    print "{} is correctly recognized as Sonali with confidence {}".format(nbr_actual, conf)    
            else:
                print "{} is Incorrect Recognized"
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            nbr_actual = nbr_actual+1
        nbr_actual = 1
        # show our detected faces, then clear the frame in
	# preparation for the next frame
	cv2.imshow("Face", frameClone)
	rawCapture.truncate(0)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
    
        cv2.waitKey(1000)
print "Training done"


#Checking changes are made or not