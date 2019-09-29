#python face_align.py base

# import the necessary packages
from face_aligner import FaceAligner
from helpers import rect_to_bb
import argparse
import glob
import imutils
import dlib
import cv2

count = 0
uid = 69
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", type=str, default="all",
	help="test-images or base-image")
args = vars(ap.parse_args())

if(args["class"] == "base"):
	images = glob.glob("test_images/*")
else:
	images = glob.glob("input_dir/*")
for img in images:
	print(img)
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(img)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 2)

	# loop over the face detections
	for rect in rects:
		count+=1
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)
		faceAligned = cv2.resize(faceAligned,(160,160))
		if(args["class"] == "base"):
			if(count<10):
				cv2.imwrite("aligned_faces/base_image/user_{}_0{}.jpg".format(uid,count),faceAligned)
			else:
				cv2.imwrite("aligned_faces/base_image/user_{}_{}.jpg".format(uid,count),faceAligned)
		else:
			if(count<10):
				cv2.imwrite("aligned_faces/verify/user_{}_0{}.jpg".format(uid,count),faceAligned)
			else:
				cv2.imwrite("aligned_faces/verify/user_{}_{}.jpg".format(uid,count),faceAligned)