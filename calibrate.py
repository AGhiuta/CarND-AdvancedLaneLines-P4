"""
Calibrate the camera based on the provided images
Save the distortion coefficients and the camera calibration matrix for later use
"""

import cv2
import os
import glob
import pickle
import numpy as np
from config import CAMERA_MATRIX, CAMERA_DIR, NX, NY
from config import CORNERS_DIR, UNDISTORTED_DIR

def calibrate():
	images = glob.glob('{}/calibration*.jpg'.format(CAMERA_DIR))

	# create the directory where the images with drawn corners will be saved
	if not os.path.exists(CORNERS_DIR):
		os.makedirs(CORNERS_DIR)

	# create the directory where the undistorted images will be saved
	if not os.path.exists(UNDISTORTED_DIR):
		os.makedirs(UNDISTORTED_DIR)

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	#arrays to store image points and object points from all the images
	objpoints = []	# 3D points in real world space
	imgpoints = []	# 2D points in image plane

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
	objp = np.zeros((NY * NX, 3), dtype=np.float32)
	objp[:,:2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)	# x, y coordinates

	for fname in images:
		img = cv2.imread(fname)

		# convert image to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

		# if corners are found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			# refine the corners
			corners2 = cv2.cornerSubPix(gray,corners,(11, 11),(-1, -1),criteria)
			imgpoints.append(corners2)

	        # draw and save the corners
			img = cv2.drawChessboardCorners(img, (NX, NY), corners, ret)
			cv2.imwrite('{}/{}'.format(CORNERS_DIR, fname.strip().split('/')[-1]),
					img)

	# perform the calibration
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
		gray.shape[::-1], None, None)

	# save the distortion coefficients and the camera matrix
	calib_info = {'mtx': mtx,
				'dist': dist}

	# save the undistorted calibration images
	for fname in images:
		img = cv2.imread(fname)
		dst = cv2.undistort(img, mtx, dist, None, mtx)

		cv2.imwrite('{}/{}'.format(UNDISTORTED_DIR, fname.strip().split('/')[-1]),
				dst)

	with open(CAMERA_MATRIX, 'wb') as f:
		pickle.dump(calib_info, f)

	return mtx, dist


















