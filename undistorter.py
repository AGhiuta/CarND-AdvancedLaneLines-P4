import cv2
import os
import pickle
from calibrate import calibrate

class Undistorter:
	def __init__(self, cam_matrix):
		if not os.path.exists(cam_matrix):
			mtx, dist = calibrate()
		else:
			with open(cam_matrix, 'rb') as f:
				calib_info = pickle.load(f)
				self.mtx = calib_info['mtx']
				self.dist = calib_info['dist']

	def undistort(self, image):
		return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)