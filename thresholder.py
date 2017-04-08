import cv2
import numpy as np

class Thresholder:
	def __init__(self):
		pass

	def color_thresh(self, image, thresh=(90, 255)):
		# convert the image to hls
		hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]
		h_channel = hls[:,:,0]

		color_binary = np.zeros_like(s_channel)
		color_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1]) &
			(h_channel >= 15) & (h_channel <= 100)] = 1

		return color_binary

	def abs_sobel_thresh(self, image, orient='x', ksize=3, thresh=(20, 100)):
		# convert the image to hsv
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		v_channel = hsv[:,:,2]
		# Apply x or y gradient with the OpenCV Sobel() function
		# and take the absolute value
		if orient == 'x':
		    abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=ksize))
		if orient == 'y':
		    abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=ksize))
		# Rescale back to 8 bit integer
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# Create a copy and apply the threshold
		binary_output = np.zeros_like(scaled_sobel)
		# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
		binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

		# Return the result
		return binary_output

	def mag_thresh(self, image, ksize=3, thresh=(30, 100)):
		# convert the image to hsv
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		v_channel = hsv[:,:,2]
		# Take both Sobel x and y gradients
		sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=ksize)
		sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=ksize)
		# Calculate the gradient magnitude
		gradmag = np.sqrt(sobelx**2 + sobely**2)
		# Rescale to 8 bit
		scale_factor = np.max(gradmag)/255 
		gradmag = (gradmag/scale_factor).astype(np.uint8) 
		# Create a binary image of ones where threshold is met, zeros otherwise
		binary_output = np.zeros_like(gradmag)
		binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

		# Return the binary image
		return binary_output

	def dir_thresh(self, image, ksize=3, thresh=(0, np.pi/2)):
		# convert the image to hsv
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		v_channel = hsv[:,:,2]
		# Compute the x and y gradients
		sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=ksize)
		sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=ksize)
		# Take the absolute value of the gradient direction, 
		# apply a threshold, and create a binary image result
		absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
		binary_output =  np.zeros_like(absgraddir)
		binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

		# Return the binary image
		return binary_output
