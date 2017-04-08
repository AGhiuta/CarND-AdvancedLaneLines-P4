import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class LineFinder:
	def __init__(self):
		pass

	def measure_curvature(self, frame, left_line, right_line):
		y_eval = frame.shape[0]

		ym_per_pix = 3 / 72  # meters per pixel in y dimension
		xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

		leftx = left_line.bestx
		lefty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
		rightx = right_line.bestx
		righty = np.linspace(0, frame.shape[0]-1, frame.shape[0])

		left_fit = left_line.best_fit
		right_fit = right_line.best_fit

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

		# Calculate the new radii of curvature
		left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
			left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
		right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
			right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

		tmpy = lefty * ym_per_pix
		leftx = left_fit_cr[0]*tmpy**2 + left_fit_cr[1]*tmpy + left_fit_cr[2]

		lane_leftx = left_fit[0] * (frame.shape[0] - 1) ** 2 + \
			left_fit[1] * (frame.shape[0] - 1) + left_fit[2]
		lane_rightx = right_fit[0] * (frame.shape[0] - 1) ** 2 + \
			right_fit[1] * (frame.shape[0] - 1) + right_fit[2]

		car_pos = ((frame.shape[1] / 2) - ((lane_leftx + lane_rightx) / 2)) * xm_per_pix

		curve_rad = .5  * (left_curverad + right_curverad)

		diff = abs(leftx[0] - leftx[-1])

		if(diff < .35):
			return 'Road is nearly straight', car_pos.round(2)

		return str(curve_rad.round()) + 'm', car_pos.round(2)


	def find_lines(self, binary_warped, left_line, right_line):
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Set the width of the windows +/- margin
		margin = 50
		# margin = 100

		if left_line.detected is False:
			# Take a histogram of the bottom half of the image
			histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
			# Create an output image to draw on and  visualize the result
			out_img = np.dstack((binary_warped, binary_warped, binary_warped))
			# Find the peak of the left and right halves of the histogram
			# These will be the starting point for the left and right lines
			leftx_base = np.argmax(histogram[200:300]) + 200
			rightx_base = np.argmax(histogram[1000:1100]) + 1000

			# Current positions to be updated for each window
			leftx_current = leftx_base
			rightx_current = rightx_base

			# Choose the number of sliding windows
			nwindows = 9
			# Set height of windows
			window_height = np.int(binary_warped.shape[0]/nwindows)
			# Set minimum number of pixels found to recenter window
			minpix = 50
			# Create empty lists to receive left and right lane pixel indices
			left_lane_inds = []
			right_lane_inds = []

			# Step through the windows one by one
			for window in range(nwindows):
				# Identify window boundaries in x and y (and right and left)
				win_y_low = binary_warped.shape[0] - (window+1)*window_height
				# print(leftx_current, rightx_current)
				win_y_high = binary_warped.shape[0] - window*window_height
				win_xleft_low = leftx_current - margin
				win_xleft_high = leftx_current + margin
				win_xright_low = rightx_current - margin
				win_xright_high = rightx_current + margin
				# Draw the windows on the visualization image
				cv2.rectangle(out_img, (win_xleft_low, win_y_low),
					(win_xleft_high, win_y_high), (0,1,0), 2) 
				cv2.rectangle(out_img, (win_xright_low, win_y_low),
					(win_xright_high, win_y_high), (0,1,0), 2) 
				# Identify the nonzero pixels in x and y within the window
				good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
					(nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
				good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
					(nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
				# Append these indices to the lists
				left_lane_inds.append(good_left_inds)
				right_lane_inds.append(good_right_inds)
				# If you found > minpix pixels, recenter next window on their mean position
				if len(good_left_inds) > minpix:
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				if len(good_right_inds) > minpix:        
					rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

			# Concatenate the arrays of indices
			left_lane_inds = np.concatenate(left_lane_inds)
			right_lane_inds = np.concatenate(right_lane_inds)
		else:
			left_fit = left_line.best_fit
			right_fit = right_line.best_fit

			left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +
				left_fit[1]*nonzeroy + left_fit[2] - margin)) &
				(nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
				left_fit[2] + margin))) 
			right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) +
				right_fit[1]*nonzeroy + right_fit[2] - margin)) &
				(nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
				right_fit[2] + margin)))

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		#return self.left_fit, self.right_fit
		left_line.current_fit = np.copy(left_fit)
		right_line.current_fit = np.copy(right_fit)

		left_line.allx = np.copy(leftx)
		left_line.ally = np.copy(lefty)
		right_line.allx = np.copy(rightx)
		right_line.ally = np.copy(righty)