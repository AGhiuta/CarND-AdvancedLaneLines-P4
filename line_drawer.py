import cv2
import numpy as np
import matplotlib.pyplot as plt

class LineDrawer:
	def __init__(self):
		pass

	def add_current_xfit(self, line, fit):
		line.recent_xfitted.append(np.copy(fit))

		if len(line.recent_xfitted) > 3:
			del line.recent_xfitted[0]

	def draw_lines(self, warped, left_line, right_line):
		left_fit = left_line.current_fit
		right_fit = right_line.current_fit

		# Generate x and y values for plotting
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# save the x values of the current fit of the line
		self.add_current_xfit(left_line, left_fitx)
		self.add_current_xfit(right_line, right_fitx)

		# compute the average x values of the fitted line over the last 3 frames
		if left_line.detected == True:
			left_line.bestx = np.average(left_line.recent_xfitted, axis=0)
			right_line.bestx = np.average(right_line.recent_xfitted, axis=0)
		else:
			left_line.bestx = np.copy(left_fitx)
			right_line.bestx = np.copy(right_fitx)


		left_line.recent_xfitted[-1] = np.copy(left_line.bestx)
		right_line.recent_xfitted[-1] = np.copy(right_line.bestx)

		left_line.best_fit = np.polyfit(ploty, left_line.bestx, 2)
		right_line.best_fit = np.polyfit(ploty, right_line.bestx, 2)

		# Create an image to draw the lines on
		color_warp = np.zeros_like(warped).astype(np.uint8)

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx,
			ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

		left_line.detected = True
		right_line.detected = True

		return color_warp