import cv2
import os
import sys
import glob
import numpy as np
from skimage import exposure
from warper import Warper
from thresholder import Thresholder
from line_finder import LineFinder
from line import Line
from line_drawer import LineDrawer
from undistorter import Undistorter
from config import CAMERA_MATRIX

from moviepy.editor import VideoFileClip

warper = Warper()
thresholder = Thresholder()
left_line = Line()
right_line = Line()
line_finder = LineFinder()
line_drawer = LineDrawer()
undistorter = Undistorter(CAMERA_MATRIX)

def process_frame(frame):
	# undistort frame
	undistorted = undistorter.undistort(frame)

	# apply perspective transform
	warped = warper.warp(undistorted)

	# color & gradient thresholding
	color_binary = thresholder.color_thresh(warped)
	gradx_binary = thresholder.abs_sobel_thresh(warped, ksize=15)
	mag_binary = thresholder.mag_thresh(warped, ksize=15, thresh=(50, 250))
	dir_binary = thresholder.dir_thresh(warped, ksize=15, thresh=(0.7, 1.3))

	# combine color & gradient thresholds
	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((gradx_binary == 1) | (color_binary == 1)) &
		((mag_binary == 1) | (dir_binary == 1))] = 1

	line_finder.find_lines(combined_binary, left_line, right_line)

	out_warped = line_drawer.draw_lines(warped, left_line, right_line)

	out_unwarped = warper.unwarp(out_warped)
	result = cv2.addWeighted(undistorted, 1, out_unwarped, 0.3, 0)
	lane_curve, car_pos = line_finder.measure_curvature(undistorted, left_line, right_line)

	if car_pos > 0:
		car_pos_text = '{}m right of lane center'.format(car_pos)
	else:
		car_pos_text = '{}m left of lane center'.format(abs(car_pos))

	cv2.putText(result, "Lane curvature: {}".format(lane_curve), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
				color=(255, 255, 255), thickness=2)
	cv2.putText(result, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
				thickness=2)


	return result


def main(argv):
	video = 'project_video'
	white_output = '{}_out.mp4'.format(video)
	clip1 = VideoFileClip('{}.mp4'.format(video))
	white_clip = clip1.fl_image(process_frame)  # NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)

	
if __name__ == '__main__':
	main(sys.argv)