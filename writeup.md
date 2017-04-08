##Writeup Template
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./undistorted_cal/calibration1.jpg "Undistorted"
[image2]: ./test_images/test6.jpg "Road"
[image2b]: ./test_images_undistorted/test6.jpg "Road Transformed"
[image3]: ./test_images_color_thresh/test6.jpg "Color Thresh"
[image4]: ./test_images_grad_thresh/test6.jpg "Grad Thresh"
[image5]: ./test_images_mag_thresh/test6.jpg "Mag Thresh"
[image6]: ./test_images_dir_thresh/test6.jpg "Dir Thresh"
[image7]: ./test_images_combined_thresh/test6.jpg "Combined Thresh"
[image8]: ./test_images_undistorted_roi/test6.jpg "Undistorted ROI"
[image9]: ./test_images_warped/test6.jpg "Warped"
[image10]: ./test_images_warped_lines/test6.jpg "Warped lines"
[image11]: ./test_images_lanes/test6.jpg "Lanes"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration step is contained in `calibrate.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I also define a `termination criteria` for corners refinement, which is done in code line 50.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In the previous step, the calibration matrix and the distortion coefficients were computed and saved on disk in a pickle file. They are then loaded every time an `Undistorter` (defined in `undistorter.py`) object is instantiated. This makes it trivial to undistort any image taken with the same camera for which the calibration matrix and distortion coefficients were computed. I applied distortion correction to the road test images and obtained this result:

![alt text][image2b]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I applied thresholding on the perspective transformed images (described in Step 3), as I thought that by reversing the 2 steps (thresholding and perspective transform), I would get more noise-free images.

I used a combination of color and gradient thresholds to generate a binary image. I applied thresholding on the H (thresh_min=15, thresh_max=100) and S (thresh_min=90, thresh_max=255) channels of the HLS-converted image, and combined them to obtain a binary mask of the color-thresholded image, as follows:

	```
	color_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1]) & \
			(h_channel >= 15) & (h_channel <= 100)] = 1
	```

This is done in the `color_thresh` method of the `Thresholder` class (defined in `thresholder.py`). The method is called on code line 33 of the `main.py` source file.

Here's an example of my output for color thresholding applied on one of the test images:

![alt text][image3]

Next, I apply thresholding on the absolute value of the Sobel taken on the X axis. I used a kernel size of 15 for the Sobel operator and threshold values of [20, 100], as these values yielded the most crisp and solid lines across all test images. The source code of this step lies within the `abs_sobel_thresh` method of the `Thresholder` class in `thresholder.py', and the method is called on code line 34 in `main.py`.

Here's an example of my output for gradient thresholding applied on one of the test images:

![alt text][image4]

I apply gradient magnitude thresholding with kernel size 15 for the Sobel operator and threshold values of [50, 250], as these values yielded the most crisp and solid lines across all test images. The source code of this step lies within the `mag_thresh` method of the `Thresholder` class in `thresholder.py` and the method is called on code line 35 in `main.py`.

Here's an example of my output for gradient magnitude thresholding applied on one of the test images:

![alt text][image5]

Finally, I apply thresholding on the direction of the gradient. The values I use are 15 for kernel size and [0.7, 1.3] for threshold. This is done in the `dir_thresh` method of the `Thresholder` class in `thresholder.py` and the method is called in code line 36 in `main.py`

Here's an example of my output for gradient direction thresholding applied on one of the test images:

![alt text][image6]

In the end, I combine these thresholded binary masks to obtain the final output, as follows:

	```
	combined_binary[((gradx_binary == 1) | (color_binary == 1)) &
		((mag_binary == 1) | (dir_binary == 1))] = 1
	```

Here's an example of my output for combined thresholds applied on one of the test images:

![alt text][image7]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a class called `Warper`, (warper.py), with two methods: `warp()` and `unwarp()` and class variables `src`, `dst` for the source and destination points and `M` and `Minv` for the projection matrices. The `warp()` and `unwarp` methods take as input an image (`image`) and apply `warpPerspective` from OpenCV to transform the image.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 75), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 6), 0],
    [(img_size[0] / 6), img_size[1]],
    [(img_size[0] * 5 / 6), img_size[1]],
    [(img_size[0] * 5 / 6), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 213, 0        | 
| 203, 720      | 213, 720      |
| 1127, 720     | 1067, 720      |
| 715, 460      | 1067, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]
![alt text][image9]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify lane-line pixels, I generated a histogram of the bottom half of the image (code line 65 in `line_finder.py`), where the x-axis represented each pixel along the width of the image, and the y-axis represented the total number of pixels which where 1.

Next, I chose a range for each of the left and right lines, where they were most likely to be found ([200, 300] for left line and [1000, 1100] for right line) and I identified the (x, y) coordinates where the histogram peaked for each of the two intervals.

Finally, I fit my lane lines with a 2nd order polynomial:

![alt text][image10]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I computed the radius of curvature of the lane and the position of the vehicle with respect to center in the `measure_curvature` method of the `LineFinder` class. I implemented the ideas described in Lesson35:

	1. Fit the polynomial line
	2. Define the y-value where the radius is computed as the bottom of the image (frame.shape[0])
	3. Define the conversion from pixel-space to meters
	4. Compute the radius value with the formula from Lesson 35
	5. Compute the offset from the center of the lane as the distance between the bottom-center of the image ([1280, 640]) and the middle of the distance between the left and right lines ([1280, (leftx + rightx) / 2]).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 43 through 60 in my code in `main.py` in the function `process_frame()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

For implementing this project, I took the classic approach, consisting of the following steps:

	1. Calibrate the camera and perform distortion correction
	2. Apply perspective transform to warp the images into a birds-eye view
	3. Convert the image to HLS and apply thresholding on H ans S channels
	4. Convert the image to HSV and apply x-axis gradient thresholding on the V channel
	5. Convert the image to HSV and apply gradient magnitude thresholding on the V channel
	6. Convert the image to HSV and apply gradient direction thresholding on the V channel
	7. Combine the color and gradiend thresholds
	8. Identify lane-line pixels and fit a 2nd degree polynomial
	9. For line smoothing, take the average of the last 3 frames

This approach works well on the project video, with very small glitches in some noisy areas. Some cases in which I think the pipeline will fail are:

	1. The left and right lane lines do not reside in the ranges the algorithm is looking for them in: [200, 300] for the left line and [1000, 1100] for the right line; in this case, the pipeline will erroneously consider some other lines in that ranges as the left and right lane-lines.
	2. The road surface contains other lines, similar to the lane lines, that might confuse the algorithm (this is the case in the challenge video, where the lane lines are less visible than other lines on the road)
	3. The road has very tight curves; in this case, it is very likely that at some point, the algorithm will fail to find lane line-corrsponding pixels in the search window, and it will move the search window in a straight direction, instead of following the curve

Here are some ideas that might help solving these issues:

	1. Come up with a smarter way of defining the ranges for the left and right lane-lines, instead of the hardcoded values which are used now
	2. Come up with a smarter way of defining the source and destination points for the perpsective transform, instead of the hardcoded values which are used now; an idea could be computing the vanishing point
	3. When the pipeline can't find any lane pixels in a given search window, move the window in the direction of the curve (computed with respect of the previously found pixels) instead of moving it in a straight direction.
	4. Find better values for thresholds