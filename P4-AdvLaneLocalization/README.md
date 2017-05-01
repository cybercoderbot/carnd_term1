
**Advanced Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[image2]: ./test_images/test1.jpg "Road Transformed"
[corner]: ./examples/corners.png "Binary Example"
[undist]: ./examples/undistorted.png "Warp Example"

[warped]: ./examples/warped.png "Binary Example"
[color]: ./examples/color_threshold.png "Warp Example"
[sobel]: ./examples/sobel_threshold.png "Binary Example"
[combine]: ./examples/combine_threshold.png "Warp Example"

[fitted]: ./examples/fitted_line.png "Fit Visual"
[polyfit]: ./examples/polyfit.png "Output"
[output]: ./examples/output.png "Output"
[youtube]: ./examples/youtube.png "Output"
[video1]: ./project_video.mp4 "Video"


**Camera Calibration**

1. Compute the camera matrix and distortion coefficients.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./adv_lane_detection.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][corner]

**Pipeline (single images)**

1. Camera calibration and distortion-correction.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images.  After obtaining `obj_points` and `img_points` from `find_corners()` function, `cv2.calibrateCamera` and `cv2.undistort` are used to compute undistorted images:

`ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)`


To show how the undistorted image is different from the original image, I displayed original and undistorted checkerboard image side by side. As is demostrated, an unwarped checkerboard image removes camera distortion. The bottom image shows how real front-view camera images are restored. 
![alt text][undist]

2. Perspective transform.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][warped]


3. Color transforms and Sobel filtering to create a thresholded binary image.  
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

In real road images, lane lines are mostly yellow and white. So HLS color space is used for detecting specfic lane line colors. First, each warped bird-view lane line images are converted in HLS color space:

` L = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:,:,1]`

` S = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:,:,2]`

Yellow color is reliably detected in the S (saturation) channel in HLS color space, we can get yellow lane lines by thresholding pixel values in S channel:

`    s_thresh = (160, 250)`

`    s_binary = np.zeros_like(S)`

`    s_binary[(s_thresh[0] < S) & (S <= s_thresh[1])] = 1`

White color is reliably detected in the L (lightness) channel in HLS color space, we can get white lane lines by thresholding pixel values in L channel:

    l_thresh = (220, 255)
    l_binary = np.zeros_like(L)
    l_binary[(l_thresh[0] < L) & (L <= l_thresh[1])] = 1

Following is result of images thresholded by combining S channel and L channel thresholding:
![alt text][color]

For the gradient method, I detect edges along horizontal and vertical directions, and then threshold the magnitude of the gradient into binary image, as follows:

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag / scale_factor).astype(np.uint8) 

Following is result of images by thresholding outputs from sobel filters:
![alt text][sobel]



4. Identified lane-line pixels and fit their positions with a polynomial.

To detect lane lines in binary mask, a sliding window is applied. Then I loop over windows, finding the lane center within the margin. The result looks like:

![alt text][fitted]

After all line lines are located using sliding window, lane lines are fitted with a 2nd order polynomial:

    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
    fitx_left = poly_left[0] * ploty ** 2 + poly_left[1] * ploty + poly_left[2]
    fitx_right = poly_right[0] * ploty ** 2 + poly_right[1] * ploty + poly_right[2]
    
    fit_cr_left = np.polyfit(ploty * yscale, fitx_left * xscale, 2)
    fit_cr_right = np.polyfit(ploty * yscale, fitx_right * xscale, 2)

Thus, both curved and straight lane lines are represented as follows:

![alt text][polyfit]


5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

After getting `fit_cr_left` and `fit_cr_left` from the polynomial fitting, curvature of the lane lines can be computed as:

    coeff = np.max(ploty) * yscale
    rad_left = ((1 + (2 * fit_cr_left[0] * coeff + fit_cr_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_left[0])
    rad_right = ((1 + (2 * fit_cr_right[0] * coeff + fit_cr_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_right[0])

Offset of the vehicle to the center can be computed as: 

    l_px = l_poly[0] * h ** 2 + l_poly[1] * h + l_poly[2]
    r_px = r_poly[0] * h ** 2 + r_poly[1] * h + r_poly[2]
    
    # Find the number of pixels per real metre
    scale = lane_width / np.abs(l_px - r_px)
    
    # Find the midpoint
    middle = (l_px + r_px) / 2
    
    # Find the offset from the centre of the frame, and then multiply by scale
    offset = (w/2 - middle) * scale


6. Plot filled lane back down onto the road.

After locating left and right lane lines correctly on the image, we can fill the lane lines using `cv2.fillPoly()` function and plot it back to the original image. Lane curvature and offset are also display on the image:

![alt text][output]

---

**Pipeline (video)**

1. Link to my final video output.

Here's a [link to my video result](./project_video.mp4)

---
| ||
|:--------:|:------------:|
|[![alt text][youtube]](https://youtu.be/PKwybqKYoZQ)|
|[YouTube Demo](https://youtu.be/PKwybqKYoZQ)|

---

**Discussion**


The current model works good when the illumination is stable. Since we used color thresholding and manually chose the threshold value, the model may fail when the lighting condition changes. Also, if the road changes texture or has crack/ unexpected object, the detector might be fooled. More robust lane line detectors, like semantic segmentation, or curve detector that's independent of colors, might be helpful to incrase accuracy. Also, temporal information could also be consider to make the model more robust (locations of lane lines in the previous frame has large probability of being lane lines in the next frames). 



