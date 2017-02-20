#**Finding Lane Lines on the Road** 

##Writeup - Yuheng Wang


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on my work in a written report


[//]: # (Image References)

[image1]: ./test_result/solidWhiteRight_edges "Edges"


[//]: # (Image References)

[image2]: ./test_result/solidWhiteRight_line_edges "ColorLines"


---

### Reflection

###1. My pipeline. 

My pipeline is as follows:

First, I converted the images to grayscale. To detect lane lines in the image, I conbined Hough Transform line detection with region of interest (ROI) selection. To do that, I first apply a 5 by 5 Gaussian filter to blur the image for edge detection, followed by canny edge detector with low threshold being 50 and high threshold being 160. This results in a black and white edge image of the road.

![alt text][image1]

To select only lane lines from the image, I apply a triangle shaped region of intest selector in the central bottom part of the image. Some of the lanes are with solid lines while some lanes are with dashed lines. To get a unified representation of lane lines, I used a line detector using hough transform. This detector will detect lines in the edge image and join broken lane line representation. In hough transform, I tune the parameters and I find that a threshold of 20, mininum line length of 150 and maximum line gap of 200, give me a good balance of join all line parts without picking up much noise.

Finally, I plot the line detection result on the original image. The result looks like:
 
![alt text][image2]



###2. Potential shortcomings and possible improvements

In this model, I assume that Gaussian blurring followed by Canny edge detection will detect the edges effectively. When the road is shadowed by the tree trunks, it's prone to pick up the tree shadows (as in the challenge video). On the opposite, when the lanes are shadowed by the buildings or another car, it's hard to pick up the lane line correctly.

Another assumption in this model is that lane lines tend to appear in the bottom part of the road image. This holds true for most high way images, however, when driving on local, it's very risky to assum the location of the lanes, especially at intersections (when the lanes are curves instead of lines). A more robust lane detection model that makes no assumption of the locaton and shape of the lane lines would be essential for robustness in lane detection.

Thirdly, all parameters (Gaussian filter size, thresholds for Canny edge detection, thresholds and line gap length in Hough transform) are tuned by manual experiments. A generalized adaptive filtering and thresholding will be very helpful to make the model scale on a more variable dataset.


