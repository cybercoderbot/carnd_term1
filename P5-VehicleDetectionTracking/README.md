**Vehicle Detection And Tracking**

In this project, a software pipeline is defined to detect vehicles in a video. The whole system can be broken into following steps:

* Compute image featues using spatial, color hist, and hog vectors
* Train the classifier for recognizing car/noncar patches
* Apply a multi-scale sliding window to detect cars at different locations and distances
* Using a heatmap to represent the probability of a patch/pixel being a vehicle
* Remove false positives by thresholding the car probability heatmap
* Display the detections using bounding boxes on original video frames


[//]: # (Image References)
[video1]: ./project_video.mp4
[video]: ./results/video.png


---

**Result:** 

Here's a [link to my video result](./project_result.mp4)

---
| ||
|:--------:|:------------:|
|[![alt text][video]](https://youtu.be/Q0HhtNoGMGA)|
|[YouTube Demo](https://youtu.be/Q0HhtNoGMGA)|

---
**Content of this repo**

- `vehicle_detection_tracking.ipynb` - Jupyter notebook with code for the project
- `results` - a directory with test images.
- `project_video.mp4` - the original raw video
- `project_result.mp4` - the result video
- `README` file

---
 **Feature Extraction**

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes.

`vehicle` class training images. These car images are realistic images sampled at various distance, viewing angle and color:

<img src=./results/car/far1.png width="100" />
<img src=./results/car/far2.png width="100" />
<img src=./results/car/left1.png width="100" />
<img src=./results/car/left2.png width="100" />
<img src=./results/car/right1.png width="100" />
<img src=./results/car/right2.png width="100" />

`non-vehicle` class training images, with nagetive samples from the sky, the tree trunks, the road curb, etc. The negative samples varies in color, texture and spatial shape as well.

<img src=./results/noncar/non1.png width="100" />
<img src=./results/noncar/non2.png width="100" />
<img src=./results/noncar/non3.png width="100" />
<img src=./results/noncar/non4.png width="100" />
<img src=./results/noncar/non5.png width="100" />
<img src=./results/noncar/non6.png width="100" />

The training dataset has 17584 car images and 17936 non-car images. The resolution of each sample is `64x64`, with 3 color channels.

<img src=./results/hog_vis.jpg width="500" />

The HOG features is a representation of the gradient distribution in a image. After the gradients being computed, it's summed within small, local patches (called cell). This representation is robust to variations in shape. The `scikit-image hog()` function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include `orientations`, `pixels_per_cell` and `cells_per_block`. The following explanation is taken fro the scikit-image.org.

The number of `orientations` is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The `pixels_per_cell` parameter specifies the cell size over which each gradient histogram is computed. This paramater is passed as a 2-tuple so you could have different cell sizes in `x` and `y`, but cells are commonly chosen to be square.

The `cells_per_block` parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized. Block normalization is not necessarily required, but generally leads to a more robust feature set.

There is another optional power law or "gamma" normalization scheme set by the flag `transform_sqrt`. This type of normalization may help reduce the effects of shadows or other illumination variation, but will cause an error if your image contains negative values (because it's taking the square root of image values).

To achieve the best result, we have to tune the parameter well to extract relevant features. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Following is what the `skimage.hog()` output looks like.

<img src=./results/car1.png width="250" />
<img src=./results/hog1.png width="250" />


Finally, the parameters are tuned as:
```
color_space = 'LUV'     # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8              # number of HOG orientations
pix_per_cell = 8        # HOG pixels per cell
cell_per_block = 2      # HOG cells per block
hog_channel = 0         # Can be 0, 1, 2, or "ALL"
spatial_size = (16,16)  # Spatial binning dimensions
n_bins = 32             # Number of histogram bins
```

 
The HOG features mainly represent the shape of the image. In addition to that, binned color (color and shape features) and color histograms (color feature are used. The features from 3 different feature extraction methods are normalized using `StandardScaler()`, and then concatenated to make a final representation of the image. Normalizing ensures that a classifier's behavior isn't dominated by one type of the features. The final feature vector length is 2432.

---
**Classifier Trainig**

I trained a linear SVM using `sklearn.svm.LinearSVC()`. This is similar to SVC with parameter `kernel=’linear’`, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. If features are choosen appropritely, the classification can use a simple linear kernel. This  proves true, since the linear SVM classifier is efficient and performs well in the task of recognizing car patches from the background. The data is splitted into 80% training and 20% testing.


**Sliding Window Search**

To detect cars at various location, sliding windows are used to scan the image. The image patch within each window is extracted for determining whether it contains a car. Thus, cars will be detected with invariance to spatial location. Also, cars appear at different distances, so they should be detected at multiple scales as well. Following image shows how windows used at different scales and different locations to find a car.

<img src=./results/sliding_win.png width="500" />

Since we detect cars at different locations and different scales, the same car can be detected by windows at multiple locations and scales, as the following image shows. 

<img src=./results/search_win.png width="500" />

Thus, a voting machnism should be reduces to improve detection and remove false positives. For example, whenever a pixel is detected as car, it will get a vote in the probability of being a car. If we have an area being detected as cars by multiple overlapping windows, it will probably be a real car image. On the opposite, if an area is being detected as cars once, it will probably be a false positive. Following is how the voting result being represented as a heat map. The larger the pixel value, the high the probability of it being a car.

<img src=./results/heatmap.png width="510" />

---

**Video Implementation**



Since the camara is mounted at a fixed position of the car, we can only search the bottom half of the image for the cars. This reduces the computation by 50% at no cost of performance. Thus, for each of the video frame, sliding windows are applied at the bottom half to locate cars. For each of the image patch in the sliding window. The HOG, color hist, and spatial features are extracted. Then these features are normalized and concatenated for recognition. The trained the SVC classifier labels each input patch is a car or not. Then a heatmap is computed by aggregate results from multiply detections. Finally, the heatmap is thresholded to remove false positives.

Here's a [link to my video result](./project_result.mp4)

| ||
|:--------:|:------------:|
|[![alt text][video]](https://youtu.be/Q0HhtNoGMGA)|
|[YouTube Demo](https://youtu.be/Q0HhtNoGMGA)|



---

**Discussion**


In this project, a robust and efficient car detection model is used to detect cars in real images on the road. The cars can be detected at different locations, different scales, and different perspectives. The HOG features, color hist and hist bin is a good representation of color and shape. The SVC classifier proved efficient and performs well. However, it can be very challenging for the current model to detect cars in following circumstances: 

1. When the lighting condition of the image changes. The underlying assumption of the models is that the frame images are properly illuminated. When the weather is gloomy or it's dark, the cars can be missed since cars are less obvious. A headlight/taillight detection component will be very helpful in those conditions.

2. When the cars are partially occluded. As the result shows, when the white car is partilly occluded, it will be missed in detection or be considered as a part of it's neighboring car. This can be solved if we used a tracking algorithm on top of the detection. When the cars appears in our field of view, it's identified and stored. When it's partially occuluded, it will be compared to the cars that's in the current database.

3. Applying a searching window in each frame is computationally expensive. When we established the car database, we can only track those cars and look for new cars that emerge into the frame. Since cars will most likely emerge in bottom-left and bottom-right corner, we can mainly search in those areas for the new cars while keep track of current cars already detected. This will reduce the computation task and make the model more efficient for real-time vehicle detection and tracking.

