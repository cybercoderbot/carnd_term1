**Behavioral Cloning** 

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[sim1]: ./images/sim_1.png "Result"
[center1]: ./images/center_1.jpg "Grayscaling"
[center2]: ./images/center_2.jpg "Recovery Image"
[center2f]: ./images/center_2_flip.jpg "Recovery Image"
[left1]: ./images/left_1.jpg "Recovery Image"
[left2]: ./images/left_2.jpg "Recovery Image"
[right1]: ./images/right_1.jpg "Normal Image"
[right2]: ./images/right_2.jpg "Flipped Image"


---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 for demonstrating the autonomous driving
* README.md summarizing the results


---
| ||
|:--------:|:------------:|
|[![alt text][sim1]](https://youtu.be/nPoJt520_MA)|
|[YouTube Demo](https://youtu.be/nPoJt520_MA)|



---
2. Submission includes functional code
Using the driving simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

The model is based on Nvidia paper (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for training a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands. This end-to-end approach is very relevant to this project since we also aim at training the network for steering with limited training data. 

In summary, it is a deep neural network for predict driving commands from front-view camera images.  The model applies 5 layers of convolutions for feature extraction and 5 layers of fully-connected network for regression, as follows:

- Image pre-processing
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 264, activation: ELU
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1, activation: Linear 


Some adjustments was made to the model: 1) A Lambda layer is used at the begining of the newtork to pre-process input images to improved performance. The pixels are centered around zero with values between [-0.5,0.5].
2) A dropout layer is introduced between convolution layers and fully-connected layers to avoid overfitting 

The activation function for convolution and fully-connected layers is ELU (Exponential Linear Units), which is similar to ReLU, the mean of the activation is more close to zero. This activation introduces nonlinearity, and prevents the vanishing gradient problem. After the convolution, dropout is introduced to reduce overfitting. The keep probability is 0.5.




2. Attempts to reduce overfitting in the model

In order to reduce overfitting, the model contains dropout layers with keep probability of 0.5. Also, the model was trained and validated on different data sets, and the model stop training when the validation errors increases. The model was tested by running it through the simulator and ensuring that the vehicle stays on the track.

3. Model parameter tuning

The model used an Adam optimizer, so the parameters are automatically updated.

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Model Architecture and Training Strategy

1. Solution Design Approach

The overall strategy for deriving a model architecture was to design an end-to-end system from front-view images to predict steering angles. My first step was to use a convolution neural network model similar to the nvidia model. I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The trainin and validation error decrease monotonically during first 3 epochs, which is a good sign. The validation error starts to fluctuate after epoch 3, so the network stops training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 

2. Final Model Architecture

The final architecture of the model is denoted as follows: 


---
| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|--------|-----------------|
|lambda_1 (Lambda)               |(None, 160, 320,3)|0       |lambda_input_1   |
|cropping2d_1 (Cropping2D)       |(None, 65, 320, 3)|0       |lambda_1         |
|convolution2d_1 (Convolution2D) |(None, 31, 158,24)|1824    |cropping2d_1     |
|convolution2d_2 (Convolution2D) |(None, 14, 77, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 5, 37, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 3, 35, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 1, 33, 64) |36928   |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 1, 33, 64) |0       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 2112)      |0       |dropout_1        |
|dense_1 (Dense)                 |(None, 264)       |557832  |flatten_1        |
|dense_2 (Dense)                 |(None, 100)       |26500   |dense_1          |
|dense_3 (Dense)                 |(None, 50)        |5050    |dense_2          |
|dense_4 (Dense)                 |(None, 10)        |510     |dense_3          |
|dense_5 (Dense)                 |(None, 1)         |11      |dense_4          |

|                                |     
|--------------------------------|
|Total params: 721,251 |
|Trainable params: 721,251 |
|Non-trainable params: 0 |

---

3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


|       | 
|:--------------------------------:|
| Centered Lane driving            | 
|![alt text][center2] |


In , three camera shots are taken for each time-stamp (one from center, one from left, and one from right):

| |   | 
|:--------:|:------------:|
|Center-camera image |Left-camera image |
|![alt text][center2] |![alt text][center2f]|
|Right-camera image |
|![alt text][right1]|


Since the track is a loop, and the car drives conter-clockwisely for the most of the time during training. Thus, the car has a tendency to drive to the left even on straight road. To help reducing the bias towards steering to the left, images and angles and angles are also flipped. For example, here is an image that has then been flipped:

| |   | 
|:--------:|:------------:|
|Original image |Flipped image |
|![alt text][center2] |![alt text][center2f]|


The steering angle for the flipped image is the negation of the original steering value.

Finally, training data is randomly shuffled and 20% of the data is used for validation, which helped determine if the model was over or under fitting. The ideal number of epochs was 3 as the validation error fluctuate after that.
