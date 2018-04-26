# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/center.png "Center"
[image3]: ./images/left.png "Left"
[image4]: ./images/right.png "Right"
[image5]: ./images/original.png "Normal Image"
[image6]: ./images/flipped.png "Flipped Image"


#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 6 convolution layers and 4 fully connected layers(model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

The model includes average pooling layers to reduce dimension.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets(8:2) to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model includes dropout layers to prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
Also, I used the correction parameter to find the best corrected angles.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Also, I used the images from the center lane, left side, and right side, which could help recovering from the sides.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a simple regresion model to test the code, and I found that the regression.h5 model could only have the car drive in one direction, soon the car went out of the lane.

Secondly, I add the Lambda layer to normalize the data and have the steering angle be at the range 0~0.5. Yet after 7 epochs, the mse does not decrease as expected. 

Third, I used the NVIDIA Deep Learning model, mainly because this model is designed for self-driving car training. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropt out layers.

Also, the NVIDIA model is relatively complicated, and it takes very long time for the GPU on AWS to train, to  resize the image, I added average pooling layers to reduce the dimension.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) 
Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when off-centered.

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help preventing model from going towards to the left or right only. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]



After the collection process, I had 6428 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, and after 5 epochs, the loss does not decrease much. I used an adam optimizer so that manually training the learning rate wasn't necessary.
