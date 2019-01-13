# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-cnn-architecture.png "Model Visualization"

[image2]: ./examples/center.jpg "center"
[image3]: ./examples/left.jpg "left"
[image4]: ./examples/right.jpg "right"
[image5]: ./examples/flipped.jpg "flipped"
[image6]: ./examples/loss.png "loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

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

#### 1. Solution Design Approach

My model is based on NVIDIA network which has a good performance on training self-drving cars. At first, I only use basic NVIDIA network which is shown below:

![alt text][image1]


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add two Dropout layers only after the fully connected layers. 

First, I converted the images to RGB after loading the images so that the model won't be confused to evaluate RGB images.

Then, I normalize the image by dividing each element by 255 and substracting 0.5 to shift the element mean down from 0.5 to 0. Then, I crop the image and output 65x320x3 images which only focus on the road rather than other unnecessary parts of the image. 



My final model consists of three convolution layer with 5x5 filter size and two convolution layer with 3x3 filter size, following ELU activation layers which tend to converge cost to zero faster and produce more accurate results. 

The model was trained and validated on different data sets by shuffling to ensure that the model was not overfitting. 


The final step was to run the simulator to see how well the car was driving around track one. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-89) consisted of a convolution neural network with the following layers and layer: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 BGR image   							| 
| Normalization     	| Normalize the image 	|
| Cropping2D					|	Crop the image, output 65x320x3 image										|
| Convolution 5x5     	| 2x2 stride, same padding, outputs 31x158x24				|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, same padding, outputs 14x77x36								|
| ELU					|												|
| Convolution 5x5      	| 2x2 stride, same padding, outputs 5x37x48 |
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 3x35x64								|
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 1x35x64								|
| ELU					|												|
| Flatten		| outputs 2112 
| Dropout					|	Probability 0.5											|
| Fully connected		| outputs 100 
| ELU					|												|
| Fully connected		| outputs 50 
| ELU					|												|
| Fully connected		| outputs 10 
| RELU					|												|
| Dropout					|	Probability 0.5											|
| Fully connected		| outputs 1 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I also used data augmentation on center image which flipped the image horizontally. 

Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to make a correct turn from being off-center. 

The following images are taken at the same time where the vehcle is facing a right turn. We can see that the left image gives a recovery map on the center of the road, thus after adding the left correction on left image, the model can use this data to correct the center position of the vehcle.

Center Image:

![alt text][image2]

Left Image:

![alt text][image3]

Right Image:

![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped (center, left and right) images and angles thinking that this would provide more training data for the model. For example, here is an image that has then been flipped:

Original Center Image:

![alt text][image2]

Flipped Image:

![alt text][image5]



I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I also used checkpoint to save the model after each epoch and save the best model with least validation loss. 

I used an adam optimizer with 0.001 learning rate since the default learning rate 0.01 is not good enough for training the model.


**NOTE** MSE loss history visualization

![alt text][image6]

By recording all the checkpoints and save the best model, I got a model with lowest validation loss of 0.01207. In the video, the vehicle could drive on the road in track 1. However, it needs improvements on some sharp corners to keep the vehicle in the center of the road.


Thanks for reviewing!
