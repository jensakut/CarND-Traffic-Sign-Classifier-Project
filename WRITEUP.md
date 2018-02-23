# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./42.jpg "Visualization"
[image2]: ./original2.png "rotate"
[image3]: ./Histogram_dataset_true.png "Relative histogram of the dataset"
[image4]: ./random-order.png "Histogram of the dataset"
[image5]: ./Histogram_dataset_false.png "Absolute histogram of the dataset"
[image6]: ./augmented.png "Augmented signs"
[image7]: ./webimages_output/webimages.jpg "webimages"
[image8]: ./softmax.png "softmax"
[image9]: ./featuremap.png "featuremap"
[image10]: ./featuremap2.png "featuremap"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jensakut/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (67 %)
* The size of validation set is 4410 (9 %)
* The size of test set is 12630 (24 %)
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43



#### 2. Include an exploratory visualization of the dataset.

Following an exploratory visualization of the data set is shown. The bar chart shows the histogram of the sign types and their relative number in the data set. The sign types are one-hot encoded in the format ClassId, SignName.
![relative Histogram of the traffic signs][image5]

The blue bar represents the training data set, the orange the validation set and the green stands for the test images. The classes are not equally distributed. Looking at the training set, some signs have only 250 samples while others are represented with 2000 images. 

The following diagram shows the relative distribution of the signs, which is equal among the sets. 
![relative Histogram of the traffic signs][image3]

In the second row, the dataset in the position 101 to 105 is printed. Looking at the background, the same sign serves as a motive out of different perspectives. Other samples plotted out of the dataset show similar series. 

The following picture shows 5 random drawn pictures out of the training set in the upper row. In the lower row, 5 consecutive pictures (position 101-105) are displayed. The lower row reveals an interesting property of the data set. The motive is the same, but the picture is taken from a different perspective. As a result, the sign is slightly different in every example. 
![Examples of signs][image3]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


Traffic signs convey some informations through color, but grayscale potentially simplifies the classification problem. Therefore, both was tested. 
To reduce noise, a Gaussian Filter with a pixel size 5x5 was tested as well.  

![alt text][image2]
As a first step, I transferred the data to an interval -0.5 to 0.5. The centered range improves the learning significantly. Non-centered data is limited to around 70-80 % validation accuracy. 

Additional augmented data was tested. The histogram revealed an unequal sample size. Traffic signs which were underrepresented were randomly modified and the additional copies were appended. Both training and validation set was augmented in this process, whereas the test-set was left untampered. 
The augmentation consists of random rotation in different intervalls between +-90 degrees and random shifts in the interval +-6 pixels. 

Here is an example of an original image and an augmented image:

![Augmented pictures][image6]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description | output       					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 normalized RGB image  		| |
| Convolution 3x3     	| 1x1 stride, valid padding | 28x28x16 	|
| RELU					|										||
| Max pooling	      	| 2x2 stride, valid padding  | 28x28x16 		|
| Convolution 3x3	    | 1x1 stride,valid padding,  |10x10x64      	|
| Max pooling   		| 1x1 stride, valid padding | 5x5x64    
| Flatten				| | 1600 
| Fully connected		|Classifier | 240							
| Relu Activation	    | | |										
| Dropout               | Training: Keep probability = 0.5 ||
| Fully connected       | Classifier |84
| Relu Activation       |||
| Dropout               | Training: Keep probability = 0.5 ||
| Fully connected       | Sign classifier | 43 
| Softmax | Cross Entropy |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the same configuration as the LeNet lab. It consists a softmax cross entropy minimization and the Adam optimizer.
After every epoch, an accuracy with the validation set gets calculated to compare it to a training accuracy sample of 50 random signs. Thus, the fitness on both training sets can be compared. 
After finishing all Epochs, the test set serves as a real world validation. 

The batch size is 128 in order to comfortably fill the available RAM on the laptop. 
With the final model, 15 Epochs are enough to reach the maximum fitness on the test set. Afterwards, the accuracy varies little, indicating possible overfitting of the model. 
The learning rate of 0.001 provides the best results. Optionally, the learning rate can be reduced after each step by multiplication with a factor < 1. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.973
* test set accuracy of 0.958

The LeNet architecture was chosen. In the LeNet lab, the architecture provided good results on classifying handwritten numbers. Thus, a RGB color classifier seemed a natural follow-up. 

The original LeNet architecture was modified to detect 43 classes instead of 10, and to accomodate RGB or grayscale, depending on experiments. 
The initial experiments were conducted using preprocessing functions. But neither grayscale to simplify, gaussian filtering to reduce noise, nor contrast enhancing improved the problem significantly. Thus, the preprocessing only consists of normalization, which proved to improve accuracy. 
Playing with initialization parameters epochs, batch size, sigma of the weights, and dynamic learning rate didn't improve the networks performance either. They were abandonded, and the values of the LeNet architecture were used in the next stage. 
Given the enhanced complexity, the LeNet architecture needed additional complexity. The color provides additional information, but the first convolution was made 3 times deeper. The following layers were deepened as well, thus the network is more likely to learn the complexity of colors. Consequently, the flattened layer now consists of 1600 elements (400 in LeNet), the first fully connected of 240 instead of 120. The last layer was left at 84 output neurons. 
This enhanced learning to be roughly 80 percent accurate on the validation set. To follow the courses recommendations, dropout was implemented on both fully connected classification layers. The keep-probability of the network improved accuracy a lot. This leads to the final performance of the network of 96 % on the test set and 99 % on the training set in 15 Epochs. 
This model was both exposed to the preprocessing methods grayscale and gaussian blur as well as decreasing learning rate and an augmented dataset. Preprocessing doesn't impact the result much. Augmenting the dataset seriosly deteriorated the results, indicating a problem with the augmenting algorithm. 
Slightly decreasing the learning rates by factorizing 0.95 helped the network to reach a little bit better results of 0.957 %, but faster decreases aren't useful. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] 

The first two images are often represented in the training data and relatively clear. Image 3 is modified with a sticker, and the roadworks sign is of deteriorated quality. The slippery road is clear but underrepresented in the dataset. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 kph      		| 50 kph  									| 
| 70 kph     			| 70 kph  										|
| No entry				| Keep right											|
| roadworks	      		| Traffic lights					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The pictures show, that the model works great in good weather conditions. But rain, snow, night, or stickers on traffic signs pose a challenge to the system. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)




| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 50 kph 									| 
| 1.     				| 70 kph										|
| .14					| keep right											|
| .91	      			| 			Traffic lights	 				|
| .99				    | Slippery Road      							|

![Augmented pictures][image8]


First and second image are classified with a very high probability. The third image cannot be classified, and doesn't fit into the existing classes. 
The fourth image is classified wrong with a high probability. The fourth image isn't clear but very well classified. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The featuremap of the first convolution shows rough shapes. It shows a round shape of a parking forbidden (or similar) shape. Depending on the input picture, the shape of the picture is highlighted, whereas the background isn't represented. 
![featuremap][image9]

The featuremap of the second layer shows either a bitwise activation or a very abstract representation of ideas. 
![featuremap2][image10]
