#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[test_freqs]: ./images/test_freqs.png "frequencies"
[training_freqs]: ./images/training_freqs.png "frequencies"
[validation_freqs]: ./images/validation_freqs.png "frequencies"
[final_freqs]: ./images/final_freqs.png "frequencies"
[no_passing]: ./images/no_passing.png "no passing"
[stop]: ./images/stop.png "stop"
[speed_80]: ./images/speed_80.png "speed_80"


[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/donallmc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34,799
* Number of testing examples = 12,630
* Number of validation examples = 4,410
* Image data shape = (32, 32, 3)
* Number of unique classes/labels = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. Here are image frequencies for test, training, and validation sets:


![alt text][training_freqs]
![alt text][validation_freqs]
![alt text][test_freqs]

Here are some examples of randomly chosen images for 3 classes:

![alt text][speed_80]
**5: Speed limit (80km/h)**

![alt text][no_passing]
**9: No passing**

![alt text][stop]
**14: Stop**

Samples for the full 43 classes can be seen in the python notebook.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale to simplify things by ignoring colour and having smaller feature sets to process. I also applied a histogram normalisation to handle the pronounced differences in brightness in the sample dataset. However, after some testing it became apparent that this kind of normalisation was not performing better than the colour dataset so I dropped it.

The only normalisation that is applied to the colour images is a simple normalisation to constrain all values to ```-1 <= x <= 1``` and I achieved satisfactory results with this approach. 

There is also some data augmentation code in this cell, which I will describe in the next section.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook along with a snarky comment about how the dataset changed and caused me to lose a lot of time! :)

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the appropriate SKLearn function.

The fifth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the provided data set is relatively small and this kind of deep learning benefits from much larger sets. In addition, the images provided vary significantly in terms of lighting conditions, image position, obscured components, etc. An augmented dataset would increase the number of examples seen of each permutation of these parameters, leading to a more robust classification. 

To add more data to the the data set, I applied some random transformations to the image including adjusting the brightness, applying a random rotation, cropping, and translating the image. The original version of my code applied some of these transformations (as well as some normalisation) using the TensorFlow library. It was my intention to perform these transformations on-the-fly. However, even running on a GPU instance the time it took to process each image was unreasonably long and it had a detrimental effect on my iteration time. Instead I resolved to pre-process each image and generate additional examples before feeding data into the model. This has the obvious advantage that everything is done only once but it adds a pre-processing dependency that could theoretically lead to bugs. I did some Googling around pre-processing images in python for this dataset and actually came across another student's solution. I used the functions he defined on the basis that they looked well put-together and I didn't think I could improve on them. I could have easily re-implemented them myself, but I'd prefer to leave them as is and credit the source.

While augmenting the images I also took the opportunity to correct the imbalance in the datasets. As shown in the charts above, some classes are significantly more common than others. To correct this, I implemented a (somewhat hacky) means of selectively generating more images for under-represented classes than for the commonly occurring ones. The final class frequency distributions looks like this:

![alt text][final_freqs]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
