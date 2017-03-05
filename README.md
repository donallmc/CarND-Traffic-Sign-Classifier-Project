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
[german1]: ./images/german1.jpg "german1"
[german2]: ./images/german2.jpg "german2"
[german3]: ./images/german3.jpg "german3"
[german4]: ./images/german4.jpg "german4"
[german5]: ./images/german5.jpg "german5"
[german6]: ./images/german6.jpg "german6"
[german7]: ./images/german7.jpg "german7"
[german8]: ./images/german8.jpg "german8"


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

While augmenting the images I also took the opportunity to correct the imbalance in the datasets. As shown in the charts above, some classes are significantly more common than others. To correct this, I implemented a (somewhat hacky) means of selectively generating more images for under-represented classes than for the commonly occurring ones. The final dataset included 450,000 examples. The final class frequency distributions looks like this:

![alt text][final_freqs]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

The model is based on the LeNet code supplied in the course with the addition of a third convolutional layer and some dropout layers to handle overfitting. I also did a lot of tinkering with the sizes of the convolutions and strides. I was interested in having the model examine smaller chunks of the image to potentially learn components better as the traffic sign dataset contains images that are distinguishable only by small pixel areas. I don't think that my final numbers are optimal but they are adequate. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 8x8     	| 2x2 stride, same padding, outputs 14x14x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 7x7x20 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 5x5x400 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x400 				|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 4x4x20 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 4x4x400 				|
| Fully connected		| output 120        									|
|	RELU				|												|
|	Dropout					|	50%											|
| Fully connected		| output 84        									|
|	RELU				|												|
|	Dropout					|	50%											|
| Fully connected		| output 43        									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I initially used an AdaGrad optimizer as I had read that it performed well on sparse data but as I began augmenting the dataset I noticed that Adam had better performance so I switched to that. In particuar, Adagrad often seemed to hit an accuracy level lesser than the Adam optimizer and then regress to a very low accuracy.

I experimented with the batch size and larger batches seemed to perform better than smaller ones, although the difference wasn't that great above a certain threshold. The submitted version of the chose a batch size of 882, chosen because it was a denominator of the validation set and it was useful for debugging reasons!

The number of epochs in the submitted notebook is 50. After about 30 epochs the rate of improvement decreases noticeably but there's still a significant amount of improvement to be had in the final 20 epochs. After 50, however, the improvements just seem to fluctuate around a maximum value.

as for hyperparameters, the learning rate was modified by Adam. I did some experimenting with the initial value but got no improvement so I went with a recommended value of 0.001. I also did some tweaking of the initial data parameters, mu and sigma, but my changes didn't result in any desirable effects.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventh cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 98.4%
* test set accuracy of 90.8%

Worth noting that I re-ran the test set accuracy a few times (after all model details were finalised!) as part of re-running the ipython notebook after clean-up. Due to the large size of my training set I ran into memory problems which I wasn't able to resolve (and which I believe were due to transient conditions on my development machine). As a result, the submitted version included 50% fewer examples than an earlier version which achieved a test set accuracy of 92.5%. This is a good indication that a larger training set would improve the accuracy of the model.


As stated previously, the architecture was initially based on LeNet with some modifications as described above. The iterative design process was heavily impacted by a sudden drop in accuracy which took 3 days to resolve; it turns out I had moved to a new machine and checked out the data from Udacity. The dataset had been substantially modified and I had received any notifications (I have since heard it was mentioned in slack, which I don't consider to be an adequate notification). This caused my accuracy to fluctuate from 80%-90%. I tore apart the model and rebuilt everything then started experimenting heavily with adding and removing both fully connected and convolutional layers. Nothing seemed to improve things and after several days I realised that the issue was the modified dataset. Having spent so much time and now concerned that I won't get all the projects submitted by the deadline, I ended up tweaking the model that I then had to get it to a reasonable state. While this was an iterative process, I don't think it was good development! If Udacity would like me to try again I would request an additional weekend of development time to do it right. I would also politely request that significant changes to the dataset should be communicated more thoroughly and that the course materials should be updated (I followed the LeNet video line by line with my code and that's how I discovered that something was fishy about the dataset because David got 96% accuracy and I got < 90% with the exact same code).
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][german1] ![alt text][german2] ![alt text][german3] 
![alt text][german4] ![alt text][german5]

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
