# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/distribution1.jpg "Distribution 1"
[image2]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/distribution2.jpg "Distribution 2"
[image3]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/GrayScaling.jpg "Grayscaling"
[image4]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001RGB.jpg "Test 1"
[image5]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/100E0005RGB.jpg "Test 2"
[image6]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/100E0001RGB.jpg "Test 3"
[image7]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/100E001RGB.jpg "Test 4"
[image8]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001RGBcv1_40cv2_60.jpg "Test 5"
[image9]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001RGBcv1_32cv2_64fc1_1000fc2_500fc3_200.jpg "Test 6"
[image10]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001RGBcv1_32cv2_64fc1_800fc2_200.jpg "Test 7"
[image11]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001RGBcv1_40cv2_64fc1_500.jpg "Test 8"
[image12]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E001Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 9"
[image13]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E0001Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 10"
[image14]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 11"
[image15]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/50E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 12"
[image16]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/100E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 13"
[image17]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/200E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 14"
[image18]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/random_images.jpg "random image"
[image19]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/random_images_result.jpg "random image result"
[image20]: https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup/feature_map1.jpg "feature map Convo1"

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### Here is a link to my [project code](https://github.com/Van-Vu/car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier_EC2.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set:

I used the pandas library to calculate summary statistics of the traffic
signs data set:

- Number of training examples = **34799**
- Number of validation examples = **4410**
- Number of testing examples = **12630**
- Image data shape = **(32, 32, 3)**
- Number of classes = **43**

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data is unbalanced

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Pre-preprocessed the image data
- As a first step, I decided to convert the images to grayscale because it gives better accuracy from my experiment below, it's also inline with Yan Lecun [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf):"The ConvNet was trained with full supervision on the color images of the GTSRB dataset and reached 98.97% accuracy on the phase 1 test set. After the end of phase 1, additional experiments with grayscale images established a new record accuracy of 99.17%"
It also helps to reduce the training time

- Here is an example of a traffic sign image before and after grayscaling then normalized the image data

![alt text][image3]

- I normalized the image data because it ensures that each pixel has a similar data distribution. This makes convergence faster while training the network
- The formula to normalize is *(image_data - image_data.mean())/image_data.std()* as recommended from [here](http://cs231n.github.io/neural-networks-2/)

**Notes:** as you can see in the code, I try different type of image conversion: Gray, RGB, HSV, HLS ... a quick conclusion is that Gray & RGB provide the best result

#### 2. Table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x40	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x40 				|
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 10x10x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Flatten		      	| output: 1600					   				|
| Fully connected		| output: 1000 									|
| Fully connected		| output: 500  									|
| Fully connected		| output: 200  									|
| Softmax				| logits: 43   									|


#### 3. Iteractive process to train my model:
I used Lenet architecture with following hyperparameters:
-AdamOptimizer
-BATCH_SIZE = 128

##### Iterative training 1
- EPOCHS = 50
- Image type: RGB
- Learningrate = 0.001
- Convo1: 40
- Convo2: 20
- FC1=500 → 300
- FC2=300 → 200
- FC3=200 → 100
![alt text][image4]
- **Test Accuracy** = 0.951
- **Comment:** test accuracy looks good, how about more epochs

##### Iterative training 2
- EPOCHS = 100
- Image type: RGB
- Learningrate = 0.001
- Convo1: 40
- Convo2: 20
- FC1=500 → 300
- FC2=300 → 200
- FC3=200 → 100
![alt text][image7]
- **Test Accuracy** = 0.920
- **Comment:** more epochs make it worse at reach plateau around 40th epoch

##### Iterative training 3
- EPOCHS = 100
- Image type: RGB
- Learningrate = 0.005
- Convo1: 40
- Convo2: 20
- FC1=500 → 300
- FC2=300 → 200
- FC3=200 → 100
![alt text][image5]
- **Test Accuracy** = 0.947
- **Comment:** reduce learningrate makes it better but still not reach 95.1, plateau aroung 35th

##### Iterative training 4
- EPOCHS = 100
- Image type: RGB
- Learningrate = 0.0001
- Convo1: 40
- Convo2: 20
- FC1=500 → 300
- FC2=300 → 200
- FC3=200 → 100
![alt text][image6]
- **Test Accuracy** = 0.920
- **Comment:** reduce learningrate more makes it worse. Increase epoch doesn't seem to work

##### Iterative training 5
- EPOCHS = 50
- Image type: RGB
- Learningrate = 0.001
- convo1=40
- convo2= 60
- FC1=1500 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image8]
- **Test Accuracy** = 0.954
- **Comment:** tweak the 2 convolutional layers bump the test accuracy to 95.4

##### Iterative training 6
- EPOCHS = 50
- Image type: RGB
- Learningrate = 0.001
- convo1=32
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image9]
- **Test Accuracy** = 0.950
- **Comment:** more tweak the 2 convolutional layers makes it worse then before

##### Iterative training 7
- EPOCHS = 50
- Image type: RGB
- convo1=32
- convo2= 64
- Learningrate = 0.001
- FC1=1600 → 800
- FC2=800 → 200
![alt text][image10]
- **Test Accuracy** = 0.950
- **Comment:** remove one fully-connected layer doesn't improve

##### Iterative training 8
- EPOCHS = 50
- Learningrate = 0.001
- Image type: RGB
- convo1=40
- convo2= 64
- FC1=1600 → 500
![alt text][image11]
- **Test Accuracy** = 0.951
- **Comment:** remove one more fully-connected layer doesn't improve

##### Iterative training 9 :boom::boom::boom::boom::boom:
- EPOCHS = 50
- Learningrate = 0.001
- **Image type: Gray**
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image12]
- **Test Accuracy** = 0.951
- **Comment:** use **Gray** image with inherited layers from RBG, test accuracy is 95.1 and **validation accuracy** is the highest **97.6**
:boom::boom::boom::boom::boom: **I choose this model**

##### Iterative training 10
- EPOCHS = 50
- Learningrate = 0.0001
- Image type: Gray
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image13]
- **Test Accuracy** = 0.932
- **Comment:** reduce learningrate, get worse

##### Iterative training 11
- EPOCHS = 50
- Learningrate = 0.0005
- Image type: Gray
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image14]
- **Test Accuracy** = 0.951
- **Comment:** increase learningrate a little bit, test accuracy is 95.1 on par with 0.001 but validation accuracy is worse. However the validation accuracy seems to plateau around 35th set. **Lets introduce dropout**

##### Iterative training 12
- EPOCHS = 50
- Learningrate = 0.0005
- Image type: Gray
- DROPOUT 0.5
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image15]
- **Test Accuracy** = 0.941
- **Comment:** introduce dropout, get worse and accuracy is unstable

##### Iterative training 13
- EPOCHS = 100
- Learningrate = 0.0005
- Image type: Gray
- DROPOUT 0.5
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image16]
- **Test Accuracy** = 0.932
- **Comment:** introduce dropout and increase epoch, even worse. Accuracy still is unstable

##### Iterative training 14
- EPOCHS = 200
- Learningrate = 0.0005
- Image type: Gray
- DROPOUT 0.5
- convo1=40
- convo2= 64
- FC1=1600 → 1000
- FC2=1000 → 500
- FC3=500 → 200
![alt text][image17]
- **Test Accuracy** = 0.949
- **Comment:** introduce dropout and increase epoch more, not bad but not as good as no dropout

#### Conclusion:
- Introduce more epochs doesn't mean the network gets better
- Dropout somehow doesn't seem to work
- This architect clearly reaches its peak around 95%, in order to increase the accuracy, better image preprocessing (augmenting, generate fake image ...) could be useful

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were in Iterative training 9:
* Validation set Accuracy = 0.976
* Test Accuracy = 0.951

### Test a Model on New Images

#### 1. Five random German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image18]

The fourth image might be difficult to classify because of the light and blurness of the sign

#### 2. The model's predictions on these new traffic signs

![alt text][image19]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing      		| Children crossing   									| 
| Ahead only     			| Ahead only 										|
| Turn right ahead					| Turn right ahead											|
| Dangerous curve to the right	      		| No passing for vehicle over	3.5 metric tons				 				|
| Keep right			| Keep right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This lower than the accuracy on the test set of 0.951

#### 3. 5 softmax probabilities for each prediction
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a Children crossing sign (probability of 1)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Children crossing   									| 
| 1.63602217e-22     				| Right-of-way at the next intersection 										|
| 3.37982091e-23					| End of no passing											|
| 3.62312691e-24	      			| Beware of ice/snow					 				|
| 2.72396806e-24				    | Road narrows on the right      							|


For the second image, the model is really sure that this is a Ahead only sign (probability of 1) and there's no other classification

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Ahead only   									| 
| 0     				| all other classes 										|

For the third image, the model is quite sure that this is a Children crossing sign (probability of 1) 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Turn right ahead   									| 
| 3.75329974e-38     				| Speed limit (100km/h) 										|
| 1.86314262e-38					| Keep right											|
| 0	      			| all other classes					 				|

For the fourth image, the model is quite sure that this is a 'No passing for vehicles over 3.5 metric tons' sign (probability of 0.979). Other probabilities are not too low. However all of 5 highest prob are incorrect

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.979         			| No passing for vehicles over 3.5 metric tons   									| 
| 1.08527457e-02     				| Right-of-way at the next intersection 										|
| 6.27199840e-03					| Speed limit (80km/h)											|
| 2.46640365e-03	      			| Turn right ahead					 				|
| 3.22315173e-04				    | Wild animals crossing      							|

For the fifth image, the model is really sure that this is a Keep right sign (probability of 1) and there's no other classification 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Keep right   									| 
| 0     				| all other classes 										|


### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

- I choose to visualize the convolution layer of the fourth image that was incorrectly identify above
- The model can clearly identify the boundary of the sign
- One feature is blackout
- The model has difficulty in identifying the sign symbol at the center, that's why the prediction varies from speed class to vehicle to turning sign

![alt text][image20]
