# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/distribution1.jpg "Distribution 1"
[image2]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/distribution2.jpg "Distribution 2"
[image3]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/GrayScaling_Augmented.jpg "Grayscaling"
[image4]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001RGB.jpg "Test 1"
[image5]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/100E0005RGB.jpg "Test 2"
[image6]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/100E0001RGB.jpg "Test 3"
[image7]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/100E001RGB.jpg "Test 4"
[image8]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001RGBcv1_40cv2_60.jpg "Test 5"
[image9]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001RGBcv1_32cv2_64fc1_1000fc2_500fc3_200.jpg "Test 6"
[image10]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001RGBcv1_32cv2_64fc1_800fc2_200.jpg "Test 7"
[image11]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001RGBcv1_40cv2_64fc1_500.jpg "Test 8"
[image12]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E001Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 9"
[image13]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E0001Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 10"
[image14]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200.jpg "Test 11"
[image15]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/50E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 12"
[image16]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/100E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 13"
[image17]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/200E0005Graycv1_40cv2_64fc1_1000fc2_500fc3_200_dropout.jpg "Test 14"
[image18]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/random_images.jpg "random image"
[image19]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/random_images_result.jpg "random image result"
[image20]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/feature_map.jpg "feature map Convo1"
[image21]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/New%20folder/50E001Graycv1_40cv2_20fc1_200_dropout05.jpg "feature map Convo1"
[image22]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/New%20folder/50E001Graycv1_20cv2_40fc1_300_dropout05.jpg "feature map Convo1"
[image23]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/New%20folder/30E001Graycv1_20cv2_40fc1_300_dropout05.jpg "feature map Convo1"
[image24]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/New%20folder/30E001Graycv1_20cv2_40fc1_300_dropout05_augmented.jpg "feature map Convo1"
[image25]: https://github.com/Van-Vu/traffic-classifier-project/blob/master/writeup/New%20folder/50E001Graycv1_20cv2_40fc1_300_dropout05_augmented_batchsize100.jpg "feature map Convo1"

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

![alt text][image2]{:width="500px"}

### Design and Test a Model Architecture

#### 1. Pre-preprocessed the image data
- Since the dataset is unbalance, I want to make up data for the missing images to make it balance (the code can be found in 4th and 5th block)
- Essentially the logic is to find the max number of images separated by class then create new image in other classes so that it could reach around that max number
- I create new images by randomly generate brightness of the image

- As the second step, I decided to convert the images to grayscale because it gives better accuracy from my experiment below, it's also inline with Yan Lecun [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf):"The ConvNet was trained with full supervision on the color images of the GTSRB dataset and reached 98.97% accuracy on the phase 1 test set. After the end of phase 1, additional experiments with grayscale images established a new record accuracy of 99.17%"
It also helps to reduce the training time

- Here is an example of a traffic sign image going through the pre-process pipeline

![alt text][image3]

- I normalized the image data because it ensures that each pixel has a similar data distribution. This makes convergence faster while training the network
- The formula to normalize is *(image_data - image_data.mean())/image_data.std()* as recommended from [here](http://cs231n.github.io/neural-networks-2/)

**Notes:** as you can see in the code, I try different type of image conversion: Gray, RGB, HSV, HLS ... a quick conclusion is that Gray & RGB provide the best result

#### 2. Table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x20	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x40 				|
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 10x10x40	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x40   				|
| Flatten		      	| output: 1000					   				|
| Fully connected		| output: 300 									|
| Dropout		| keep_prob: 0.5  									|
| Softmax				| logits: 43   									|


#### 3. Iteractive process to train my model:
I used Lenet architecture with following hyperparameters:
- AdamOptimizer
- BATCH_SIZE = 128 (later change to 100)

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

##### Iterative training 9
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
- **Comment:** introduce dropout and increase epoch, even worse. Accuracy is still unstable

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
- **Comment:** introduce *dropout* and increase epoch more, not bad but not as good as no dropout


##### Iterative training 15
- EPOCHS = 50
- Learningrate = 0.001
- Image type: Gray
- DROPOUT 0.5
- convo1=40
- convo2= 20
- FC1=500 → 200
![alt text][image21]
- **Validation Accuracy** = 0.967
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.951

- **Comment:** tweak the convolution layers looks promising. The model is overfitting although dropout is introduced

##### Iterative training 16
- EPOCHS = 50
- Learningrate = 0.001
- Image type: Gray
- DROPOUT 0.5
- convo1=20
- convo2= 40
- FC1=1000 → 300
![alt text][image22]
- **Validation Accuracy** = 0.975
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.953

- **Comment:** switch the *convolution layers* (20, 40) increases the test result a little bit. The model is still overfitting

##### Iterative training 17
- EPOCHS = 30
- BATCH_SIZE = 128
- Learningrate = 0.001
- DROPOUT 0.5
- convo1=20
- convo2= 40
- FC1=1000 → 300
![alt text][image23]
- **Validation Accuracy** = 0.977
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.951
- **Comment:** reduce the *Epoch* to 30. The model is still overfitting

##### Iterative training 18
- AUGMENTED
- EPOCHS = 30
- BATCH_SIZE = 128
- Learning rate = 0.001
- dropout=0.5
- convo1=20
- convo2= 40
- FC1=1000 → 300
![alt text][image24]
- **Validation Accuracy** = 0.965
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.955
- **Comment:** use *balanced augmented* images. The model is still overfitting but test accuracy increases a little bit

##### Iterative training 19  :boom::boom::boom::boom::boom:
- EPOCHS = 50
- BATCH_SIZE = 100
- rate = 0.001
- DROPOUT 0.5
- convo1=20
- convo2= 40
- FC1=1000 → 300
![alt text][image25]
- **Validation Accuracy** = 0.978
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.962
- **Comment:** drop the *augmented* images. Test accuracy is highest. However, model is still overfitting

:boom::boom::boom::boom::boom: **I choose this as the final model**

#### Conclusion:
- Introduce more epochs doesn't mean the network gets better
- Dropout seems working to reduce overfitting but not apparent in this model
- This architect clearly reaches its peak around 95%-96&, in order to increase the accuracy, better image preprocessing (rotate, projection ...) could be useful
- After adopting Training accuracy from iterative training 15, I can clearly see the model is overfitting. In an attempt to fight this symptom, I decrease the convo layers, remove fully-connected, decrease epochs, use Augmented data ... but those changes don't seem to work. Again, better preprocessing could be a useful tool

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.

My final model results were in Iterative training 9:
- **Validation Accuracy** = 0.978
- **Training Accuracy** = 1.000
- **Test Accuracy** = 0.962

### Test a Model on New Images

#### 1. Five random German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image18]

The first image might be difficult to classify because of the light and blurness of the sign

#### 2. The model's predictions on these new traffic signs

![alt text][image19]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Dangerous curve to the right	      		| General caution				 				|
| Ahead only     			| Ahead only 										|
| Turn right ahead					| Turn right ahead											|
| Children crossing      		| Children crossing   									| 
| Keep right			| Keep right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy on the test set of 0.962

#### 3. 5 softmax probabilities for each prediction
The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a 'General cautions' sign (probability of 0.97). Other probabilities are not too low. However all of 5 highest prob are incorrect

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.97         			| General caution   									| 
| 2.88549475e-02     				| Right-of-way at the next intersection 										|
| 3.21525469e-04					| Priority road											|
| 2.70083819e-05	      			| Speed limit (30km/h)					 				|
| 4.84815075e-07				    | End of speed limit (80km/h)      							|

For the second image, the model is really sure that this is a Ahead only sign (probability of 1). Other classifications are very low

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Ahead only   									| 
| 2.14249171e-32     				| Turn right ahead 										|
| 1.04943693e-34     				| Priority road 										|
| 1.74392508e-35     				| Road work 										|
| 1.71017388e-36     				| Dangerous curve to the right 										|


For the third image, the model is quite sure that this is a Turn right ahead sign (probability of 1). Other classifications are very low

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Turn right ahead   									| 
| 3.87683199e-23     				| Stop        |
| 1.84349099e-28					| Go straight or left|
| 2.72228731e-30	      			| Ahead only|
| 9.72755253e-32				    | Keep left|


For the fourth image, the model is quite sure that this is a Children crossing sign (probability of 1). Other classifications are very low

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Children crossing| 
| 9.29654183e-12     				| Bicycles crossing|
| 3.86336101e-19					| Speed limit (60km/h)|
| 2.78409607e-20					| Priority road|
| 1.41634162e-20					| Speed limit (120km/h)|

For the fifth image, the model is 100% sure that this is a Keep right sign (probability of 1) and there's no other classification 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Keep right   									| 
| 0     				| all other classes 										|


### Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

- I choose to visualize the convolution layer of the first image that was incorrectly identify above
- The model can clearly identify the boundary of the sign
- One feature is blackout
- The model has difficulty in identifying the sign symbol at the center, that's why the prediction varies from speed class to turning sign and finally end up at General caution (which literally has a line, exclamation mark, at the center)

![alt text][image20]
