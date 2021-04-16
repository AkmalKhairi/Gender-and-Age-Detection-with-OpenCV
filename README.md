# GENDER AND AGE DETECTION USING DEEP LEARNING 

## A. PROJECT SUMMARY

**Project Title:** Gender and Age Detection

![family](https://user-images.githubusercontent.com/44885554/114744544-6329f180-9d80-11eb-874b-4e89cc841c5d.png)

**Team Members:** 
- MUHAMMAD AKMAL BIN MOHD SABRI
- WAN MUHAMMAD ISMAT BIN WAN AZMY
- MUHAMMAD AKMAL KHAIRI BIN ABDUL HALIM
- MUHAMMAD IMRAN BIN ISMAIL

**Objectives:**
- To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using Deep Learning on the Adience dataset.

##  B. ABSTRACT 

In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression.

**The CNN Architecture:**
The convolutional neural network for this python project has 3 convolutional layers:

- Convolutional layer; 96 nodes, kernel size 7
- Convolutional layer; 256 nodes, kernel size 5
- Convolutional layer; 384 nodes, kernel size 3

It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

- Detect faces
- Classify into Male/Female
- Classify into one of the 8 age ranges
- Put the results on the image and display it

![python-project-example-output-1](https://user-images.githubusercontent.com/58213194/114746369-3f67ab00-9d82-11eb-8f39-0730d750ef30.png)

**Figure 1:** AI output of detecting the user's gender & age.

## C.  DATASET

For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

**Prerequisites:**

You’ll need to install OpenCV (cv2) to be able to run this project. You can do this with pip-

```python
  pip install opencv-python
```

Other packages you’ll be needing are math and argparse, but those come as part of the standard Python library.

**Purposes:**

- Detect human's gender either ‘Male’ and ‘Female’ in images
- Detect human's age in images
- Detect human's gender & age in real-time video streams

Let’s try this gender and age classifier out on some of our own images now.

We’ll get to the command prompt, run our script with the image option and specify an image to classify:

**Example 1**

![interesting-python-project-example](https://user-images.githubusercontent.com/58213194/114747254-388d6800-9d83-11eb-9748-dfaeb2f416d1.png)

**Figure 2:** Example python command to execute & evaluate the image from the dataset.

**Output:**

![python-project-example-output-1](https://user-images.githubusercontent.com/58213194/114747518-80ac8a80-9d83-11eb-9402-5a7699d5d153.png)

**Figure 3:** Program output after evaluating the image provided into the command.

***

**Example 2**

![2](https://user-images.githubusercontent.com/58213194/114747675-ac2f7500-9d83-11eb-9b6a-641f3e778a93.png)

**Figure 4:** Example python command to execute & evaluate the second image from the dataset.

**Output:**

![22](https://user-images.githubusercontent.com/58213194/114747704-b3568300-9d83-11eb-991e-6f5326d95b21.png)

**Figure 5:** Program output after evaluating the second image provided into the command.

***

**Example 3**

![3](https://user-images.githubusercontent.com/58213194/114747910-ea2c9900-9d83-11eb-8cfa-b8b98aa0f5b8.png)

**Figure 6:** Example python command to execute & evaluate the third image from the dataset.

**Output:**

![33](https://user-images.githubusercontent.com/58213194/114747968-fa447880-9d83-11eb-8552-8b296cd14b69.png)

**Figure 7:** Program output after evaluating the third image provided into the command.

***

**Example 4**

![4](https://user-images.githubusercontent.com/58213194/114748191-3d065080-9d84-11eb-8c20-4d1287070689.png)

**Figure 8:** Example python command to execute & evaluate the fourth image from the dataset.

**Output:**

![44](https://user-images.githubusercontent.com/58213194/114748224-45f72200-9d84-11eb-8386-69d3a754cf40.png)

**Figure 9:** Program output after evaluating the fourth image provided into the command.

***

**Example 5**

![5](https://user-images.githubusercontent.com/58213194/114748285-57d8c500-9d84-11eb-960a-c0b67a25f9fc.png)

**Figure 10:** Example python command to execute & evaluate the fifth image from the dataset.

**Output:**

![55](https://user-images.githubusercontent.com/58213194/114748307-5dcea600-9d84-11eb-98d0-b6a04b6a571b.png)

**Figure 11:** Program output after evaluating the fifth image provided into the command.

***

**Example 6**

![6](https://user-images.githubusercontent.com/58213194/114748359-6cb55880-9d84-11eb-8356-51be189eb69b.png)

**Figure 12:** Example python command to execute & evaluate the sixth image from the dataset.

**Output:**

![66](https://user-images.githubusercontent.com/58213194/114748392-76d75700-9d84-11eb-8f67-d76539b2933e.png)

**Figure 13:** Program output after evaluating the sixth image provided into the command.

## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --filelimit 14
- .
- ├── age_deploy.protxt
- ├── age_net.caffeemodel
- ├── gad.py
- ├── age_deploy.protxt
- ├── age_net.caffeemodel
- ├── girl1.jpg
- ├── girl2.jpg
- ├── kid1.jpg
- ├── man1.jpg
- ├── minion.jpg
- ├── opencv_face_detector.pbtxt
- ├── opencv_face_detector.uint8.pb
- ├── woman1.jpg
- ├── woman2.jpg
- 14 files


The dataset/ directory contains the data described in the “Gender and Age Detection” section.

Eight image examples/ are provided so that you can test the static image gender and age detector.

We’ll be reviewing a Python scripts in this tutorial:

- detect_mask_image.py: Performs gender and age detection in static images & using your webcam, this script applies Age & Gender detection to every frame in the stream

In the next two sections, we will train our Age & Gender detector.


## E   TRAINING THE AGE & GENDER DETECTION

We are now ready to train our face mask detector using Keras, TensorFlow, and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python gad.py --image girl1.jpg
- Gender: Female
- Age: 25-32 years
- 
- $ python gad.py --image man1.jpg
- Gender: Male
- Age: 60-100 years
- 
- $ python gad.py --image kid1.jpg
- Gender: Male
- Age: 4-6 years

<img width="800" alt="opencv_age_detection_confusion_matrix" src="https://user-images.githubusercontent.com/73923156/114961386-7b426380-9e9b-11eb-9994-2531d74e8633.png">


**Figure 4: Age & Gender detector confusion matrix

As you can see, one of the biggest issues with the age prediction model is that it’s heavily biased toward the age group 25-32.

Looking at Figure 4, we can see that our model may predict the 25-32 age group when in fact the actual age belongs to a different age bracket.

**You can combat this bias by:**

- Gathering additional training data for the other age groups to help balance out the dataset

- Applying class weighting to handle class imbalance

- Being more aggressive with data augmentation

- Implementing additional regularization when training the model

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting Age & Gender with OpenCV in real-time

You can then launch the mask detector in real-time video streams using the following command:
- $ python gad.py

![Capture](https://user-images.githubusercontent.com/73923156/114885569-5ae1bd00-9e39-11eb-9b26-8338096ac69c.JPG)


**Figure 5: Age & Gender Detection in real-time video streams

In Figure 5, you can see that our Age & Gender detector is capable of running in real-time (and is correct in its predictions as well.



## G.   PROJECT PRESENTATION 

In this python project, we implemented Convolutional Neural Network (CNN) to detect gender and age from a single picture of a face

**What is Computer Vision?**

Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

**What is OpenCV?**

OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

**What is a CNN?**

A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

[![image](https://user-images.githubusercontent.com/73923156/114887390-00496080-9e3b-11eb-8c25-02ee7df3f450.png)](https://www.youtube.com/watch?v=ReeccRD21EU "demo")
