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

## ACKNOWLEDGEMENTS

Our group makes use of the following open source projects:

 - [data-flair.training](https://data-flair.training/blogs/python-project-gender-age-detection/)

## B. ABSTRACT 

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

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/58213194/114746369-3f67ab00-9d82-11eb-8f39-0730d750ef30.png" alt="Material Bread logo">
</p>

<p align="center">
Figure 1: AI output of detecting the user's gender & age.
</p>

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
- ├── opencv_face_detector.pbtxt
- ├── opencv_face_detector.uint8.pb
- ├── Dataset
- │ └── anwar-1.jpg
- │ └── iman naim-1.jpg
- │ └── kj-1.jpg
- │ └── mahathir-1.jpg
- │ └── saddiq-1.jpg
- │ └── taju-1.jpg
- │ └── girl1.jpg
- │ └── kid1.jpg
- │ └── man1.jpg
- │ └── woman1.jpg
- │ └── woman2.jpg
- 14 files

The dataset/ directory contains the data described in the “Gender and Age Detection” section. Eleven image examples/ are provided so that you can test the static image gender and age detector.

* In the next two sections, we will train our Age & Gender detector.

## E.   TRAINING THE AGE & GENDER DETECTION
<p align="center">
     <img width="800" alt="opencv_age_detection_confusion_matrix" src="https://user-images.githubusercontent.com/73923156/114961386-7b426380-9e9b-11eb-9994-2531d74e8633.png">
</p>

<p align="center">
     Figure 14: Age estimation confusion matrix benchmark
</p>

* The age groups 0-2, 4-6, 8-13 and 25-32 are predicted with relatively high accuracy. ( see the diagonal elements )
* The output is heavily biased towards the age group 25-32 ( see the row belonging to the age group 25-32 ). This means that it is very easy for the network to get confused between the ages 15 to 43. So, even if the actual age is between 15-20 or 38-43, there is a high chance that the predicted age will be 25-32. This is also evident from the Results section.

Apart from this, we observed that the accuracy of the models improved if we use padding around the detected face. This may be due to the fact that the input while training were standard face images and not closely cropped faces that we get after face detection.

We also analysed the use of face alignment before making predictions and found that the predictions improved for some examples but at the same time, it became worse for some. It may be a good idea to use alignment if you are mostly working with non-frontal faces.

As you can see, one of the biggest issues with the age prediction model is that it’s heavily biased toward the age group 25-32. Looking at Figure 14, we can see that our model may predict the 25-32 age group when in fact the actual age belongs to a different age bracket.

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


**Figure 15:** Age & Gender Detection in real-time video streams

In Figure 15, you can see that our Age & Gender detector is capable of running in real-time (and is correct in its predictions as well.


## G.   PROJECT PRESENTATION 

In this python project, we implemented Convolutional Neural Network (CNN) to detect gender and age from a single picture of a face

**What is Computer Vision?**

Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

**What is OpenCV?**

OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

**What is a CNN?**

A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

[![image](https://user-images.githubusercontent.com/73923156/114887390-00496080-9e3b-11eb-8c25-02ee7df3f450.png)](https://www.youtube.com/watch?v=ReeccRD21EU "demo")

## H. CONCLUSION

Overall, we think the accuracy of the models is decent but can be improved further by using more data, data augmentation and better network architectures. Thanks to our AI lecturer, Prof. Goh Ong Sing for giving us the opportunity to learn how to implement a real-world AI project using Python. We gain a lot of knowledge throughout this journey, AI is really interesting and crucial in our life.
