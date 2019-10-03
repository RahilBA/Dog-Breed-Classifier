# Dog Breed Classifier
This work is part of Udacity’s Data Science capstone project. 

## project overview
The goal is to create a pipeline that detects dog images and classifies them according to their breed using CCN . This model can be used as part of a mobile or web app for real world and user provided images.Given an image to the model, it will return if an image includes dog and an estimation of breed. If the image is not a dog, it will return the resembling dog breed. You can read more about this work in my blog here.

## contents:

    Intro
    Step 0: Import Datasets
    Step 1: Detect Humans
    Step 2: Detect Dog
    Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
    Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
    Step 5: Write Your Algorithm
    Step 6: Test Your Algorithm


## Data
You can download the dog and human data sets here. 

## Dog and human detector
I used OpenCV’s implementation of Haar feature-based cascade object classifier to detect human faces in images. But first the images have to be converted to grayscale before using the detector. This detector has a 11% false positive rate tested on 100 sample dog files. So, I used pretrained ResNet_50 weights in keras to detect dog from images. If the predicted class of RestNet50 on ImageNet falls into the dog breed categories dog detector performs well and if not we can use another keras models as dog detector. But, it performed well without false positive. 

## Dog classifier
Here I created a 4-layer CNN in Keras that classifies dog breeds, but the accuracy is about 12% which is not that far from random guessing. So, I leveraged the latest state of art techiniques like VGG, Inception V3, and ResNet to classify dogs. The Exception model outperforms all the other models with the accuracy of 86%. 

## Files
