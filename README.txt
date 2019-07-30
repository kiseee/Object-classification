README
Assignment 2
This assignment will implement KNN, CNN and random forest on Fashion-MINST.
Getting Started
The main.ipynb contain the mian function, that implementing three methods. 
The mnist_reader.py includes function to load dataset.
The CNNModel.py includes the CNN model.
Prerequisites
You need to install TensorFlow with GPU version, this will reduce the running. However, it still can be run under CPU environment. 
In order to save time, there are three saved models are existing, you can load them to test by following command.
knn = joblib.load('kin.model')
rf=joblib.load('rf.model')
cnn = CNNModel.CNNModel()
cnn.load()