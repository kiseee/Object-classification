# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from numpy import argmax


class CNNModel():
    def __int__(self):
        self.model = None
        
    def initilize(self):
        self.model = keras.Sequential([
            # reshape pictures from 784 * 1 to 28 * 28
            keras.layers.Reshape((28, 28, 1), input_shape=(28*28, )),
            
            # CNN
            keras.layers.Conv2D(filters=64, kernel_size=3, use_bias=False, padding="same"),
            keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
            keras.layers.ReLU(),
            # CNN
            keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same', use_bias=False),
            keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
            keras.layers.ReLU(),
            
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
        
    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(x=X_train, y=y_train, batch_size=512, epochs=100, verbose=0, 
                       callbacks=[keras.callbacks.EarlyStopping(patience=10)], 
                       validation_data=(X_test, y_test))
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return argmax(predictions, axis=1)

    def save(self, filename='cnn_model.h5'):
        self.model.save(filename)

    def load(self, filename='cnn_model.h5'):
        self.model = keras.models.load_model(filename)