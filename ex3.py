# __author__ = 'benji'

from __future__ import print_function
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# K.set_image_dim_ordering('th')

batch_size = 200
nb_epoch = 1

# color of input, gray 1, rgb 3
color = 1
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# flatten 28*28 images to use the color and to have a good shape for the entry of the network
X_train = X_train.reshape(X_train.shape[0], color, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], color, img_rows, img_cols).astype('float32')

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0, :, :, :])

# normalize inputs from 0-255 to 0-1
X_train /= 255
X_test /= 255

def mixing(y_train, p):
    # randomize 5% of the training images
    p_t = int(y_train.shape[0] * (1-p))
    start = np.random.randint(0, p_t)
    # print('a', p_t, start, start + int(y_train.shape[0] * p))
    for i in range(start, start + int(y_train.shape[0] * p)):
        y_train[i] = np.random.randint(0, 10)
        # print(y_train[i])
    # print('b*********************************')
    return y_train

# one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# y = y_test[0:10, :]
ya = pd.DataFrame(y_test)
# print(X_test[[0, 2, 4], :, :].shape)
y_cat = []
X_cat = []
for i in range(10):
    y_cat.append(ya[ya[i] > 0].values)
    # print(ya[ya[i] > 0].index)
    X_cat.append(X_test[ya[ya[i] > 0].index, :, :])
    print(y_cat[i].shape, X_cat[i].shape)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=(color, img_rows, img_cols), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_scores(y_train):
    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(scores)
    # Final evaluation of the model for each categories
    for i in range(num_classes):
        scores_i = model.evaluate(X_cat[i], y_cat[i], verbose=1)
        print(scores_i)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    return scores

# y_train = np.sort(y_train)
print(y_train)
y_train_before_categorical = np.copy(y_train)
y_train = np_utils.to_categorical(y_train)
scores = [fit_scores(y_train)[0]]
for i in [0.05, 0.15, 0.5, 0.9, 0.95, 0.99]:
    print('pourcentage ', i)
    y_train_new = np.copy(y_train_before_categorical)
    y_train_mix = mixing(y_train_new, i)
    y_train_mix = np_utils.to_categorical(y_train_mix)
    scores.append(fit_scores(y_train_mix)[0])

plt.plot([0, 5, 15, 50, 90, 95, 99], scores)
plt.show()
