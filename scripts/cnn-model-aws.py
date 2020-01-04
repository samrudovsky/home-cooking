import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.pyplot import imshow

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Model, load_model, Sequential
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import decode_predictions
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers

from PIL import Image
from glob import glob

####################
# Train neural net #
####################

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3)) # exclude the final dense layer

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Establish new fully connected block
x = base_model.output
x = Flatten()(x) 
x = Dense(700, activation='relu')(x)
x = Dense(700, activation='relu')(x)
predictions = Dense(13, activation='softmax')(x) # multiclass predictions

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image augmentation
train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split = 0.25)

batch_size = 25
target_size = (150, 150)

train_generator = train_datagen.flow_from_directory(
    'downloads', # subdirectory with labeled images
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # training set

validation_generator = train_datagen.flow_from_directory(
    'downloads',
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # validation set

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 40,
    verbose = 2)

model.save('model.h5') # save model 
