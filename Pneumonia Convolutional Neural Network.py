import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#################################################### PRE-PROCESSING THE DATA ############################################################################
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_gen = ImageDataGenerator(rescale = 1./255)
#Here, we set up the spyder working directory to contain the train and validation folders of images
training_set = train_gen.flow_from_directory('train', target_size = (64, 64),batch_size = 32, class_mode = 'binary')
val_set = test_gen.flow_from_directory('val', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
####################################################### CREATING THE CONVOLUTIONAL NEURAL NETWORK #########################################################################
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set, steps_per_epoch = 5216, epochs = 25,  validation_data = val_set, validation_steps = 16)
###########################################################################################################################################################################