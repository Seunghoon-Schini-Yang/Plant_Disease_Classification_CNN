import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_tsv = pd.read_csv('E:\\Python\\Bigdata leader\\인공지능 문제해결 경진대회\\train.tsv', header=None)
train_tsv.columns = ['name']
train_tsv['image'] = train_tsv.name.str.split('\t').str[0]
train_tsv['label'] = train_tsv.name.str.split('\t').str[1] + '_' + train_tsv.name.str.split('\t').str[2]
train_labels = train_tsv['label']

base_dir = 'E://Python//Bigdata leader//인공지능 문제해결 경진대회//train'

data_generator = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = data_generator.flow_from_directory(
    base_dir,
    subset='training',
    target_size=(256, 256),
    batch_size=1,
    class_mode='categorical'
)

val_generator = data_generator.flow_from_directory(
    base_dir,
    subset='validation',
    target_size=(256, 256),
    batch_size=1,
    class_mode='categorical'
)

# from tensorflow.python.types import core as core_tf_types

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 20, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(train_generator,
steps_per_epoch=160,
epochs =1,
validation_data =val_generator,
validation_steps=40)

test_image = image.load_img(''E:\\Python\\Bigdata leader\\인공지능 문제해결 경진대회\\test\\0.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
