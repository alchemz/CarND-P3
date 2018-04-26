import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

samples=[]
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	#center
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                #left
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                #right
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                #steering angle
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #correction
                correction=0.1
                images.append(left_image)
                angles.append(center_angle+correction)
                images.append(right_image)
                angles.append(center_angle-correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def _normalize(X):
    a = -0.1
    b = 0.1
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)

#normalize image(0,1), mean center(0.5, 0)
model = Sequential()
model.add(Cropping2D(cropping=((50, 30), (0, 0)),input_shape=(160, 320, 3)))
model.add(Lambda(_normalize))


model.add(Convolution2D(3, 1, 1))
# reshape image by 1/4 using average pooling later
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu'))
model.add(MaxPooling2D((2, 2), (1, 2)))
model.add(Convolution2D(48, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))


# Fully connected layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
	steps_per_epoch= len(train_samples),
	validation_data=validation_generator, 
	validation_steps=len(validation_samples), 
	epochs=5, verbose = 1)

model.save('generator.h5')