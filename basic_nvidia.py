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

lines=[]
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)

images=[]
measurements=[]
for line in lines:
	#center
	center_current = 'data/IMG/'+line[0].split('/')[-1]
	center = cv2.imread(center_current)
	#left
	left_current = 'data/IMG/'+line[1].split('/')[-1]
	left = cv2.imread(left_current)
	#right
	right_current = 'data/IMG/'+line[2].split('/')[-1]
	right = cv2.imread(right_current)

	images.append(center)
	steering = float(line[3])
	measurements.append(steering)

	correction=0.1
	images.append(left)
	measurements.append(steering+correction)
	images.append(right)
	measurements.append(steering-correction)


X_train=np.array(images)
y_train=np.array(measurements)

#normalize image(0,1), mean center(0.5, 0)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,  activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(36,5,5, activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(48,5,5, activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('nvidia.h5')