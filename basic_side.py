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
	center_path = line[0]
	center_name = center_path.split('/')[-1]
	center_current = 'data/IMG/'+center_name
	center = cv2.imread(center_current)
	#left
	left_path = line[1]
	left_name = left_path.split('/')[-1]
	left_current = 'data/IMG/'+left_name
	left = cv2.imread(left_current)
	#right
	right_path = line[2]
	right_name = right_path.split('/')[-1]
	right_current = 'data/IMG/'+right_name
	right = cv2.imread(right_current)

	images.append(center)
	steering = float(line[3])
	measurements.append(steering)

	correction=0.1
	images.append(left)
	measurements.append(steering+correction)
	images.append(right)
	measurements.append(steering-correction)

# augmented_images, augmented_measurements=[],[]
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)
# 	augmented_images.append(cv2.flip(image,1 ))
# 	augmented_measurements.append(measurement*-1.0)

# X_train=np.array(augmented_images)
# y_train=np.array(augmented_measurements)
X_train=np.array(images)
y_train=np.array(measurements)

#normalize image(0,1), mean center(0.5, 0)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model_pooling.h5')