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
	center_path = line[0]
	center_name = center_path.split('/')[-1]
	center_current = 'data/IMG/'+center_name
	center = cv2.imread(center_current)
	images.append(center)
	steering = float(line[3])
	measurements.append(steering)

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

model.save('pooling.h5')