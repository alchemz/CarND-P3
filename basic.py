import csv
import cv2
import numpy as np

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

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model1.h5')