import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Folder for storing data
data_path = 'data/'
img_path = data_path + 'IMG/'
log_file = data_path + 'driving_log.csv'


# extract path to the camera images
samples = []
with open(log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del(samples[0])

# Randomly split data into training and testing
samples_train, samples_valid = train_test_split(samples, test_size=0.2)

#use a generator to load data and preprocess it on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = img_path + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Augment the image by flipping it
                images.append(cv2.flip(center_image, 1))
                angles.append(-1.0* center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

            
# compile and train the model using the generator function
batch_train = generator(samples_train, batch_size=32)
batch_valid = generator(samples_valid, batch_size=32)

# Define the model
model = Sequential()
model.add(Lambda(lambda x : x /255.0 - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolution layers with elu activation
model.add(Conv2D(24, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="elu"))
model.add(Conv2D(64, (3,3), activation="elu"))
model.add(Conv2D(64, (3,3), activation="elu"))

# Dropout layers to avoid overfitting
model.add(Dropout(0.5))
model.add(Flatten())

# Fully connected layers 
model.add(Dense(264, activation="elu"))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))

# Linear for regresstion (steering angle)
model.add(Dense(1, activation="linear"))
model.summary()

# Train the model with Adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(batch_train, steps_per_epoch = len(samples_train), epochs=3, 
                    validation_data = batch_valid, 
                    validation_steps = len(samples_valid))

# Save the model and exit
model.save('model_toy.h5')
exit()