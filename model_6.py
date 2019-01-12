import csv
import sklearn
import cv2
from sklearn.model_selection import train_test_split
import numpy as np 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import matplotlib
import os, sys,  json, math
import pickle
matplotlib.use('Agg')

lines = []
with open('data/driving_log.csv') as csvfile:
# with open('new_data/data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

print("Number of data:" )
print(len(lines))
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
left_image_angle_correction = 0.20
right_image_angle_correction = -0.20

DATA_PATH = 'data/IMG/'
# DATA_PATH = 'new_data/data/IMG/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Center image
                name = DATA_PATH + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if center_image is not None:
                    images.append(center_image)
                    angles.append(center_angle)
                    # Data Augmentation: Flip the image horizontally
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)
                # Left image
                name = DATA_PATH +batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                if left_image is not None:
                    images.append(left_image)
                    left_angle = center_angle + left_image_angle_correction
                    angles.append(left_angle)
                    # Data Augmentation: Flip the image horizontally
                    images.append(cv2.flip(left_image, 1))
                    angles.append(left_angle * -1.0)                
                # Rigth image
                name = DATA_PATH + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                if right_image is not None:
                    images.append(right_image)
                    right_angle = center_angle + right_image_angle_correction
                    angles.append(right_angle)
                    # Data Augmentation: Flip the image horizontally
                    images.append(cv2.flip(right_image, 1))
                    angles.append(right_angle * -1.0)        
    
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

model = Sequential()
# Normalization: normalize the image by dividing each element by 255, 
# which is the maximum value of an image pixel.
# Once the image is normalized to a range between 0 and 1, 
# I'll mean center the image by substracting 0.5 from each element, 
# which will shift the element mean down from 0.5 to 0.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Cropping images
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# architecture of NVIDIA team
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(SpatialDropout2D(0.2))
#model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(SpatialDropout2D(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, init='he_normal', activation='elu'))
#model.add(Dropout(0.5))
model.add(Dense(50, init='he_normal', activation='elu'))
#model.add(Dropout(0.25))
model.add(Dense(10, init='he_normal', activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Hyperparameters
#BATCH_SIZE = 32
#EPOCHS = 3
BATCH_SIZE = 64
EPOCHS = 10

#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(lr=0.001))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


checkpointer = ModelCheckpoint('model_best.h5', monitor='val_loss', verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*4, \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples)*4, \
                                     nb_epoch=EPOCHS, verbose=1, callbacks=[checkpointer])

#samples_per_epoch = int(len(training_data) / BATCH_SIZE) * BATCH_SIZE

#history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 4, verbose=1, validation_data=validation_generator, nb_val_samples=len(validation_samples)*4, nb_epoch=EPOCHS)


model.save('model_6.h5')

# history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
#                                      validation_data=validation_generator, 
#                                      nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)


#model.save('models/model.h5')

#with open("history_4.json", "w") as json_file:
#    json_file.write(history_object)



### plot the training and validation loss for each epoch
fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig.savefig('loss_6.png')

#loss = np.concatenate([mh.history['loss'] for mh in history_object])
with open('loss_6.p', 'wb') as fp:
    pickle.dump(history['loss'], fp)

#val_loss = np.concatenate([mh.history['val_loss'] for mh in history_object])
with open('val_loss_6.p', 'wb') as fp:
    pickle.dump(history['val_loss'], fp)
