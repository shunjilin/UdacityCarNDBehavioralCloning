import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from tensorflow.python.client import device_lib

# data augmentation functions


# flip image and return corresponding adjusted steering angle
def flip_image(image, steering_angle):
    return np.fliplr(image), -steering_angle


# returns corrected angle of left camera image
def correct_left(steering_angle, correction=0.2):
    return steering_angle + correction


# returns corrected angle of right camera image
def correct_right(steering_angle, correction=0.2):
    return steering_angle - correction


# generate training / validation images
# includes data augmentation
# creates csv file with image filepath : steering angle pairs
def generate_data():
    if os.path.isdir('./generated/IMG/'):
        # data is already generated
        return

    os.makedirs('./generated/IMG/')
    generated_csv = open('generated.csv', 'w+')
    generated_csv_writer = csv.writer(generated_csv)

    image_count = 0
    with open('./data/driving_log.csv') as data_csvfile:
        data_reader = csv.reader(data_csvfile)
        for line in data_reader:
            angle = float(line[3])
            for pos in range(3):  # center, left, right images
                name = './data/IMG/' + line[pos].split('/')[-1]
                if pos == 1:  # left camera correction
                    angle = correct_left(angle)
                if pos == 2:  # right camera correction
                    angle = correct_right(angle)
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output_path = './generated/IMG/' + str(image_count) + '.jpg'
                cv2.imwrite(output_path, image)
                image_count += 1
                generated_csv_writer.writerow([output_path, angle])

                # generate flipped image
                image, angle = flip_image(image, angle)
                output_path = './generated/IMG/' + str(image_count) + '.jpg'
                cv2.imwrite(output_path, image)
                image_count += 1
                generated_csv_writer.writerow([output_path, angle])


# generator for lazy evaluation of data during training of network
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


if __name__ == "__main__":
    
    generate_data()
    print(device_lib.list_local_devices())

    samples = []
    with open('./generated.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples,
                                                         test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # NVIDIA model
    # https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    model = Sequential()

    # normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # crop 75 pixels top, 25 pixels bottom, 0 left and right
    model.add(Cropping2D(cropping=((75, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, activation="relu",
                            subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation="relu",
                            subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation="relu",
                            subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation="relu",
                            subsample=(1, 1)))
    model.add(Convolution2D(64, 2, 2, activation="relu",
                            subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    print(model.summary())

    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(
        train_generator, samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=10,
        verbose=1)

    # save model
    model.save('model.h5')

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
