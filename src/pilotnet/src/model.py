# This file handles the neural network of PilotNet
# One key difference from the original paper is that we have 3 output neurons (throttle, brake & steering)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime

class PilotNet():
    def __init__(self, width, height, predict=False):
        self.image_height = height
        self.image_width = width
        self.model = self.build_model() if predict == False else []
    
    def build_model(self):
        inputs = keras.Input(name='input_shape', shape=(self.image_height, self.image_width, 3))
        
        # convolutional feature maps
        x = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')(inputs)
        x = layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
        x = layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)

        # flatten layer
        x = layers.Flatten()(x)

        # fully connected layers with dropouts for overfit protection
        x = layers.Dense(units=1152, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=100, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=50, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=10, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)

        # derive steering angle value from single output layer by point multiplication
        steering_angle = layers.Dense(units=1, activation='linear')(x)
        steering_angle = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='steering_angle')(steering_angle)

        # derive throttle pressure value from single output layer by point multiplication
        throttle_press = layers.Dense(units=1, activation='linear')(x)
        throttle_press = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='throttle_press')(throttle_press)

        # derive brake pressure value from single output by point multiplication
        brake_pressure = layers.Dense(units=1, activation='linear')(x)
        brake_pressure = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='brake_pressure')(brake_pressure)

        # build and compile model
        model = keras.Model(inputs = [inputs], outputs = [steering_angle, throttle_press, brake_pressure])
        model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-4),
            loss = {'steering_angle': 'mse', 'throttle_press': 'mse', 'brake_pressure': 'mse'}
        )
        model.summary()
        return model

    def train(self, name: 'Filename for saving model', data: 'Training data as an instance of pilotnet.src.Data()', epochs: 'Number of epochs to run' = 30, steps: 'Number of steps per epoch' = 10, steps_val: 'Number of steps to validate' = 10, batch_size: 'Batch size to be used for training' = 64):
        # x_train & y_train are np.array() objects with data extracted directly from the PilotData object instances

        # fit data to model for training
        self.model.fit(np.array([frame.image for frame in data.training_data()]), np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.training_data()]), batch_size=batch_size, epochs=epochs, steps_per_epoch=steps, validation_split=0.2, validation_steps=steps_val)
        # test the model by fitting the test data
        stats = self.model.evaluate(np.array([frame.image for frame in data.testing_data()]), np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.testing_data()]), verbose=2)
        # print the stats
        print(f'Model accuracy: {stats[1]}\nModel loss: {stats[0]}')
        input('\nPress [ENTER] to continue...')
        # save the trained model
        self.model.save(f"models/{name}.h5")
    
    # this method can be used for enabling the feature mentioned in app.py but needs more work
    def predict(self, data, given_model = 'default'):
        if given_model != 'default':
            try:
                # load the model
                model = keras.models.load_model(f'models/{given_model}', custom_objects = {"tf": tf})
            except:
                raise PilotError('An unexpected error occured when loading the saved model. Please rerun...')
        else: model = self.model
        # predict using the model
        predictions = model.predict(data.image)
        return predictions
        