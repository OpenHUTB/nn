# This file handles the neural network of PilotNet
# One key difference from the original paper is that we have 3 output neurons (throttle, brake & steering)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.piloterror import PilotError

class PilotNet():
    def __init__(self, width, height, predict=False):
        self.image_height = height
        self.image_width = width
        self.model = self.build_model() if predict == False else []
        self.log_dir = None
    
    def build_model(self):
        inputs = keras.Input(name='input_shape', shape=(self.image_height, self.image_width, 3))
        
        x = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu', name='conv1')(inputs)
        x = layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu', name='conv2')(x)
        x = layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu', name='conv3')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', name='conv4')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', name='conv5')(x)

        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(units=1152, activation='relu', name='fc1')(x)
        x = layers.Dropout(rate=0.1, name='dropout1')(x)
        x = layers.Dense(units=100, activation='relu', name='fc2')(x)
        x = layers.Dropout(rate=0.1, name='dropout2')(x)
        x = layers.Dense(units=50, activation='relu', name='fc3')(x)
        x = layers.Dropout(rate=0.1, name='dropout3')(x)
        x = layers.Dense(units=10, activation='relu', name='fc4')(x)
        x = layers.Dropout(rate=0.1, name='dropout4')(x)

        steering_angle = layers.Dense(units=1, activation='linear', name='steering_output')(x)
        steering_angle = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='steering_angle')(steering_angle)

        throttle_press = layers.Dense(units=1, activation='linear', name='throttle_output')(x)
        throttle_press = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='throttle_press')(throttle_press)

        brake_pressure = layers.Dense(units=1, activation='linear', name='brake_output')(x)
        brake_pressure = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='brake_pressure')(brake_pressure)

        model = keras.Model(inputs = [inputs], outputs = [steering_angle, throttle_press, brake_pressure])
        model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-4),
            loss = {'steering_angle': 'mse', 'throttle_press': 'mse', 'brake_pressure': 'mse'},
            metrics = {'steering_angle': ['mae'], 'throttle_press': ['mae'], 'brake_pressure': ['mae']}
        )
        model.summary()
        return model

    def create_tensorboard_callback(self, log_dir):
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None
        )
        return tensorboard_callback

    def train(self, name: 'Filename for saving model', data: 'Training data as an instance of pilotnet.src.Data()', epochs: 'Number of epochs to run' = 30, steps: 'Number of steps per epoch' = 10, steps_val: 'Number of steps to validate' = 10, batch_size: 'Batch size to be used for training' = 64, enable_tensorboard: 'Enable TensorBoard logging' = True):
        safe_name = name.replace(':', '-').replace('\\', '-').replace('/', '-').replace(' ', '_')
        self.log_dir = f"logs/{safe_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        x_train = np.array([frame.image for frame in data.training_data()])
        y_train = np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.training_data()])
        
        x_val = np.array([frame.image for frame in data.testing_data()])
        y_val = np.array([(frame.steering, frame.throttle, frame.brake) for frame in data.testing_data()])
        
        callbacks = []
        if enable_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)
            callbacks.append(self.create_tensorboard_callback(self.log_dir))
            print(f"\nTensorBoard log directory: {self.log_dir}")
            print(f"To view TensorBoard, run: tensorboard --logdir={self.log_dir.replace(chr(92), '/')}\n")
        
        self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=(x_val, y_val),
            validation_steps=steps_val,
            callbacks=callbacks,
            verbose=1
        )
        
        stats = self.model.evaluate(x_val, y_val, verbose=2)
        print(f'\nModel Evaluation:')
        print(f'Total Loss: {stats[0]:.4f}')
        print(f'Steering MAE: {stats[1]:.4f}, Throttle MAE: {stats[2]:.4f}, Brake MAE: {stats[3]:.4f}')
        
        os.makedirs('models', exist_ok=True)
        self.model.save(f"models/{name}.h5")
        print(f'\nModel saved to: models/{name}.h5')
        
        if enable_tensorboard:
            print(f'View training results with TensorBoard: tensorboard --logdir={self.log_dir.replace(chr(92), "/")}')
        
        input('\nPress [ENTER] to continue...')
    
    # this method can be used for enabling the feature mentioned in app.py but needs more work
    def predict(self, data, given_model = 'default'):
        if given_model != 'default':
            try:
                model = keras.models.load_model(f'models/{given_model}', custom_objects = {"tf": tf})
            except:
                raise PilotError('An unexpected error occured when loading the saved model. Please rerun...')
        else: model = self.model
        predictions = model.predict(data.image)
        return predictions
    
    def get_log_dir(self):
        return self.log_dir
        