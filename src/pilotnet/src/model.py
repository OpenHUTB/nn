# This file handles the neural network of PilotNet
# One key difference from the original paper is that we have 3 output neurons (throttle, brake & steering)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import datetime
from utils.logger import logger

class PilotNet():
    def __init__(self, width, height, predict=False):
        self.image_height = height
        self.image_width = width
        logger.info(f'Initializing PilotNet model with input size: {width}x{height}')
        self.model = self.build_model() if predict == False else []
    
    def build_model(self):
        logger.info('Building PilotNet neural network model')
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

        # derive brake pressure value from single output layer by point multiplication
        brake_pressure = layers.Dense(units=1, activation='linear')(x)
        brake_pressure = layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name='brake_pressure')(brake_pressure)

        # build and compile model
        model = keras.Model(inputs = [inputs], outputs = [steering_angle, throttle_press, brake_pressure])
        model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-4),
            loss = {'steering_angle': 'mse', 'throttle_press': 'mse', 'brake_pressure': 'mse'}
        )
        model.summary()
        logger.info('PilotNet model built successfully')
        return model

    def train(self, name: 'Filename for saving model', data: 'Training data as an instance of pilotnet.src.Data()', epochs: 'Number of epochs to run' = 50, steps: 'Number of steps per epoch' = None, steps_val: 'Number of steps to validate' = None, batch_size: 'Batch size to be used for training' = 64):
        # x_train & y_train are np.array() objects with data extracted directly from the PilotData object instances
        training_frames = data.training_data()
        testing_frames = data.testing_data()
        
        # Calculate steps automatically based on data size
        num_train_samples = len(training_frames)
        num_val_samples = int(num_train_samples * 0.2)
        
        if steps is None:
            steps = max(1, num_train_samples // batch_size)
        if steps_val is None:
            steps_val = max(1, num_val_samples // batch_size)
        
        logger.info(f'Starting model training - epochs: {epochs}, steps: {steps}, batch_size: {batch_size}')
        logger.info(f'Training samples: {num_train_samples}, Validation samples: {num_val_samples}')
        
        # Prepare data
        x_train = np.array([frame.image for frame in training_frames])
        y_train = np.array([(frame.steering, frame.throttle, frame.brake) for frame in training_frames])
        
        # Learning rate scheduler - reduce learning rate when stuck
        def lr_scheduler(epoch, lr):
            if epoch % 10 == 0 and epoch > 0:
                new_lr = lr * 0.8
                logger.info(f'Reducing learning rate from {lr:.6f} to {new_lr:.6f}')
                return new_lr
            return lr
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'models/{name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        lr_schedule = LearningRateScheduler(lr_scheduler)
        
        callbacks = [early_stopping, model_checkpoint, lr_schedule]
        
        # fit data to model for training
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_split=0.2,
            validation_steps=steps_val,
            callbacks=callbacks,
            shuffle=True
        )
        
        # test the model by fitting the test data
        logger.info('Evaluating model on test data')
        x_test = np.array([frame.image for frame in testing_frames])
        y_test = np.array([(frame.steering, frame.throttle, frame.brake) for frame in testing_frames])
        stats = self.model.evaluate(x_test, y_test, verbose=2)
        
        # print the stats
        print(f'\nModel Evaluation Results:')
        print(f'  - Total Loss: {stats[0]:.6f}')
        print(f'  - Steering Angle Loss: {stats[1]:.6f}')
        print(f'  - Throttle Pressure Loss: {stats[2]:.6f}')
        print(f'  - Brake Pressure Loss: {stats[3]:.6f}')
        logger.info(f'Training completed - loss: {stats[0]}, steering_loss: {stats[1]}, throttle_loss: {stats[2]}, brake_loss: {stats[3]}')
        
        input('\nPress [ENTER] to continue...')
        
        # save the trained model
        self.model.save(f"models/{name}.h5")
        logger.info(f'Model saved to: models/{name}.h5')
        
        return history
    
    # this method can be used for enabling the feature mentioned in app.py but needs more work
    def predict(self, data, given_model = 'default'):
        logger.info(f'Starting prediction with model: {given_model}')
        if given_model != 'default':
            try:
                # load the model
                model = keras.models.load_model(f'models/{given_model}', custom_objects = {"tf": tf})
                logger.info(f'Model loaded successfully: {given_model}')
            except Exception as e:
                logger.error(f'Failed to load model {given_model}: {e}')
                raise PilotError('An unexpected error occured when loading the saved model. Please rerun...')
        else: 
            model = self.model
            logger.info('Using current model for prediction')
        
        # predict using the model
        predictions = model.predict(data.image)
        logger.info('Prediction completed successfully')
        return predictions
        