# This file handles the neural network of PilotNet
# One key difference from the original paper is that we have 3 output neurons (throttle, brake & steering)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
import datetime
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time plotting
import matplotlib.pyplot as plt
from utils.logger import logger
from utils.piloterror import PilotError
from utils.screen import message

# Custom callback for real-time loss visualization
class RealTimeLossPlot(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_steering_losses = []
        self.val_steering_losses = []
        self.train_throttle_losses = []
        self.val_throttle_losses = []
        self.train_brake_losses = []
        self.val_brake_losses = []
        self.learning_rates = []  # Track learning rates
        self.epochs = []          # Track epoch numbers
        
        # Initialize plot with 3 rows x 2 columns for better layout
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(14, 14))
        self.fig.suptitle('PilotNet Training Monitor', fontsize=16, y=0.98)
        
        # Create subplots
        self.ax1 = plt.subplot(3, 2, 1)  # Total Loss
        self.ax2 = plt.subplot(3, 2, 2)  # Steering Angle Loss
        self.ax3 = plt.subplot(3, 2, 3)  # Throttle Pressure Loss
        self.ax4 = plt.subplot(3, 2, 4)  # Brake Pressure Loss
        self.ax5 = plt.subplot(3, 2, 5)  # Learning Rate
        self.ax6 = plt.subplot(3, 2, 6)  # Loss Ratio (Train/Val)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.25)
        
    def on_epoch_end(self, epoch, logs=None):
        # Store losses
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_steering_losses.append(logs.get('steering_angle_loss'))
        self.val_steering_losses.append(logs.get('val_steering_angle_loss'))
        self.train_throttle_losses.append(logs.get('throttle_press_loss'))
        self.val_throttle_losses.append(logs.get('val_throttle_press_loss'))
        self.train_brake_losses.append(logs.get('brake_pressure_loss'))
        self.val_brake_losses.append(logs.get('val_brake_pressure_loss'))
        
        # Track epoch number
        self.epochs.append(epoch + 1)
        
        # Get current learning rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if epoch > 0 and epoch % 10 == 0:
            lr = lr * (0.8 ** (epoch // 10))
        self.learning_rates.append(lr)
        
        # Clear and redraw plots
        self._update_plot()
        
    def _update_plot(self):
        # Total Loss plot
        self.ax1.clear()
        self.ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        self.ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.ax1.set_title('Total Loss', fontsize=12)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Steering Angle Loss plot
        self.ax2.clear()
        self.ax2.plot(self.epochs, self.train_steering_losses, 'b-', label='Training', linewidth=2)
        self.ax2.plot(self.epochs, self.val_steering_losses, 'r-', label='Validation', linewidth=2)
        self.ax2.set_title('Steering Angle Loss', fontsize=12)
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend()
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Throttle Loss plot
        self.ax3.clear()
        self.ax3.plot(self.epochs, self.train_throttle_losses, 'b-', label='Training', linewidth=2)
        self.ax3.plot(self.epochs, self.val_throttle_losses, 'r-', label='Validation', linewidth=2)
        self.ax3.set_title('Throttle Pressure Loss', fontsize=12)
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Loss')
        self.ax3.legend()
        self.ax3.grid(True, linestyle='--', alpha=0.7)
        self.ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Brake Loss plot
        self.ax4.clear()
        self.ax4.plot(self.epochs, self.train_brake_losses, 'b-', label='Training', linewidth=2)
        self.ax4.plot(self.epochs, self.val_brake_losses, 'r-', label='Validation', linewidth=2)
        self.ax4.set_title('Brake Pressure Loss', fontsize=12)
        self.ax4.set_xlabel('Epoch')
        self.ax4.set_ylabel('Loss')
        self.ax4.legend()
        self.ax4.grid(True, linestyle='--', alpha=0.7)
        self.ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Learning Rate plot
        self.ax5.clear()
        self.ax5.plot(self.epochs, self.learning_rates, 'g-', label='Learning Rate', linewidth=2, marker='o', markersize=4)
        self.ax5.set_title('Learning Rate', fontsize=12)
        self.ax5.set_xlabel('Epoch')
        self.ax5.set_ylabel('Learning Rate')
        self.ax5.legend()
        self.ax5.grid(True, linestyle='--', alpha=0.7)
        self.ax5.ticklabel_format(style='sci', axis='y', scilimits=(-5,-3))
        
        # Loss Ratio plot (Training / Validation) - helps detect overfitting
        self.ax6.clear()
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            loss_ratio = [train / val if val > 0 else 1 for train, val in zip(self.train_losses, self.val_losses)]
            self.ax6.plot(self.epochs, loss_ratio, 'purple', label='Train/Val Loss Ratio', linewidth=2)
            self.ax6.axhline(y=1.0, color='red', linestyle='--', label='Equal Loss')
        self.ax6.set_title('Loss Ratio (Train/Validation)', fontsize=12)
        self.ax6.set_xlabel('Epoch')
        self.ax6.set_ylabel('Ratio')
        self.ax6.legend()
        self.ax6.grid(True, linestyle='--', alpha=0.7)
        self.ax6.set_ylim(bottom=0.5)
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.05)

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
        
        # Data processing before training
        from src.data_processor import DataProcessor
        
        logger.info('Starting data processing before training...')
        message('='*60)
        message('          数据预处理阶段')
        message('='*60)
        
        # Process training data
        training_frames, train_stats = DataProcessor.process(training_frames)
        
        # Process testing data
        testing_frames, test_stats = DataProcessor.process(testing_frames, enable_balancing=False)
        
        message('数据预处理完成！')
        message('='*60 + '\n')
        
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
        
        # Real-time loss plot callback
        real_time_plot = RealTimeLossPlot()
        
        callbacks = [early_stopping, model_checkpoint, lr_schedule, real_time_plot]
        
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
        
        # 在测试数据上评估模型
        logger.info('Evaluating model on test data')
        x_test = np.array([frame.image for frame in testing_frames])
        y_test = np.array([(frame.steering, frame.throttle, frame.brake) for frame in testing_frames])
        stats = self.model.evaluate(x_test, y_test, verbose=2)
        
        # 打印评估结果
        print(f'\n模型评估结果：')
        print(f'  - 总损失: {stats[0]:.6f}')
        print(f'  - 转向角度损失: {stats[1]:.6f}')
        print(f'  - 油门压力损失: {stats[2]:.6f}')
        print(f'  - 刹车压力损失: {stats[3]:.6f}')
        logger.info(f'Training completed - loss: {stats[0]}, steering_loss: {stats[1]}, throttle_loss: {stats[2]}, brake_loss: {stats[3]}')
        
        input('\n按 [ENTER] 继续...')
        
        # 保存训练好的模型
        self.model.save(f"models/{name}.h5")
        logger.info(f'Model saved to: models/{name}.h5')
        
        return history
    
    # this method can be used for enabling the feature mentioned in app.py but needs more work
    def predict(self, data, given_model = 'default'):
        import os
        logger.info(f'Starting prediction with model: {given_model}')
        if given_model != 'default':
            try:
                # Try to load model from multiple possible paths
                model_path_h5 = f'models/{given_model}.h5'
                model_path_no_ext = f'models/{given_model}'
                
                # Check all possible paths
                possible_paths = [model_path_h5, model_path_no_ext]
                found_path = None
                
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path is None:
                    # List available model files for debugging
                    available_models = []
                    if os.path.exists('models/'):
                        with os.scandir('models/') as entries:
                            for entry in entries:
                                if entry.is_file():
                                    available_models.append(entry.name)
                    raise FileNotFoundError(f"Model file not found. Available models: {available_models}")
                
                # Try to load the model
                model = keras.models.load_model(found_path, custom_objects={"tf": tf})
                logger.info(f'Model loaded successfully from: {found_path}')
                
            except Exception as e:
                logger.error(f'Failed to load model {given_model}: {e}')
                raise PilotError(f'An unexpected error occured when loading the saved model: {e}')
        else: 
            model = self.model
            logger.info('Using current model for prediction')
        
        # predict using the model
        predictions = model.predict(data.image)
        logger.info('Prediction completed successfully')
        return predictions
        