from semantic.unet.dataset import Carla

import keras
import math
import os

from focal_loss import SparseCategoricalFocalLoss
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from keras.losses import SparseCategoricalCrossentropy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def train_unet(
        model,
        epochs : int,
        batch_size : int,
        img_size : Tuple[int, int] = (128,128),
        test_size : float = 0.25,
        dataset_folder : str = "./dataset",
        checkpoint_directory : str = "./checkpoints/",
        load_from_checkpoint : Optional[str] = None
    ):

    if load_from_checkpoint is not None:
        model.load_weights(load_from_checkpoint)
    
    rgb_folder = f"{dataset_folder}/rgb"
    rgb_paths = sorted(
        [
            os.path.join(rgb_folder, fname)
            for fname in os.listdir(rgb_folder)
            if fname.endswith(".png")
        ]
    )

    label_folder = f"{dataset_folder}/semantic"
    label_paths = sorted(
        [
            os.path.join(label_folder, fname)
            for fname in os.listdir(label_folder)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    train_rgb_paths, validation_rgb_paths, train_label_paths, validation_label_paths = train_test_split(
        rgb_paths, label_paths, test_size=test_size
    )

    training_generator = Carla(batch_size, img_size, train_rgb_paths, train_label_paths)
    validation_generator = Carla(batch_size, img_size, validation_rgb_paths, validation_label_paths, data_augmentation=False)

    # Below is a weight calculation built around:
    # (# of pixels in average in a category per image) /
    # (# of pixels in average for the given category per image)
    class_weight=[
            0.2314920504970292,
            64.11203165414611,
            0.4338712221910821,
            9.73727668152528,
            2.421319361944825,
            2.8451573153682137,
            2.0520724385563724,
            189.19925153003993,
        ]

    # We use a sigmoid to compress our given class weights to a
    # 0-1 range. We multiply this by 2 to increase our range to
    # 0-2.
    class_weight = [(2*sigmoid(x)) for x in class_weight]

    model.compile(optimizer='rmsprop',
    loss=SparseCategoricalFocalLoss(
        2.0,
        class_weight=class_weight
    ))

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{checkpoint_directory}/unet.h5", save_best_only=True),
        keras.callbacks.BackupAndRestore(backup_dir=f"{checkpoint_directory}/"),
        keras.callbacks.TensorBoard(log_dir="./log_dir", histogram_freq=1)
    ]

    model.fit(training_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)

    return model