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


# 训练时使用的类别像素比例常量，按 SEMANTIC_CATEGORIES 索引顺序对应 8 类（来自
# 原始项目对 CARLA 采集数据集的统计）。其语义为：
#   CLASS_PIXEL_RATIOS[i] = (各类平均每张图像素数) / (第 i 类平均每张图像素数)
# 越大表示该类越少。Pedestrians ≈ 189 是最少的，Unlabeled ≈ 0.23 是最多的。
# 这两端的比例约 818:1，量化体现了 CARLA 街景的类别极度不平衡。
CLASS_PIXEL_RATIOS = [
    0.2314920504970292,    # 0 Unlabeled
    64.11203165414611,     # 1 Traffic Sign/Lights
    0.4338712221910821,    # 2 Roads
    9.73727668152528,      # 3 Road Lines
    2.421319361944825,     # 4 Sidewalk
    2.8451573153682137,    # 5 Ground
    2.0520724385563724,    # 6 Vehicles
    189.19925153003993,    # 7 Pedestrians
]


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

    # CLASS_PIXEL_RATIOS（见模块顶部）即 "(各类平均像素数) / (该类平均像素数)"，
    # 直接作为类别权重时数值跨度过大（约 818:1），训练不稳。下面用 sigmoid 压
    # 缩到 0-1，再乘 2 拉到 0-2 区间，将极端不平衡压成可训练的温和加权。
    class_weight = [(2 * sigmoid(x)) for x in CLASS_PIXEL_RATIOS]

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