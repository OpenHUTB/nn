#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from keras.callbacks import Callback, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K


# ## Loading Data
# Data is loaded from images in the folder *"dataset/"*. There are two classes: *"true"* specifies images with the pedestrian using a smartphone, while *"false"* specifies images with the pedestrian not holding a smartphone. The images are loaded, shuffled and batched.

# In[5]:


BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)

IMAGE_DIR = "dataset/"
MODEL_NAME = "smato_mobilenet_v2m"
SAVE_DIR = f"saved_models/{MODEL_NAME}/"


# In[6]:


dataset = tf.keras.utils.image_dataset_from_directory(IMAGE_DIR,
                                                      shuffle = False,
                                                      seed = 141,
                                                      batch_size = None,
                                                      image_size = IMAGE_SIZE)
CLASS_NAMES = dataset.class_names
dataset_size = len(dataset)
dataset = dataset.shuffle(buffer_size = dataset_size, seed = 141, reshuffle_each_iteration = False)
# count images in each category
counts = {
    1: 0,
    0: 0
}
for i in dataset:
    _, cls = i
    counts[int(cls)] += 1
print("Number of pedestrians with a smartphone: ", counts[1])
print("Number of pedestrians without a smartphone: ", counts[0])

# batch the dataset
dataset = dataset.batch(batch_size = BATCH_SIZE)
print("Classes: ", CLASS_NAMES)
print("Total number of batches created: ", len(dataset))


# ## Splitting Dataset
# The dataset is split into training, validation, and test set. 85% of the dataset is reserved for training, while the rest is divided into validation and test dataset in the ratio of 60:40.

# In[7]:


# train-validation split
num_batches = len(dataset)
val_dataset = dataset.take(int(num_batches * 0.15))
train_dataset = dataset.skip(int(num_batches * 0.15))

# validation-test split
num_val_batches = len(val_dataset)
test_dataset = val_dataset.take(int(num_val_batches * 0.4))
validation_dataset = val_dataset.skip(int(num_val_batches * 0.4))


# In[8]:


print("Number of batches in Training Dataset: ", len(train_dataset))
print("Number of batches in Validation Dataset: ", len(validation_dataset))
print("Number of batches in Test Dataset: ", len(test_dataset))


# In[9]:


# rescale for tf hub - models on tf-hub require pixel values b/w 0 and 1
normalize = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalize(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalize(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalize(x), y))


# In[10]:


# buffered prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size = AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)


# ## Model Definition
# A model is defined using MobileNet-V2 as the feature extractor, followed by three fully connected hidden layers. Batch Normalization is performed after each FC layer, and dropout is used to reduce overfitting.

# In[7]:


mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(
    mobilenet_v2,
    trainable = False)

model = tf.keras.Sequential([
    
    tf.keras.layers.InputLayer(IMAGE_SIZE + (3, )),
    feature_extractor_layer,

    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.6),
    
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(1, activation = "sigmoid")
], name = MODEL_NAME)

model.summary()


# ## Metric for choosing best model: F1 Score

# In[11]:


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall)/(precision + recall + K.epsilon())
    return f1_val


# ## Model Training
# While training the model, only the best model is saved. This is decided by the metric accuracy on the validation set

# In[9]:


model.compile(
  optimizer = tf.keras.optimizers.Adam(),
  loss = tf.keras.losses.BinaryCrossentropy(),
  metrics = ['acc', f1]
)


# In[10]:


NUM_EPOCHS = 50

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = SAVE_DIR,
    save_weights_only = False,
    monitor = 'val_f1',
    mode = 'max',
    save_best_only = True
)

history = model.fit(train_dataset,
                    validation_data = validation_dataset,
                    epochs = NUM_EPOCHS,
                    callbacks = [model_checkpoint_callback])


# ## Model Evaluation
# The best trained model is loaded and the test dataset is used for evaluating performance on data never been used in this process.

# In[12]:


model = keras.models.load_model(SAVE_DIR, custom_objects = {"f1": f1})
model.evaluate(test_dataset)


# ## Example Images
# A bunch of images from the test dataset are chosen to show the performance of the model and to understand the reasons for misclassification.

# In[13]:


# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.shuffle(10, seed = None).take(1).as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch[: 32]).flatten()

predictions = np.where(predictions < 0.5, 0, 1)
label_batch = label_batch[: 32]

print('Predictions:\n', predictions)
print('Labels:\n', label_batch)

plt.figure(figsize=(12, 8))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(image_batch[i])
    plt.title(str(CLASS_NAMES[predictions[i]]) + ", " + str(label_batch[i]))
    plt.axis("off")


# In[ ]:




