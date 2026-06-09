# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

from what.utils.proj import proj_lp

class PCBAttack:
    def __init__(self, model, attack_type, classes, init="zero", learning_rate = 4 / 255.0, batch = 1, decay = 0.98, custom_objects = None):
        self.classes = len(classes)
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()
        self.use_filter = False

        if init == "uniform":
            self.noise = np.random.uniform( -2 / 255.0, 2 / 255.0, size=(416, 416, 3))
        else:
            self.noise = np.zeros((416, 416, 3))

        self.adv_patch_boxes = []
        self.fixed = True

        self.model = load_model(model, custom_objects=custom_objects)
        self.model.summary()
        self.attack_type = attack_type

        self.lr = learning_rate
        self.delta = 0
        self.decay = decay

        self.current_batch = 0
        self.batch = batch

        loss = 0

        for out in self.model.output:
            out = K.reshape(out, (-1, 5 + self.classes))

            # Targeted One Box
            if attack_type == "one_targeted":
                loss = K.max(K.sigmoid(out[:, 4]) * K.sigmoid(out[:, 5]))

            # Targeted Multi boxes
            if attack_type == "multi_targeted":
                loss = K.sigmoid(out[:, 4]) * K.sigmoid(out[:, 5])

            # Untargeted Multi boxes
            if attack_type == "multi_untargeted":
                for i in range(0, self.classes):
                    # PC Attack
                    loss = loss + tf.reduce_sum( K.sigmoid(out[:, 4]) * K.sigmoid(out[:, i+5]))

                    # Small centric
                    # loss = loss + tf.reduce_sum( K.sigmoid(out[:, 4]) * K.sigmoid(out[:, i+5]) / K.pow(K.sigmoid(out[:, 2]) * K.sigmoid(out[:, 3]), 2) )

                    # Large overlapped boxes
                    # loss = loss + tf.reduce_sum( K.sigmoid(out[:, 4]) * K.sigmoid(out[:, i+5])) / tf.reduce_sum(K.pow(K.sigmoid(out[:, 2]) * K.sigmoid(out[:, 3]), 2))

                # Small distributed boxes (PCB Attack)
                loss = loss / tf.reduce_sum(K.pow(K.sigmoid(out[:, 2]) * K.sigmoid(out[:, 3]), 2))

        grads = K.gradients(loss, self.model.input)
        self.delta = self.delta + K.sign(grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

    def attack(self, input_cv_image):
        with self.graph.as_default():
            # Draw each adversarial patch on the input image
            input_cv_image = input_cv_image + self.noise
            input_cv_image = np.clip(input_cv_image, 0.0, 1.0).astype(np.float32)

            if not self.fixed:
                outputs, grads = self.sess.run([self.model.output, self.delta], feed_dict={self.model.input:np.array([input_cv_image])})

                self.noise = self.noise + self.lr * grads[0, :, :, :]

                self.current_batch = self.current_batch + 1

                if self.current_batch == self.batch:
                    self.lr = self.lr * self.decay
                    self.current_batch = 0

                self.noise = np.clip(self.noise, -1.0, 1.0)

                self.noise = proj_lp(self.noise, xi=8/255.0, p = np.inf)
            else:
                outputs = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})

            return input_cv_image, outputs
