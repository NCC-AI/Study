
import numpy as np
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.backend.tensorflow_backend import set_session


# Parameter
num_classes = 10
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# Model
class ConfidenceEstimationModel(Model):
    """Model which collects updates from loss_func.updates"""
    @property
    def updates(self):
        updates = super().updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates
        return updates

VGG13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
input_shape = (32, 32, 3)
input_tensor = Input(shape=input_shape)
net = input_tensor

for x in VGG13:
    if x == 'M':
        net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(net)
    else:
        net = Conv2D(x, (3, 3), padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        
net = AveragePooling2D(pool_size=(1, 1), strides=(1,1), padding='same')(net)
net = Flatten()(net)
classifier = Dense(num_classes, activation='softmax')(net)
confidence = Dense(1, activation='sigmoid')(net)
prediction = keras.layers.concatenate([classifier, confidence], axis=1)
model = ConfidenceEstimationModel(inputs=input_tensor, outputs=prediction)


# Dataset
def preprocess(images, length, train=True):
    ## Normalize
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    images = ((images - mean) / std)
    ## Cutout
    if train:
        if np.random.choice([0, 1]):
            h, w = images[0].shape[0], images[0].shape[1]
            mask = np.ones((h, w), dtype='float32')
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = int(np.clip(y - length / 2, 0, h)), int(np.clip(y + length / 2, 0, h))
            x1, x2 = int(np.clip(x - length / 2, 0, w)), int(np.clip(x + length / 2, 0, w))
            mask[y1:y2, x1:x2] = 0.
            mask = mask[:, :, np.newaxis]
            mask = np.tile(mask, 3)
    return images * mask

def add_label(labels):
    conf_label = np.ones((labels.shape[0], 1))
    new_labels = np.hstack((labels, conf_label))
    return new_labels

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = preprocess(x_train, 16, True)
x_test = preprocess(x_test, 16, False)
y_train = add_label(to_categorical(y_train, num_classes=num_classes))
y_test = add_label(to_categorical(y_test, num_classes=num_classes))


# Loss
class confidence_estimation_loss(object):

    def __init__(self, lmbda=0.1):
        self.lmbda = tf.Variable(lmbda)
        self.updates = []

    def __call__(self, y_true, y_pred):

        # Separate output
        prediction = tf.slice(y_pred, [0, 0],           [y_pred.shape[0], num_classes])
        confidence = tf.slice(y_pred, [0, num_classes], [y_pred.shape[0], 1])

        # Clip output value
        eps = 1e-12
        pred_original = tf.clip_by_value(prediction, 0. + eps, 1. - eps)
        confidence = tf.clip_by_value(confidence, 0. + eps, 1. - eps)

        # Randomly set half of the confidences to 1 (i.e. no hints)
        means = tf.constant([.5])
        b = tf.where(
            tf.random_uniform([tf.shape(confidence)[0], 1], minval=0, maxval=1) - means < 0,
            tf.ones([tf.shape(confidence)[0], 1]),
            tf.zeros([tf.shape(confidence)[0], 1]))
        conf = tf.add(confidence * b, 1.0 - b)
        conf = tf.tile(conf, [1, 10])

        # Modify predictions
        pred_new = tf.add(pred_original * conf, y_true[:, :-1] * (1 - conf))
        pred_new = tf.log(pred_new)

        # Calculate loss
        xentropy_loss = tf.reduce_mean(-tf.reduce_sum(y_true[:, :-1] * pred_new, reduction_indices=[1]))
        confidence_loss = tf.reduce_mean(-tf.log(confidence))
        total_loss = tf.add(xentropy_loss, (self.lmbda * confidence_loss))

        # Update lambda
        lm_val = tf.cond(0.3 > confidence_loss, lambda: 1.01, lambda: 0.99)
        new_lmbda = tf.divide(self.lmbda, lm_val)
        self.updates.append(tf.assign(self.lmbda, new_lmbda))

        return total_loss


# Callback
class PlotHistograms(keras.callbacks.Callback):

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.classify = []
        self.targets = []
        self.conf = []

    def on_epoch_end(self, epoch, logs={}):
        x_data, y_data = self.dataset
        outputs = self.model.predict(x_data, verbose=1)
        
        self.classify.append(outputs[:, :-1])
        self.targets.append(y_data[:, :-1])
        self.conf.append(outputs[:, -1])
        
        pred_label = np.array([x.argmax() for x in self.classify[epoch]])
        true_label = np.array([x.argmax() for x in self.targets[epoch]])
        corr = pred_label == true_label
        conf = np.array([x for x in self.conf[epoch]])
        
        plt.figure(figsize=(6, 4))
        sns.distplot(conf[corr], kde=False, bins=50, norm_hist=True, label='Correct')
        sns.distplot(conf[np.invert(corr)], kde=False, bins=50, norm_hist=True, label='Incorrect')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('logs/%03d.png' % epoch)
        plt.close()

# Fit
def schedule(epoch, learning_rate=0.1, decay=0.2):
    if epoch in [60, 120, 160]:
        return learning_rate * decay
    else:
        return learning_rate

callbacks = [LearningRateScheduler(schedule), PlotHistograms((x_test, y_test))]
optimizer = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
loss = confidence_estimation_loss()
model.compile(optimizer=optimizer, loss=loss)
model.fit(x_train, y_train, batch_size=16, epochs=200, verbose=1, callbacks=callbacks)
model.save('logs/confidence_estimation.h5')
