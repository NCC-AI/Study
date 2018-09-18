import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model, Sequential
from keras.layers import Activation, Dense,Flatten, Dropout, Input, Concatenate
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import keras
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# ----------------------------------------------------------------------------------------------------
# GPU config
# ----------------------------------------------------------------------------------------------------
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

# ----------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------
batch_size =16 # ネットワークを軽くするか、バッチサイズを減らすしかないらしい
epochs = 200
seed = 0
learning_rate = 0.1
data_augmentation = False
cutout = 16
budget = 0.3

np.random.seed(seed)
tf.set_random_seed(seed)

# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------
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
num_classes = 10
name = 1

input_tensor = Input(shape=input_shape, name='inputs')
net = input_tensor

for x in VGG13:
    if x == 'M':
        net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool'+str(name))(net)
        name += 1
    else:
        net = Conv2D(x, (3, 3), padding='same', name='conv'+str(name))(net)
        net = BatchNormalization(name='norm'+str(name))(net)
        net = Activation('relu', name='acti'+str(name))(net)
        name += 1
        
net = AveragePooling2D(pool_size=(1, 1), strides=(1,1), padding='same', name='avepool')(net)
net = Flatten(name='flatten')(net)
classifier = Dense(num_classes, activation='softmax', name='classifier')(net)
confidence = Dense(1, activation='sigmoid',name='confidence')(net)
prediction = keras.layers.concatenate([classifier, confidence], axis=1, name='prediction')

model = ConfidenceEstimationModel(input_tensor, prediction, name='confidence_estimator')

# ----------------------------------------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------------------------------------
def preprocess(images, length, train=True):
    
    # Normalize
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    images = ((images - mean) / std)
    
    # Cutout
    if train:
        if np.random.choice([0, 1]):
            h = images[0].shape[0]
            w = images[0].shape[1]
            mask = np.ones((h, w), dtype='float32')
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = int(np.clip(y - length / 2, 0, h))
            y2 = int(np.clip(y + length / 2, 0, h))
            x1 = int(np.clip(x - length / 2, 0, w))
            x2 = int(np.clip(x + length / 2, 0, w))
            mask[y1: y2, x1: x2] = 0.
            mask = mask[:, :, np.newaxis]
            mask = np.tile(mask, 3)
            images = images * mask
        
    return images

def add_label(labels):
    conf_label = np.ones((labels.shape[0], 1))
    new_labels = np.hstack((labels, conf_label))
    
    return new_labels

# ----------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

x_train = preprocess(x_train, cutout, True)
x_test = preprocess(x_test, cutout, False)
y_train = add_label(keras.utils.to_categorical(y_train, num_classes=num_classes))
y_test = add_label(keras.utils.to_categorical(y_test, num_classes=num_classes))

# ----------------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------------
class confidence_estimation_loss(object):
    __name__ = 'confidence_estimation_loss'
    
    def __init__(self, lmbda=0.1):
        self.lmbda = tf.Variable(lmbda)
        self.updates = []

    def __call__(self, y_true, y_pred):

        # 変数定義
        # global lmbda

        # 順伝播の出力
        prediction = tf.slice(y_pred, [0, 0], [batch_size, num_classes])
        confidence = tf.slice(y_pred, [0, num_classes], [batch_size, 1])

        # clipメソッドでインプットを範囲内に収める
        eps = 1e-12
        pred_original = tf.clip_by_value(prediction, 0. + eps, 1. - eps)
        confidence = tf.clip_by_value(confidence, 0. + eps, 1. - eps)

        # 予測値の補正を行う（ヒント部分）
        # Randomly set half of the confidences to 1 (i.e. no hints)
        means = tf.constant([.5])
        b = tf.where(tf.random_uniform([tf.shape(confidence)[0], 1], minval=0, maxval=1) - means < 0,  
                     tf.ones([tf.shape(confidence)[0], 1]), 
                     tf.zeros([tf.shape(confidence)[0], 1]))

        # confを設定
        conf = tf.add(confidence * b, 1.0 - b)
        conf = tf.tile(conf, [1, 10])

        # 予測を小さくして、正解ラベルの分布を足す
        pred_new = tf.add(pred_original * conf, y_true[:, :-1] * (1 - conf))
        pred_new = tf.log(pred_new)

        # 損失計算
        xentropy_loss = tf.reduce_mean(-tf.reduce_sum(y_true[:, :-1] * pred_new, reduction_indices=[1]))
        confidence_loss = tf.reduce_mean(-tf.log(confidence))

        # 損失を定義
        total_loss = tf.add(xentropy_loss, (self.lmbda * confidence_loss))

        # lambdaを更新
        lm_val = tf.cond(budget > confidence_loss, lambda: 1.01, lambda: 0.99)
        new_lmbda = tf.divide(self.lmbda, lm_val)
        self.updates.append(tf.assign(self.lmbda, new_lmbda))

        return total_loss

# ----------------------------------------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------------------------------------
def plot_histograms(corr, conf, bins=50, norm_hist=True):
    plt.figure(figsize=(6, 4))
    sns.distplot(conf[corr], kde=False, bins=bins, norm_hist=norm_hist, label='Correct')
    sns.distplot(conf[np.invert(corr)], kde=False, bins=bins, norm_hist=norm_hist, label='Incorrect')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()

# ----------------------------------------------------------------------------------------------------
# Custom callback
# ----------------------------------------------------------------------------------------------------
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
        # conf = outputs[:, -1]
        conf = np.array([x for x in self.conf[epoch]])
        
        bins=50
        norm_hist=True
        plt.figure(figsize=(6, 4))
        sns.distplot(conf[corr], kde=False, bins=bins, norm_hist=norm_hist, label='Correct')
        sns.distplot(conf[np.invert(corr)], kde=False, bins=bins, norm_hist=norm_hist, label='Incorrect')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig( 'logs/%03d.png' % epoch )
        plt.close()

# ----------------------------------------------------------------------------------------------------
# Compile
# ----------------------------------------------------------------------------------------------------
optim = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=5e-4, nesterov=True)

def schedule(epoch, decay=0.2):
    if epoch in [60, 120, 160]:
        return learning_rate * decay
    else:
        return learning_rate

cbk = PlotHistograms((x_test, y_test))
callbacks = [keras.callbacks.LearningRateScheduler(schedule), cbk]
loss = confidence_estimation_loss(lmbda=0.1)

model.compile(optimizer=optim, loss=loss)

# ----------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    verbose=1, 
                    callbacks=callbacks)

model.save('logs/confidence_estimation.h5')
