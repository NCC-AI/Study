from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.utils import plot_model
from keras.losses import categorical_crossentropy, mean_squared_error

import matplotlib.pyplot as plt

import numpy as np

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
num_labeled_images = 100
features_dim = 4096
num_classes = 10
noise_dim = 100
latent_dim = noise_dim
batch_size=100
steps_per_epoch = (60000 - num_labeled_images) // batch_size
epochs = 100


optimizer = Adam(0.001, 0.5)

# inputs
input_noise = Input(shape=(latent_dim,))

# hidden layer
g = Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim)(input_noise)
g = Reshape((7, 7, 128))(g)
g = BatchNormalization(momentum=0.8)(g)
g = UpSampling2D()(g)
g = Conv2D(128, kernel_size=3, padding="same")(g)
g = Activation("relu")(g)
g = BatchNormalization(momentum=0.8)(g)
g = UpSampling2D()(g)
g = Conv2D(64, kernel_size=3, padding="same")(g)
g = Activation("relu")(g)
g = BatchNormalization(momentum=0.8)(g)
g = Conv2D(1, kernel_size=3, padding="same")(g)

# outputs
g_image = Activation("tanh")(g)

generator = Model(input_noise, g_image)

# inputs
input_image = Input(shape=img_shape)

d = Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(input_image)
d = LeakyReLU(alpha=0.2)(d)
d = Dropout(0.25)(d)
d = Conv2D(64, kernel_size=3, strides=2, padding="same")(d)
d = ZeroPadding2D(padding=((0,1),(0,1)))(d)
d = LeakyReLU(alpha=0.2)(d)
d = Dropout(0.25)(d)
d = BatchNormalization(momentum=0.8)(d)
d = Conv2D(128, kernel_size=3, strides=2, padding="same")(d)
d = LeakyReLU(alpha=0.2)(d)
d = Dropout(0.25)(d)
d = BatchNormalization(momentum=0.8)(d)
d = Conv2D(256, kernel_size=3, strides=1, padding="same")(d)
d = LeakyReLU(alpha=0.2)(d)
d = Dropout(0.25)(d)
features = Flatten(name='features_output')(d)

label = Dense(num_classes, name='y_output')(features)
# there is no activation here

discriminator = Model(input_image, [features, label])

discriminator.trainable = False

validity = discriminator(g_image)
combined = Model(input_noise, validity)

##############
#  Loss functions #
##############

def softmax_cross_entropy(y_true, y_output):
    y_pred = K.softmax(y_output)
    loss =categorical_crossentropy(y_true, y_pred)
    return loss

def discriminate_real(y_output, batch_size=batch_size):
    # logD(x) = logZ(x) - log(Z(x) + 1)  where Z(x) = sum_{k=1}^K exp(l_k(x))
    log_zx = K.logsumexp(y_output, axis=1)
    log_dx = log_zx - K.softplus(log_zx)
    dx = K.sum(K.exp(log_dx)) / batch_size
    loss = -K.sum(log_dx) / batch_size
    return loss, dx
    
def discriminate_fake(y_output, batch_size=batch_size):
    # log{1 - D(x)} = log1 - log(Z(x) + 1)
    log_zx_g = K.logsumexp(y_output, axis=1)
    loss = K.sum(K.softplus(log_zx_g)) / batch_size
    return loss

#################
#  Discriminator Loss #
#################

def labeled_loss(y_true, y_output):
    class_loss = softmax_cross_entropy(y_true, y_output)
    _,dx = discriminate_real(y_output, batch_size=batch_size)
    return class_loss

def unlabeled_loss(g_label, y_output, batch_size=batch_size):    
    loss_real,dx = discriminate_real(y_output, batch_size=batch_size)
    loss_fake = discriminate_fake(g_label, batch_size=batch_size)
    return loss_real + loss_fake
    
###############
#  Generator Loss #
###############

def feature_matching(features_true, features_fake):
    return mean_squared_error(features_true, features_fake)

def generator_loss(_, y_output):
    loss_real,dx = discriminate_real(y_output, batch_size=batch_size)
    return loss_real

# Load the dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_samples = 10
x_labeled = []
y_labeled = []
x_unlabeled = []

for class_index in range(10):
    label_index = np.where(y_train == class_index)
    class_input_data = x_train[label_index]
    
    # labeled data
    x_labeled.append(class_input_data[:num_samples])
    y_labeled.append(np.full(num_samples, class_index, int))
    
    # unlabeled data
    x_unlabeled.append(class_input_data[num_samples:])
    
x_labeled = np.concatenate(x_labeled, axis=0)
x_unlabeled = np.concatenate(x_unlabeled, axis=0)
x_labeled = x_labeled.astype('float32') / 255
x_unlabeled = x_unlabeled.astype('float32') / 255

x_labeled = x_labeled.reshape(x_labeled.shape+(1,))
x_unlabeled = x_unlabeled.reshape(x_unlabeled.shape+(1,))

y_labeled = np.concatenate(y_labeled, axis=0)
y_labeled_onehot = np.eye(num_classes)[y_labeled]


# test data
x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape+(1,))
y_test = np.eye(num_classes)[y_test]

print('labeled input_shape: {}, {}\nunlabeled input_shape: {}'.format(x_labeled.shape, y_labeled_onehot.shape, x_unlabeled.shape))
print('test input_shape: ', x_test.shape, y_test.shape)

# 教師なしの枚数が、教師ありと一致するようにリピート
labeled_index = []
for i in range(len(x_unlabeled) // len(x_labeled)):
    l = np.arange(len(x_labeled))
    np.random.shuffle(l)
    labeled_index.append(l)
    
labeled_index = np.concatenate(labeled_index)
unlabeled_index = np.arange(len(x_unlabeled))
print(labeled_index.shape, unlabeled_index.shape)

dummy_features = np.zeros((batch_size, features_dim))
dummy_label = np.zeros((batch_size, num_classes))

history = []

for epoch in range(epochs):
    print('epoch {}/{}'.format(epoch+1, epochs))
    
    np.random.shuffle(unlabeled_index)
    np.random.shuffle(labeled_index)
    
    for step in range(steps_per_epoch):
        print('step {}/{}'.format(step+1, steps_per_epoch))
        unlabel_index_range = unlabeled_index[step*batch_size:(step+1)*batch_size]
        label_index_range = labeled_index[step*batch_size:(step+1)*batch_size]
        
        images_l = x_labeled[label_index_range]
        label_l = y_labeled_onehot[label_index_range]
        images_u = x_unlabeled[unlabel_index_range]
        

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        #########
        # for label
        #########
        discriminator.compile(
            optimizer=optimizer,
            loss= labeled_loss,
            loss_weights={'features_output': 0., 'y_output': 1.},
            metrics = {'y_output': 'accuracy'})
        
        # Train the discriminator
        d_loss_label = discriminator.train_on_batch(images_l, [dummy_features, label_l])
        print('label_loss: {}, label_acc: {}'.format(d_loss_label[0], d_loss_label[3]))
        
        ############
        # for unlabeled
        ############
        discriminator.compile(
            optimizer=optimizer,
            loss= unlabeled_loss,
            loss_weights={'features_output': 0., 'y_output': 1.})
        
        z_batch = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)
        _, g_label = combined.predict(z_batch)
        
        # Train the discriminator
        d_loss_unlabel = discriminator.train_on_batch(images_u, [dummy_features, g_label])
        print('unlabel_loss : ', d_loss_unlabel[0])


        # ---------------------
        #  Train Generator
        # ---------------------
        
        combined.compile(
            optimizer=optimizer,
            loss= [feature_matching, generator_loss],
            loss_weights=[1, 1])
        
        # Train the generator
        z_batch = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)
        features_true, _ = discriminator.predict(images_l)
        g_loss = combined.train_on_batch(z_batch, [features_true, dummy_label])

        # Plot the progress
        print ('g_loss', g_loss)
    
        # validation
        discriminator.compile(
            optimizer=optimizer,
            loss= labeled_loss,
            loss_weights={'features_output': 0., 'y_output': 1.},
            metrics = {'y_output': 'accuracy'})
        
        test_eval = discriminator.evaluate(x_test, [np.zeros((10000, features_dim)), y_test])
        print('val_acc: ', test_eval[3])
        history.append(test_eval)


combined.save('combined.h5')
discriminator.save('discriminator.h5')

import json
with open('history.json', 'w') as fw:
    json.dump(fw, history)
