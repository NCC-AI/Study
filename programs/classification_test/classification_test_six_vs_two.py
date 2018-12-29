import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from keras.callbacks import EarlyStopping
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


#############
# load data #
#############
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

######################
# six classification #
######################
class_six = [0, 2, 3,   #clothes  = 0
            5, 7, 9]    #shoes    = 1
"""
    ['T-shirt/top',
    'Pullover',
    'Dress',

    'Sandal',
    'Sneaker',
    'Ankle boot']
"""

def six_classification(x_train, y_train, x_test, y_test, class_six):
    train_index_range = np.array([index for index, content in enumerate(y_train) if content in class_six])
    y_train_six = y_train[train_index_range]
    x_train_six = x_train[train_index_range]

    test_index_range = np.array([index for index, content in enumerate(y_test) if content in class_six])
    y_test_six = y_test[test_index_range]
    x_test_six = x_test[test_index_range]

    convert_dict = { 0: 0, 2: 1, 3: 2, 5: 3, 7: 4, 9: 5}
    y_train_six = np.array([convert_dict[i] for i in y_train_six])
    y_test_six = np.array([convert_dict[i] for i in y_test_six])

    plt.figure(1)
    for i,num in enumerate([60000,6000,600,60]):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.hist(y_train_six[:num],bins=6)
        plt.title('data ='+str(num))

    x_train_six=x_train_six.reshape(x_train_six.shape[0],28,28,1)
    x_test_six=x_test_six.reshape(x_test_six.shape[0],28,28,1)

    x_train_six = x_train_six/255
    x_test_six = x_test_six/255

    y_train_six = to_categorical(y_train_six,6)
    y_test_six = to_categorical(y_test_six,6)

    return x_train_six,y_train_six,x_test_six,y_test_six

######################
# two classification #
######################
def two_classification(y_train, y_test, class_six):
    y_train_six_to_two = np.array([0 if content<4 else 1 for index, content in enumerate(y_train) if content in class_six])
    y_test_six_to_two = np.array([0 if content<4 else 1 for index, content in enumerate(y_test) if content in class_six])

    plt.figure(2)
    for i,num in enumerate([60000,6000,600,60]):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.hist(y_train_six_to_two[:num],bins=2)
        plt.title('data ='+str(num))

    return y_train_six_to_two, y_test_six_to_two

###############
# prepare data#
###############
x_train_six, y_train_six, x_test_six, y_test_six = six_classification(x_train, y_train, x_test, y_test, class_six)
y_train_six_to_two, y_test_six_to_two = two_classification(y_train, y_test, class_six)

x_train_L = x_train_six
x_train_M = x_train_six[:6000]
x_train_S = x_train_six[:600]
x_train_SS = x_train_six[:60]

y_train_L = y_train_six
y_train_M = y_train_six[:6000]
y_train_S = y_train_six[:600]
y_train_SS = y_train_six[:60]

y_train_L_two = y_train_six_to_two
y_train_M_two = y_train_six_to_two[:6000]
y_train_S_two = y_train_six_to_two[:600]
y_train_SS_two = y_train_six_to_two[:60]

##############
# make model #
##############
model_six = Sequential()
model_six.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model_six.add(MaxPooling2D(2,2))
model_six.add(Conv2D(64,(3,3), activation='relu'))
model_six.add(MaxPooling2D(2,2))
model_six.add(Conv2D(64,(3,3), activation='relu'))
model_six.add(Flatten())
model_six.add(Dense(64,activation='relu'))
model_six.add(Dense(6,activation='softmax'))

early_stopping = EarlyStopping(patience=10, verbose=-1)

model_six.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model_two = Sequential()
model_two.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model_two.add(MaxPooling2D(2,2))
model_two.add(Conv2D(64,(3,3), activation='relu'))
model_two.add(MaxPooling2D(2,2))
model_two.add(Conv2D(64,(3,3), activation='relu'))
model_two.add(Flatten())
model_two.add(Dense(64,activation='relu'))
model_two.add(Dense(1,activation='sigmoid'))

model_two.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

##########
# result #
##########
model_six.fit(x_train_L, y_train_L, batch_size=256, epochs=50, verbose=2, validation_split=0.2, callbacks=[early_stopping])
score_L = model_six.evaluate(x_test_six,y_test_six)

model_two.fit(x_train_L, y_train_L_two, batch_size=256, epochs=50, verbose=2, validation_split=0.2, callbacks=[early_stopping])
score_L_two = model_two.evaluate(x_test_six,y_test_six_to_two)
print('i=60000')
print('Test loss: Six {}, Two {}'.format(score_L[0],score_L_two[0]))
print('Test accuracy: Six {}, Two {}'.format(score_L[1],score_L_two[1]))


model_six.fit(x_train_M, y_train_M, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_M = model_six.evaluate(x_test_six,y_test_six)

model_two.fit(x_train_M, y_train_M_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_M_two = model_two.evaluate(x_test_six,y_test_six_to_two)
print('i=6000')
print('Test loss: Six {}, Two {}'.format(score_M[0],score_M_two[0]))
print('Test accuracy: Six {}, Two {}'.format(score_M[1],score_M_two[1]))


model_six.fit(x_train_S, y_train_S, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_S = model_six.evaluate(x_test_six,y_test_six)

model_two.fit(x_train_S, y_train_S_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_S_two = model_two.evaluate(x_test_six,y_test_six_to_two)
print('i=600')
print('Test loss: Six {}, Two {}'.format(score_S[0],score_S_two[0]))
print('Test accuracy: Six {}, Two {}'.format(score_S[1],score_S_two[1]))


model_six.fit(x_train_SS, y_train_SS, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_SS = model_six.evaluate(x_test_six,y_test_six)

model_two.fit(x_train_SS, y_train_SS_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_SS_two = model_two.evaluate(x_test_six,y_test_six_to_two)
print('i=60')
print('Test loss: Six {}, Two {}'.format(score_SS[0],score_SS_two[0]))
print('Test accuracy: Six {}, Two {}'.format(score_SS[1],score_SS_two[1]))
