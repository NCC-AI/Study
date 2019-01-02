import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from keras.callbacks import EarlyStopping
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#############
# load data #
#############
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

##################
# select pattern #
##################
pattern = int(input()) #0: clothes:shoes = 2:2, 1: clothes:shoes = 3:1, 2: clothes:shoes = 1:3
if pattern == 0:
    class_four = [0, 2,     #clothes = 0
                7, 9]       #shoes   = 1
    """
        ['T-shirt/top',
        'Pullover',

        'Sneaker',
        'Ankle boot']
    """
    convert_dict = { 0: 0, 2: 1, 7: 2, 9: 3}

elif pattern == 1:
    class_four = [0, 2, 3,  #clothes = 0
                9]          #shoes   = 1
    """
        ['T-shirt/top',
        'Pullover',
        'Dress',

        'Ankle boot']
    """
    convert_dict = { 0: 0, 2: 1, 3: 2, 9: 3}

elif pattern == 2:
    class_four = [0,        #clothes = 0
                5, 7, 9]    #shoes   = 1
    """
        ['T-shirt/top',

        'Sandal',
        'Sneaker',
        'Ankle boot']
    """
    convert_dict = { 0: 0, 5: 1, 7: 2, 9: 3}

#######################
# four classification #
#######################
def four_classification(x_train, y_train, x_test, y_test, class_four, convert_dict):
    train_index_range = np.array([index for index, content in enumerate(y_train) if content in class_four])
    y_train_four = y_train[train_index_range]
    x_train_four = x_train[train_index_range]

    test_index_range = np.array([index for index, content in enumerate(y_test) if content in class_four])
    y_test_four = y_test[test_index_range]
    x_test_four = x_test[test_index_range]

    y_train_four = np.array([convert_dict[i] for i in y_train_four])
    y_test_four = np.array([convert_dict[i] for i in y_test_four])

    plt.figure(1)
    for i,num in enumerate([40000,4000,400,40]):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.hist(y_train_four[:num],bins=4)
        plt.title('data ='+str(num))

    x_train_four=x_train_four.reshape(x_train_four.shape[0],28,28,1)
    x_test_four=x_test_four.reshape(x_test_four.shape[0],28,28,1)

    x_train_four = x_train_four/255
    x_test_four = x_test_four/255

    y_train_four = to_categorical(y_train_four,4)
    y_test_four = to_categorical(y_test_four,4)

    return x_train_four,y_train_four,x_test_four,y_test_four

######################
# two classification #
######################
def two_classification(y_train, y_test, class_four):
    y_train_four_to_two = np.array([0 if content<4 else 1 for index, content in enumerate(y_train) if content in class_four])
    y_test_four_to_two = np.array([0 if content<4 else 1 for index, content in enumerate(y_test) if content in class_four])

    plt.figure(2)
    for i,num in enumerate([40000,4000,400,40]):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.hist(y_train_four_to_two[:num],bins=2)
        plt.title('data ='+str(num))

    return y_train_four_to_two, y_test_four_to_two

###############
# prepare data#
###############
x_train_four, y_train_four, x_test_four, y_test_four = four_classification(x_train, y_train, x_test, y_test, class_four, convert_dict)
y_train_four_to_two, y_test_four_to_two = two_classification(y_train, y_test, class_four)

x_train_L = x_train_four
x_train_M = x_train_four[:4000]
x_train_S = x_train_four[:400]
x_train_SS = x_train_four[:40]

y_train_L = y_train_four
y_train_M = y_train_four[:4000]
y_train_S = y_train_four[:400]
y_train_SS = y_train_four[:40]

y_train_L_two = y_train_four_to_two
y_train_M_two = y_train_four_to_two[:4000]
y_train_S_two = y_train_four_to_two[:400]
y_train_SS_two = y_train_four_to_two[:40]

##############
# make model #
##############
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
early_stopping = EarlyStopping(patience=10, verbose=-1)

model_architect = model.to_json()

model.save_weights('model_four_reset.hdf5')
model.save_weights('model_two_reset.hdf5')

model_four = model_from_json(model_architect)
model_two = model_from_json(model_architect)

model_four.add(Dense(4,activation='softmax'))
model_two.add(Dense(1,activation='sigmoid'))

model_four.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_two.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

##########
# result #
##########
model_four.fit(x_train_L, y_train_L, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_L = model_four.evaluate(x_test_four,y_test_four)

model_two.fit(x_train_L, y_train_L_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_L_two = model_two.evaluate(x_test_four,y_test_four_to_two)
print('i=40000')
print('Test loss: Four {}, Two {}'.format(score_L[0],score_L_two[0]))
print('Test accuracy: Four {}, Two {}'.format(score_L[1],score_L_two[1]))


model_four.load_weights('model_four.hdf5')
model_two.load_weights('model_two.hdf5')

model_four.fit(x_train_M, y_train_M, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_M = model_four.evaluate(x_test_four,y_test_four)

model_two.fit(x_train_M, y_train_M_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_M_two = model_two.evaluate(x_test_four,y_test_four_to_two)
print('i=4000')
print('Test loss: Four {}, Two {}'.format(score_M[0],score_M_two[0]))
print('Test accuracy: Four {}, Two {}'.format(score_M[1],score_M_two[1]))


model_four.load_weights('model_four.hdf5')
model_two.load_weights('model_two.hdf5')

model_four.fit(x_train_S, y_train_S, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_S = model_four.evaluate(x_test_four,y_test_four)

model_two.fit(x_train_S, y_train_S_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_S_two = model_two.evaluate(x_test_four,y_test_four_to_two)
print('i=400')
print('Test loss: Four {}, Two {}'.format(score_S[0],score_S_two[0]))
print('Test accuracy: Four {}, Two {}'.format(score_S[1],score_S_two[1]))


model_four.load_weights('model_four.hdf5')
model_two.load_weights('model_two.hdf5')

model_four.fit(x_train_SS, y_train_SS, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_SS = model_four.evaluate(x_test_four,y_test_four)

model_two.fit(x_train_SS, y_train_SS_two, batch_size=256, epochs=50, verbose=0, validation_split=0.2, callbacks=[early_stopping])
score_SS_two = model_two.evaluate(x_test_four,y_test_four_to_two)
print('i=40')
print('Test loss: Four {}, Two {}'.format(score_SS[0],score_SS_two[0]))
print('Test accuracy: Four {}, Two {}'.format(score_SS[1],score_SS_two[1]))
