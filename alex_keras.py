import tensorflow as tf
import keras.backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from keras.datasets import mnist
from keras.utils import np_utils
#fashion_mnist = keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255


number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
#create model
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)
model = Sequential()
#add model layers

#Conv 1
model.add(Conv2D(filters=96, input_shape=(28,28,1), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

#Conv 2
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
# Batch Normalisation
model.add(BatchNormalization())

#Conv 3
model.add(Conv2D(filters=384,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(3,3),  strides= (2,2),padding='same'))
#Normalization
model.add(BatchNormalization(axis=-1))

#Conv4
# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2),  strides= (2,2),padding='same'))
#Normalization
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(4096,input_shape=(28*28*1,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(10,activation='softmax'))


model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=3,batch_size=64)
# model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
# validation_split=0.2, shuffle=True)
