import tensorflow as tf
import keras.backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split

import tflearn.datasets.oxflower17 as oxflower17
X, Y= oxflower17.load_data(one_hot=True)


x_train, x_test_pre, y_train, y_test_pre = train_test_split(X, Y, test_size=0.20, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)



import time as time 
from tensorflow.keras.callbacks import TensorBoard

NAME = "vgg"
np.random.seed(1000)
model = Sequential()


#Conv 1
model.add(Conv2D(filters=64, input_shape=(224,224,3), kernel_size=(3,3),\
 strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, input_shape=(224,224,3), kernel_size=(3,3),\
 strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))



#Conv 2
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


#Conv 3
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Conv4
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Conv5
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(17,activation='softmax'))
bCallBack = keras.callbacks.TensorBoard(log_dir='logs/{}', histogram_freq=0, write_graph=True, write_images=True)
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.summary()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=3,batch_size=2,callbacks = [tensorboard])