
#Code Reference from https://github.com/pskrunner14/resnet-classifier
# https://github.com/SakhriHoussem/RestNet
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import keras.backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from sklearn.model_selection import train_test_split
# from resnet_utils import load_dataset, convert_to_one_hot
#from utils import split, one_hot_encoding
from keras.models import model_from_json

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

batch_size = 16
epochs = 1
classes = 17
input_shape = (224,224,3)


def preprocess():
    # # Splitting dataset in train,test and validate
    # x_train,y_train,x_test,y_test,valid_set_x,valid_set_y = split(training_per=0.8,test_per=0.1,validation_per=0.1)

    # x_train = x_train.reshape(x_train.shape[0],32,32,1)
    # x_test = x_test.reshape(x_test.shape[0],32,32,1)
    # valid_set_x = valid_set_x.reshape(valid_set_x.shape[0],32,32,1)
    # # print(x_train.shape)
    # # print(x_test.shape)
    # # print(valid_set_x.shape)

    # hot_enc = one_hot_encoding()
    # y_train = [hot_enc[int(x)] for x in y_train]
    # y_train = np.asarray(y_train)
    # y_train = y_train.reshape(x_train.shape[0], 10)

    # y_test = [hot_enc[int(x)] for x in y_test]
    # y_test = np.asarray(y_test)
    # y_test = y_test.reshape(x_test.shape[0], 10)

    # y_val = [hot_enc[int(x)] for x in valid_set_y]
    # y_val = np.asarray(y_val)
    # valid_set_y = y_val.reshape(valid_set_x.shape[0], 10)

    import tflearn.datasets.oxflower17 as oxflower17
    X, Y= oxflower17.load_data(one_hot=True)


    x_train, x_test_pre, y_train, y_test_pre = train_test_split(X, Y, test_size=0.20, random_state=42)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)

    return x_train,y_train,x_test, y_test, x_validation, y_validation

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape, classes):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f = 3, filters =  [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2),padding='same',data_format='channels_last')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

def train():
    #x_train,y_train,x_test,y_test,valid_set_x,valid_set_y = preprocess()
    
    resnet50 = ResNet50(input_shape, classes)

    resnet50.summary()

    checkpointer = ModelCheckpoint(filepath="res_model_weight.h5", verbose=0, save_best_only=True)  

    resnet50.compile(
                    optimizer='adam', 
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )

    resnet50.fit(
                x_train, 
                y_train, 
                validation_data=(valid_set_x,valid_set_y),
                epochs=epochs, 
                batch_size=batch_size, 
                verbose = 1,
                callbacks=[checkpointer]
                )

    preds = resnet50.evaluate(
                        x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose  = 1
                        )

    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    # serialize model to JSON
    model_json = resnet50.to_json()
    with open("res_model.json", "w+") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    resnet50.save_weights("res_model_weight.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    print('Inside main:')
    #preprocess()
    print('Preprocesssing Completed - Training Started')
    train()
