# -*- coding: UTF-8 -*-
'''
Created on 2016年12月30日

@author: frankzhan
'''
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image,ImageFilter
from pylab import*
from scipy.misc import imread, imsave

h=32
w=32
n_chan=3
input_shape=(n_chan, h, w)
n_filters=64
n_classes=8
conv_size=3 
pool_size = (2, 2)
batch_size=4 
n_epoch=1
dropout1=0.25 
model_name='segment_model'
path_model='/home/frankzhan/kerasmodel/{}'.format(model_name)
jsonfile_name='{}.json'.format(path_model)
weightsfile_name='{}.h5'.format(path_model)
testdir='/home/frankzhan/test/19result.jpg'

class KerasModel():

    def compile_model(self):
        print('start compiling model')
        model = Sequential()
        #1
        model.add(Convolution2D(n_filters, conv_size, conv_size,
                        border_mode='same',
                        input_shape=input_shape
                        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        #2
        model.add(Convolution2D(n_filters*4, conv_size, conv_size, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
#         #3
#         model.add(Convolution2D(n_filters/2, conv_size, conv_size, border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=pool_size))
#         #4
#         model.add(Convolution2D(n_filters, conv_size, conv_size, border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=pool_size))
#         #5
#         model.add(Convolution2D(n_filters, conv_size, conv_size, border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(n_filters, conv_size, conv_size, border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=pool_size))
#         Dropout(dropout1)
        
        #FCN
        model.add(Convolution2D(n_filters*10, 8, 8, border_mode='valid'))
        model.add(Activation('relu'))
        model.add(Convolution2D(n_filters*10, 1, 1, border_mode='valid'))
        model.add(Activation('relu'))
#         model.add(Convolution2D(n_classes, 1, 1, border_mode='valid'))
#         model.add(Activation('relu'))
        
#         #Deconvolution
        model.add(Deconvolution2D(n_classes, 32, 32, output_shape=(None, n_classes, h, w), subsample=(4, 4), border_mode='valid'))
        model.add(Reshape((h*w, n_classes)))
#         model.add(Activation('relu'))
        
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.001, decay=2e-4, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        print('compile model complete.')
        return model
    
    
    def fit_save_model(self, model, x, y):
        model.fit(x, y, batch_size=batch_size, nb_epoch=n_epoch, verbose=1)
        print('fit model complete')
        json_string=model.to_json()
        open(jsonfile_name, 'w').write(json_string)
        model.save_weights(weightsfile_name)
        return model
        
    def load_model_weights(self):
        model=model_from_json(open(jsonfile_name).read())
        model.load_weights(weightsfile_name)
        return model
        
    
    def pixelwise_prediction(self, model):
        test_im=imread(testdir)
        test_im=test_im.astype('float32')
        test_im=test_im.reshape(1, 3, h, w)
        test_im /=255
        y_pred=model.predict(test_im)
        return y_pred
        
        
        
        
        
        
        
        
    
    
    