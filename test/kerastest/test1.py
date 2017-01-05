# -*- coding:utf-8 -*-  
'''
Created on 2016年12月12日

@author: frankzhan
'''
  
import numpy as np  
  
#np.random.seed(100)  
  
from keras.optimizers import SGD  
  
import os  
import  matplotlib.pyplot as plt  
  
import h5py  
from keras.models import Sequential  
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D  
from keras.layers.core import Dense,Dropout,Activation,Flatten  
from keras.layers.normalization import  BatchNormalization  
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop  
from keras.layers.advanced_activations import PReLU  
  
from  keras.callbacks import ModelCheckpoint,Callback  
  
  
class LossHistory(Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = []  
  
    def on_batch_end(self, batch, logs={}):  
        self.losses.append(logs.get('loss'))  
  
def Net_Mouth():  
    keras_model=Sequential()#单支线性网络模型  
    #卷积层输出的特征图为20个，卷积核大小为5*5  
    keras_model.add(Convolution2D(20, 5, 5,input_shape=(3, 60, 60)))#网络输入每张图片大小为3通道，60*60的图片。  
    #激活函数层  
    keras_model.add(Activation('relu'))  
    #最大池化层  
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))  
  
    #卷积层，特征图个数为40，卷积核大小为5*5  
    keras_model.add(Convolution2D(40, 5, 5))  
    keras_model.add(Activation('relu'))  
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))  
  
    keras_model.add(Convolution2D(60, 3, 3))  
    keras_model.add(Activation('relu'))  
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))  
  
    keras_model.add(Convolution2D(80, 3, 3))  
    keras_model.add(Activation('relu'))  
  
    #全连接展平  
    keras_model.add(Flatten())  
    #全连接层，神经元个数为1000  
    keras_model.add(Dense(1000))  
    keras_model.add(Activation('relu'))  
  
    keras_model.add(Dense(500))  
    keras_model.add(Activation('relu'))  
  
  
    keras_model.add(Dense(38))  
    keras_model.add(Activation('tanh'))  
  
    #采用adam算法进行迭代优化，损失函数采用均方误差计算公式  
    keras_model.compile(loss='mean_squared_error', optimizer='adam')  
    return keras_model  
  
keras_model=Net_Mouth()  
#用于保存验证集误差最小的参数，当验证集误差减少时，立马保存下来  
checkpointer =ModelCheckpoint(filepath="mouth.hdf5", verbose=1, save_best_only=True)  
history = LossHistory()  
#训练函数，对于cnn来说，网络的输入x是（nsamples,nchanels,height,width）  
#y的输入是（nsamples,output_dimension）  
x=(4,5,4,4)
y=(4,5)
keras_model.fit(x, y, batch_size=128, nb_epoch=100,shuffle=True,verbose=2,show_accuracy=True,validation_split=0.1,callbacks=[checkpointer,history])




 