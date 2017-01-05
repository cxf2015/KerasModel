# -*- coding:utf-8 -*- 
'''
Created on 2016年12月12日

@author: frankzhan
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

model = Sequential([
Dense(32, input_dim=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])

# model.compile(loss='categorical_crossentropy', optimizer='sgd' ,metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=5)
# loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
# classes = model.predict_classes(X_test, batch_size=32)
# proba = model.predict_proba(X_test, batch_size=32)




# for a single-input model with 2 classes (binary):
model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))
# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)





