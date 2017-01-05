# -*- coding: UTF-8 -*-
'''
Created on 2016年12月30日

@author: frankzhan
'''
from image_show import Img
from load_data import Data
from model import KerasModel
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image,ImageFilter
from pylab import*
from scipy.misc import imread, imsave

ld=Data()
# ld.change_imsize()     

x, y = ld.load_data()
km=KerasModel()
# model=km.load_model_weights()
model=km.compile_model()  
model=km.fit_save_model(model, x, y)
y_pre=km.pixelwise_prediction(model)
im=Img()
im.show_img(y_pre)








