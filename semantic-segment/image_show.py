# -*- coding: UTF-8 -*-
'''
Created on 2017年1月2日

@author: frankzhan
'''
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image,ImageFilter
from pylab import*
from scipy.misc import imread, imsave

h=32
w=32
n_classes=8
classes_to_colors={0: (128, 0, 0), 
                                     1: (0, 128, 0), 
                                     2: (0, 128, 128), 
                                     3: (128, 128, 128),
                                     4: (64, 128, 128), 
                                     5: (192, 128, 128), 
                                     6: (64, 0, 0), 
                                     7: (0, 0, 0)
                                     }

class Img():
    
    def show_img(self, y):
        y=y.reshape(h, w, n_classes)
        print(y)
        pixelwise_prediction=np.argmax(y[:, :, :], axis=2)
        pixelwise_color=np.zeros((h, w, 3))
        for class_num in range(n_classes):
            class_color=classes_to_colors[class_num]
            class_locs=np.where(pixelwise_prediction==class_num)
            x_locs=class_locs[0]
            y_locs=class_locs[1]
            for rgb_idx in range(3):
                pixelwise_color[x_locs, y_locs, rgb_idx] = class_color[rgb_idx]
        
        imshow(pixelwise_color)
        show()
        
        
        
        
        
        
        