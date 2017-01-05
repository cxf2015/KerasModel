# -*- coding: UTF-8 -*-
'''
Created on 2016年12月30日

@author: frankzhan
'''
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image,ImageFilter
from pylab import*
# from scipy.misc import imread, imsave

# im=Image.open('D:/computer vision/qing.jpg')
# print(im.format,im.size,im.mode)
# imarr=array(im)
# imshow(imarr)
# show()

# Person: person[192 128 128]
# Animal: bird, cat, cow, dog, horse, sheep
# Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
# Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

# person[192 128 128] bus[0 128 128] train[128 192 0] car[128 128 128] 
# motorbike[64 128 128] bicycle[0 128 0] 边缘白[224 224 192] background[0 0 0]

imdir='/home/frankzhan/Dataset/JPEGImages/2007_000333.jpg'
segdir='/home/frankzhan/Dataset/SegmentationClass/2007_000333.png'
im=imread(imdir)
print(im.shape,im.dtype)
imshow(im)
imdata=im[:300,:300,:3]
imdata[:50,:50]=[255,0,0]
figure()
imshow(imdata)
a=im[183,200]
b=im[1,1]
print(a,b)
show()






