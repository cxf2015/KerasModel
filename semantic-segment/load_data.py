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
from scipy.misc import imread, imsave

h=32
w=32
n_classes=8
imdir='/home/frankzhan/Dataset/JPEGImages'                                           #image dir has been resized
segdir='/home/frankzhan/Dataset/SegmentationClass'                           #segment dir has been resized
imdir_resize='/home/frankzhan/Dataset2007/JPEGImages'                    #original image dir to be resize
segdir_resize='/home/frankzhan/Dataset2007/SegmentationClass'    #original segment dir to be resize 
colors_to_classes={(128, 0, 0): 0, 
                                     (0, 128, 0): 1, 
                                     (0, 128, 128): 2, 
#                                      (128, 192, 0): 2,
                                     (128, 128, 128): 3,
                                     (64, 128, 128): 4, 
                                     (192, 128, 128): 5, 
                                     (64, 0, 0): 6, 
                                     (64, 128, 0): 6, 
                                     (64, 0, 128): 6, 
                                     (192, 0, 128): 6, 
                                     (128, 64, 0): 6, 
                                     (0, 0, 0): 7,
#                                      (128, 128, 0): 7,
#                                      (0, 0, 128): 7,
#                                      (128, 0, 128): 7,
#                                      (192, 0, 0): 7,
#                                      (192, 128, 0): 7,
#                                      (0, 64, 0): 7,
#                                      (0, 192, 0): 7,
#                                      (0, 64, 128): 7
                                     }

class Data():  
    
    def load_data(self):
        print('start loading data')
        im_files=os.listdir(imdir)       
        seg_files=os.listdir(segdir)
        im_files.sort()
        seg_files.sort()
        num_im=len(im_files)
        num_seg=len(seg_files)
        im_data = np.zeros((num_im, h, w, 3), dtype=np.uint8)
        seg_data = np.zeros((num_seg, h, w, 3), dtype=np.uint8)
        for idx,im_file in enumerate(im_files):
            im=imread('{}/{}'.format(imdir,im_file))
            im_data[idx]=im[:h,:w,:3]    #data of the image
        for idx,seg_file in enumerate(seg_files):
            seg=imread('{}/{}'.format(segdir,seg_file))
            seg_data[idx]=seg[:h,:w,:3]    #data of segment
        print('load im,seg, complete')
        
        seg_data_onehot = np.zeros((num_seg, h, w, n_classes), dtype=np.float32)
        for rgb_color in colors_to_classes.keys():
            color_true=np.logical_and(
                seg_data[:, :, :, 0]==rgb_color[0],
                seg_data[:, :, :, 1]==rgb_color[1],
                seg_data[:, :, :, 2]==rgb_color[2]) 
            locs=np.where(color_true)
            seg_data_onehot[locs[0], locs[1], locs[2], colors_to_classes[rgb_color]]=1
        im_data=im_data.astype('float32')
        seg_data_onehot=seg_data_onehot.astype('float32')
        im_data /=255
        im_data=im_data.reshape((num_im, 3, h, w))
        seg_data_onehot=seg_data_onehot.reshape(num_seg, h*w, n_classes)
        print('load data complete')
        return im_data, seg_data_onehot
            
#         print('**********')
#         a=seg_data_onehot[0]
#         b=a[10:20, 10:20, 0]
#         c =a[10:20, 10:20, 7]
#         print(b)
#         print('***********')
#         print(c)
#         print('***********')
#         for i in range(10):
#             a=seg_data_onehot[i]
#             b=np.sum(a)
#             print(b)
        



    def change_imsize(self): 
        size=(h,w)
        im_files=os.listdir(imdir_resize)
        seg_files=os.listdir(segdir_resize)
        im_files.sort()
        seg_files.sort()
        num_im=len(im_files)
        num_seg=len(seg_files)
        for idx,im_file in enumerate(im_files):            
            im=Image.open('{}/{}'.format(imdir_resize,im_file))
            im=im.resize(size)
            im.save('{}/{}result.jpg'.format(imdir,idx))
        for idx,seg_file in enumerate(seg_files):
            seg=Image.open('{}/{}'.format(segdir_resize,seg_file))
            seg=seg.resize(size)
            seg.save('{}/{}result.png'.format(segdir,idx))
          

# # 0: buildings; 1: water; 2: road, 3: background
# colors_to_classes = {(233, 229, 220): 3, (0, 0, 255): 1,
#                           (0, 255, 0): 2, (242, 240, 233): 0}
# classes_to_colors = {3: (233, 229, 220), 1: (0, 0, 255), 
#                           2: (0, 255, 0), 0: (242, 240, 233)}           
             
# Person: person[192 128 128]
# Animal: bird, cat, cow, dog, horse, sheep
# Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
# Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

# person[192 128 128] bus[0 128 128] train[128 192 0] car[128 128 128] 
# motorbike[64 128 128] bicycle[0 128 0] 边缘白[224 224 192] background[0 0 0]

# b-ground     0     0     0
# aero plane   128     0     0
# bicycle     0   128     0
# bird   128   128     0
# boat     0     0   128
# bottle   128     0   128
# bus     0   128   128
# car   128   128   128
# cat    64     0     0
# chair   192     0     0
# cow    64   128     0
# dining-table   192   128     0
# dog    64     0   128
# horse   192     0   128
# motorbike    64   128   128
# person   192   128   128
# potted-plant     0    64     0
# sheep   128    64     0
# sofa     0   192     0
# train   128   192     0
# tv/monitor     0    64   128

# 1.plane   128     0     0
# 2.bicycle     0   128     0
# 3.bus     0   128   128,   128   192     0
# 4.car   128   128   128
# 5.motorbike    64   128   128
# 6.person   192   128   128
# 7.animal   64     0     0,    64   128     0,    64     0   128,    192     0   128,   128    64     0      
# 8.b-ground     0     0     0,128   128     0,0     0   128,128     0   128,192     0     0,192   128     0,0    64     0,0   192     0,0    64   128







