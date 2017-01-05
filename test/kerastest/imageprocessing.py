# -*- coding:utf-8 -*-
'''
Created on 2016��10��7��

@author: Lenovo
'''

import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image,ImageFilter
from pylab import*

#格式转换
# datapath = r'D:\computer vision'
# filelist =  os.listdir(datapath)
# for infile in filelist:
#     f,e=os.path.splitext(infile)
#     outfile=f+".jpg"
#     if infile!=outfile:
#         try:
#             print(outfile)
#             inroute='D:/computer vision/'+infile
#             outroute='D:/computer vision/'+outfile
#             img=Image.open(inroute).save(outroute)
#         except IOError:
#             print ("cannot convert"),infile

#缩小                        
# im=Image.open('D:/computer vision/james.jpg')
# print(im.format,im.size,im.mode)
# w,h=im.size
# im.thumbnail((w//2,h//2))
# print(im.format,im.size,im.mode)
# im.save('D:/computer vision/thumbnail.jpg','jpeg')

#模糊
# im=Image.open('D:/computer vision/james.jpg')
# im2=im.filter(ImageFilter.BLUR)
# im2.save('D:/computer vision/blur1.jpg','jpeg')

#绘制图像，点，线
# im=array(Image.open('D:/computer vision/james.jpg'))
# imshow(im)
# x=[100,100,400,400]
# y=[200,500,200,500]
# plot(x,y,'r*')
# plot(x[:2],y[:2])
# show()
# t=np.arange(-1,2,.01)
# s=np.sinc(2*np.pi*t)
# plt.plot(t,s)
# plt.show()

#图像轮廓和直方图
# im=array(Image.open('D:/computer vision/james.jpg').convert('L'))
# figure()
# gray()
# contour(im,origin='image')
# axis('equal')
# axis('off')
# figure()
# hist(im.flatten(),128)
# show()

#交互式标注
# im=array(Image.open('D:/computer vision/james.jpg'))
# imshow(im)
# print("please click 3 points")
# x=ginput(3)
# print("you click:",x)
# show()

#图像数组表示
#行，列，颜色通道
# im=array(Image.open('D:/computer vision/james.jpg'))
# print(im.shape,im.dtype)
# im=array(Image.open('D:/computer vision/james.jpg').convert('L'),'f')
# print(im.shape,im.dtype)
# value=im[i,j,k]     #位于坐标i，j，以及颜色通道k的像素值
# im[i,:]=im[j,:]     #将第j行的数值赋值给第i行
# im[:,i]=100         #将第i列的所有数值设为100
# im[:100,:50].sum()  #计算前100行，前50列所有数值的和
# im[50:100,50:100]   #50~100行，50~100列（不包括第100行和第100列）
# im[i].mean()        #第i行所有数值的平均值
# im[:,-1]            #最后一列
# im[-2,:]            #or im[-2]，倒数第二行 

# #灰度变换
# im=array(Image.open('D:/computer vision/james.jpg').convert('L'))
# gray()
# im2=255-im                  #对图像进行反相处理
# im3=(100.0/255)*im+100      #将图像像素值变换到100~200        
# im4=255.0*(im/255.0)**2     #对图像像素值求平方后得到的图像
# imshow(im)
# # figure()
# # imshow(im2)
# # figure()
# # imshow(im3)
# # figure()
# # imshow(im4)
# show()

a=[192, 0, 0]
b=np.zeros((100, 100, 3), dtype=np.uint8)
for i in range(3):
    b[:, :, i]=a[i]
imshow(b)
show()
    












