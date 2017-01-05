# -*- coding:utf-8 -*-  
'''
Created on 2016年12月4日

@author: frankzhan
'''
import numpy as np
import random

class Network(object):
    
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        
# sizes=[2,3,1]
# print(sizes[1:])
# bias=[np.random.randn(y,1) for y in sizes[1:]]
# print("bias",bias)
# for x, y in zip(sizes[:-1], sizes[1:]):
#     print(x,y)
# np.random.rand(y,1)    #随机正态分布
# net.weights[1]    #存储链接第二层和第三层的权重（python索引从0开始数）
  
net=Network([2,3,1])
print(net.num_layers)
print(net.sizes)
print(net.biases)
print(net.weights)

def feedforward(self, a):
    for b,w in zip(self.biases, self.weights):
        a=sigmoid(np.dot(w,a)+b) 
    return a

#随机梯度算法    
# def SGD(self, training_data, epochs,mini_batch_size,eta,test_data=None):
    
