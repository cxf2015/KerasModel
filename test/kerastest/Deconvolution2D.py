'''
Created on 2017年1月2日

@author: frankzhan
'''
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D

# Deconvolution2D(nb_filter, nb_row, nb_col, output_shape, 
#                                        init='glorot_uniform', activation='linear',
#                                        weights=None, border_mode='valid', 
#                                        subsample=(1, 1), dim_ordering='tf', 
#                                        W_regularizer=None, b_regularizer=None, 
#                                        activity_regularizer=None, W_constraint=None, 
#                                        b_constraint=None, bias=True)

# 当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (3,128,128)代表128*128的彩色RGB图像

# nb_filter：卷积核的数目  
# nb_row：卷积核的行数
# nb_col：卷积核的列数
# output_shape：反卷积的输出shape，为整数的tuple，形如（nb_samples,nb_filter,nb_output_rows,nb_output_cols），
# 计算output_shape的公式是：o = s (i - 1) + a + k - 2p,其中a的取值范围是0~s-1，其中：
#     i:输入的size（rows或cols）
#     k：卷积核大小（nb_filter）
#     s: 步长（subsample）
#     a：用户指定的的用于区别s个不同的可能output size的参数
# init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递weights参数时有意义。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，
# 将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

# weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。
# border_mode：边界模式，为“valid”，“same”，或“full”，full需要以theano为后端
# subsample：长为2的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”
# W_regularizer：施加在权重上的正则项，为WeightRegularizer对象
# b_regularizer：施加在偏置向量上的正则项，为WeightRegularizer对象
# activity_regularizer：施加在输出上的正则项，为ActivityRegularizer对象
# W_constraints：施加在权重上的约束项，为Constraints对象
# b_constraints：施加在偏置上的约束项，为Constraints对象

# dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。
# 例如128*128的三通道彩色图片，在‘th’模式中input_shape应写为（3，128，128），
# 而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为input_shape不包含样本数的维度，在其内部实现中，
# 实际上是（None，3，128，128）和（None，128，128，3）。默认是image_dim_ordering指定的模式，可在~/.keras/keras.json中查看，若没有设置过则为'tf'。

# bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

# apply a 3x3 transposed convolution with stride 1x1 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), border_mode='valid', input_shape=(3, 12, 12)))
# output_shape will be (None, 3, 14, 14)

# apply a 3x3 transposed convolution with stride 2x2 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25), subsample=(2, 2), border_mode='valid', input_shape=(3, 12, 12)))
model.summary()
# output_shape will be (None, 3, 25, 25)











