from __future__ import print_function
from pyexpat.model import XML_CTYPE_MIXED
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data  1-9 digits Data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

def weight_variable(shape):
    # 从截断正态分布输出随机值。 shape
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # stride [1,x_movement,y_movement, 1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])  # 28x28
ys = tf.placeholder(tf.float32,[None,10])
#-1可以理解为导入数据有多少个图片
keep_prob = tf.placeholder(xs,[-1,28,28,1])
x_image = tf.reshape(xs,[-1,28,28,1])  
# print(x_image.shape) # [n_samples,28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5×5,in_size 1,out_size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # out_put size 28×28×32
h_pool1 = max_pool_2x2(h_conv1)
## conv2 layer ##
W_conv1 = weight_variable([5,5,32,64]) # patch 5×5,in_size 1,out_size 64
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # out_put size 28×28×32
h_pool1 = max_pool_2x2(h_conv1)
## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# [n_sample, 7, 7, 64]->[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool1,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
## fun2 layer  ##
