from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    # %s会转换成字符串
    layer_name = 'layer%s'% n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            # 观看他的变量变化，加上下面这一行
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

# x_data 定义-1到1之间300个维数
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
# 噪点，没有完全按照他的二元一次方程走，在它附近有一些点 ,方差是0.05，后面是格式
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise 
#None代表这个列和行不定
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
# add_layer(inputs, in_size, out_size, activation_function)
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
# 定义输出层 隐藏层就是他的input_size,他的output_size就是y_data的size 1,activation_function=None所以他是线性函数
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)
# reduction_indices表示结果压缩的方向
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]),name='loss')
    # 在EVENT中查看机器学习学习到的东西，曲线不断减小，说明神经网络是有学习到东西的
    tf.summary.scalar('loss',loss)
# GradientDescentOptimizer需要给他一个learning_rate
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()
# if using tensorflow >= 0.12
sess = tf.Session()
# 把所有summary合并到一起
merged=tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
# plot the real data
# fig = plt.figure()
# # 1,1,1表示长宽，和位置
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        # sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        # lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        # plt.pause(0.1)

