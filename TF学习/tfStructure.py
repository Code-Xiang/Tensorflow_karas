from pickletools import optimize
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3


### create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化这个结构，让这个结构活动起来
init = tf.initialize_all_variables()

### create tensorflow structure start ###

### ****** ###
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))