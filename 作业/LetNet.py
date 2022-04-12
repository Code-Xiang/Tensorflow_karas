from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
x=tf.expand_dims(x,axis=3)
x_val=tf.expand_dims(x_val,axis=3)

db = tf.data.Dataset.from_tensor_slices((x,y))#构建Dataset对象，转换成Dataset对象，才能利用TensorFlow提供的各种便捷功能。
db = db.map(preprocess).shuffle(60000).batch(batchsz)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)
from tensorflow.keras import Sequential
network=Sequential([
    layers.Conv2D(6,kernel_size=3,strides=1),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Conv2D(16,kernel_size=3,strides=1),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Flatten(),#打平层
    layers.Dense(120,activation='relu'),
    layers.Dense(84,activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(4,28,28,1))
network.summary()


from tensorflow.keras import losses,optimizers

network.compile(optimizer=optimizers.Adam(lr=0.01),loss=losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=network.fit(db,epochs=5,validation_data=ds_val,validation_freq=2)#,validation_data=ds_val,validation_freq=2
print(history)
