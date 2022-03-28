import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
date = np.linspace(1,30,30)
beginPrice = np.array([2923.19,2928.06,2943.92,2946.26,2944.40,2920.85,2861.33,2854.58,2776.69,2789.02,
                       2784.18,2805.59,2781.98,2798.05,2824.49,2762.34,2817.57,2835.52,2879.08,2875.47,
                       2887.66,2885.15,2851.02,2879.52,2901.63,2896.00,2907.38,2886.94,2925.94,2927.75])
endPrice = np.array([2937.36,2944.54,2941.01,2952.34,2932.51,2908.77,2867.84,2821.50,2777.56,2768.68,
                     2794.55,2774.75,2814.99,2797.26,2808.91,2815.80,2823.82,2883.10,2880.00,2880.33,
                     2883.44,2897.43,2863.57,2902.19,2893.76,2890.92,2886.24,2924.11,2930.15,2957.41])

for i in range(0,30):  # 画柱状图
    dateOne = np.zeros([2])
    dateOne[0] = i
    dateOne[1] = i
    priceOne = np.zeros([2])
    priceOne[0] = beginPrice[i]
    priceOne[1] = endPrice[i]
    print('priceOne:\n',priceOne)
    if endPrice[i]>beginPrice[i]:
        plt.plot(dateOne,priceOne,'r',lw=6)
    else:
        plt.plot(dateOne,priceOne,'g',lw=6)
plt.xlabel("date")
plt.ylabel("price")
# plt.show()

dataNormal = np.zeros([30,1])
priceNormal =  np.zeros([30,1])
# 归一化
for i in range(0,30):
    dataNormal[i,0] = i/29.0
    priceNormal[i,0] = endPrice[i]/3000.0
x = tf.keras.Input([None,1],dtype='float32')
y = tf.keras.Input([None,1],dtype='float32')
# x->hidden_layer
wb1 = tf.Variable(tf.random_uniform_initializer([1,25],0,1))