import numpy as np
# numpy.array
exam = np.array([[1,2,3],[4,5,6]])
# print(exam)

#定义结构化数据
Stu = np.dtype([('name','S20'),('age','i1')])
# print(Stu)
stu_exam = np.array([('Zhangsan',18),('Wuji',28)],dtype= Stu)
# print(stu_exam)

# ===================================================================
# 举例
a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print('a[0:2]:::',a[0:2])
# print('a[0:2][0]:::',a[0:2][1])

# print(a[0][2])  # 先行后列

# =====================================================================
#计算 
#数学函数
#1、NumPy 提供了标准的三角函数： sin()、cos()、tan()\
#2、舍入函数：numpy.around()函数返回指定数字的四舍五入值。

a2 = np.array([1.0, 5.55, 123, 0.567, 25.532])
# print(np.around(a))
# [  1.   6. 123.   1.  26.]

# numpy.floor()返回小于或等于指定表达式的最大整数，及向下取整.
# print(np.floor(a))
# [  1.   5. 123.   0.  25.]

#numpy.ceil()返回大于或者指定表达式的最小整数，即向上取整。
# print(np.ceil(a))
# [  1.   6. 123.   1.  26.]

# ====================================================================
# https://www.runoob.com/numpy/numpy-statistical-functions.html
# numpy.amin() 用于计算数组中的元素沿指定轴的最小值。
# numpy.amax() 用于计算数组中的元素沿指定轴的最大值。
# numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值)。
# numpy.percentile() 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。 函数numpy.percentile()接受以下参数。
# numpy.median() 函数用于计算数组 a 中元素的中位数（中值）
# numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
a3 = np.array([[3,7,5],[8,4,3],[2,4,9]])
# print ('调用 amin() 函数：')
# print (np.amin(a3,1)) # [3 3 2]
# print ('\n')
# print ('再次调用 amin() 函数：') 
# print (np.amin(a3,axis = 0)) # [2 4 3]
# print ('\n')
# print ('调用 amax() 函数：')
# print (np.amax(a3)) # 9
# print ('\n')
# print ('再次调用 amax() 函数：')
# print (np.amax(a3,0))  # axis = 0 相当于 0   # [8 7 9]


a4 = np.array([[3,7,5],[8,4,3],[2,4,9]])  
# print ('我们的数组是：')
# print (a4)
# print ('\n')
# print ('调用 ptp() 函数：')
# print (np.ptp(a4))
# print ('\n')
# print ('沿轴 1 调用 ptp() 函数：')
# print (np.ptp(a4, axis =  1))
# print ('\n')
# print ('沿轴 0 调用 ptp() 函数：')
# print (np.ptp(a4, axis =  0))

# numpy.percentile() 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。函数numpy.percentile()接受以下参数。
# numpy.percentile(a, q, axis)
# a: 输入数组
# q: 要计算的百分位数，在 0 ~ 100 之间
# axis: 沿着它计算百分位数的轴
# 第 p 个百分位数是这样一个值，它使得至少有 p% 的数据项小于或等于这个值，且至少有 (100-p)% 的数据项大于或等于这个值。
# 举个例子：高等院校的入学考试成绩经常以百分位数的形式报告。比如，假设某个考生在入学考试中的语文部分的原始分数为 54 分。
# 相对于参加同一考试的其他学生来说，他的成绩如何并不容易知道。但是如果原始分数54分恰好对应的是第70百分位数，
# 我们就能知道大约70%的学生的考分比他低，而约30%的学生考分比他高。
# 这里的 p = 70。

a5 = np.array([[10, 7, 4], [3, 2, 1]])
# print ('调用 percentile() 函数：')
# # 50% 的分位数，就是 a5 里排序之后的中位数
# print (np.percentile(a5, 50)) # [2 4 3]
# # axis 为 0，在纵列上求
# print (np.percentile(a5, 50, axis=0)) # [6.5 4.5 2.5] 列的中位数
# # axis 为 1，在横行上求
# print (np.percentile(a5, 50, axis=1)) # [7. 2.] 行的中位数
# # 保持维度不变
# print (np.percentile(a5, 50, axis=1, keepdims=True))
# # 输出 [[7.]
#      [2.]]

# numpy.median() 用于计算数组 a 中元素的中位数（中值）
a6 = np.array([[30,65,70],[80,95,10],[50,90,60]])  
# 先将二维数组转换成一维数组，然后将其排序
# print(np.sort(a6.flatten())) 
# print ('调用 median() 函数：')
# print (np.median(a6))
# print ('\n')
# print ('沿轴 0 调用 median() 函数：')
# print (np.median(a6, axis =  0))
# print ('\n')
# print ('沿轴 1 调用 median() 函数：')
# print (np.median(a6, axis =  1))


# numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
# 算术平均值是沿轴的元素的总和除以元素的数量。
a7 = np.array([[1,2,3],[3,4,5],[4,5,6]])  
# print ('我们的数组是：')
# print (a7)
# print ('\n')
# print ('调用 mean() 函数：')
# print (np.mean(a7))
# print ('\n')
# print ('沿轴 0 调用 mean() 函数：')
# print (np.mean(a7, axis =  0))
# print ('\n')
# print ('沿轴 1 调用 mean() 函数：')
# print (np.mean(a7, axis =  1))

# numpy.average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。
# 该函数可以接受一个轴参数。 如果没有指定轴，则数组会被展开。
# 加权平均值即将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数。
# 考虑数组[1,2,3,4]和相应的权重[4,3,2,1]，通过将相应元素的乘积相加，并将和除以权重的和，来计算加权平均值。
# 加权平均值 = (1*4+2*3+3*2+4*1)/(4+3+2+1)
a8 = np.array([1,2,3,4])  
# print ('我们的数组是：')
# print (a8)
# print ('\n')
# print ('调用 average() 函数：')
# print (np.average(a8))
# print ('\n')
# # 不指定权重时相当于 mean 函数
# wts = np.array([4,3,2,1])  
# print ('再次调用 average() 函数：')
# print (np.average(a8,weights = wts))
# print ('\n')
# # 如果 returned 参数设为 true，则返回权重的和  
# print ('权重的和：')
# print (np.average([1,2,3,4],weights =  [4,3,2,1], returned =  True))

# 标准差是一组数据平均值分散程度的一种度量。
# 标准差是方差的算术平方根
# std = sqrt(mean((x - x.mean())**2))
# 例子： 如果数组是 [1，2，3，4]，则其平均值为 2.5。 因此，差的平方是 [2.25,0.25,0.25,2.25]，
# 并且再求其平均值的平方根除以 4，即 sqrt(5/4) ，结果为 1.1180339887498949。

# print (np.std([1,2,3,4]))

# 统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数，即 mean((x - x.mean())** 2)。
# 换句话说，标准差是方差的平方根。、
# print (np.var([1,2,3,4]))

# NumPy广播机制（Broacast）
# https://www.runoob.com/numpy/numpy-broadcast.html
a9 = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度
# 4表示复制4行，1表示横向*1倍，相当于原来的
# print('bb:\n',bb)
# print(a9 + bb)
# print('-----',a9 + b)
